# main.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from config.config import Config
from models.detr import DETR
from data.dataloader import build_dataloader
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import warnings

# Disable debug logs
logging.getLogger().setLevel(logging.INFO)

# Disable warnings
warnings.filterwarnings('ignore')

# Disable Torch debug output
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# Configure tqdm to be less verbose
from tqdm import tqdm
tqdm.monitor_interval = 0

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = config.OUTPUT_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'detr_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DETRTrainer:
    """DETR Trainer class for object detection."""

    def __init__(self, config, logger):
        """Initialize trainer with config and logger."""
        self.config = config
        self.logger = logger
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        # Setup data
        self.train_loader, self.num_classes = build_dataloader(config, 'train')
        self.val_loader, _ = build_dataloader(config, 'valid')

        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Training on device: {self.device}")

        # Initialize model
        self.model = DETR(
            num_classes=self.num_classes,
            hidden_dim=config.HIDDEN_DIM,
            nheads=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dropout=config.DROPOUT
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Initialize scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=0.1
        )

        # Initialize loss weights
        self.loss_weights = {
            'loss_ce': config.CLASS_LOSS_COEF,
            'loss_bbox': config.BBOX_LOSS_COEF,
            'loss_giou': config.GIOU_LOSS_COEF
        }

        # Initialize training state
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train(self):
        """Full training loop with consistent output format."""
        self.logger.info("Starting training...")

        for epoch in range(self.config.EPOCHS):
            # Training phase with progress bar
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_loss, val_metrics = self.validate()

            # Update learning rate
            self.lr_scheduler.step()

            # Print epoch results in consistent format
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}:")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                self.logger.info(f"Train {k}: {v:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                self.logger.info(f"Val {k}: {v:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')
                self.logger.info("New best model saved!")

            # Early stopping check
            self.early_stopping_counter = (self.early_stopping_counter + 1
                                           if val_loss >= self.best_val_loss else 0)

            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Regular checkpoint save
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')

    def train_epoch(self, epoch):
        """Train for one epoch with clean progress bar."""
        self.model.train()
        train_loss = 0.0
        train_metrics = {
            'loss_ce': 0.0,
            'loss_bbox': 0.0,
            'loss_giou': 0.0
        }

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}',
            ncols=120,
            position=0,
            leave=True
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = [{k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = self.model(inputs)

            # Calculate loss
            loss_dict = self.criterion(outputs, targets, inputs.shape[0])
            loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            if hasattr(self.config, 'CLIP_MAX_NORM'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_MAX_NORM)

            self.optimizer.step()

            # Update metrics
            train_loss += loss.item()
            for k, v in loss_dict.items():
                train_metrics[k] += v.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            }, refresh=True)

        # Close progress bar
        pbar.close()

        # Calculate average metrics
        num_batches = len(self.train_loader)
        train_loss /= num_batches
        train_metrics = {k: v / num_batches for k, v in train_metrics.items()}

        return train_loss, train_metrics

    def validate(self):
        """Validate with minimal output."""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'loss_ce': 0.0,
            'loss_bbox': 0.0,
            'loss_giou': 0.0
        }

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = [{k: v.to(self.device) if torch.is_tensor(v) else v
                            for k, v in t.items()} for t in targets]

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss_dict = self.criterion(outputs, targets, inputs.shape[0])
                loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items())

                # Update metrics
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_metrics[k] += v.item()

        # Calculate average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        return val_loss, val_metrics

    def criterion(self, outputs, targets, batch_size):
        """Calculate losses silently."""
        device = outputs['pred_logits'].device

        try:
            # Get matching indices using Hungarian matching
            indices = self.hungarian_matching(outputs, targets)

            # Initialize losses
            total_loss_ce = torch.tensor(0.0, device=device)
            total_loss_bbox = torch.tensor(0.0, device=device)
            total_loss_giou = torch.tensor(0.0, device=device)
            num_valid_batches = 0

            # Process each batch item silently
            for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
                if len(pred_idx) == 0:
                    continue

                # Get predictions and targets
                pred_logits = outputs['pred_logits'][batch_idx, pred_idx]
                pred_boxes = outputs['pred_boxes'][batch_idx, pred_idx]
                target_labels = targets[batch_idx]['labels'][tgt_idx]
                target_boxes = targets[batch_idx]['boxes'][tgt_idx]

                # Calculate losses
                loss_ce = nn.CrossEntropyLoss()(pred_logits, target_labels)
                loss_bbox = nn.L1Loss()(pred_boxes, target_boxes)
                loss_giou = (1 - torch.diag(generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_boxes),
                    box_cxcywh_to_xyxy(target_boxes)
                ))).mean()

                # Accumulate losses
                total_loss_ce += loss_ce
                total_loss_bbox += loss_bbox
                total_loss_giou += loss_giou
                num_valid_batches += 1

            # Average losses
            if num_valid_batches > 0:
                total_loss_ce /= num_valid_batches
                total_loss_bbox /= num_valid_batches
                total_loss_giou /= num_valid_batches

            return {
                'loss_ce': total_loss_ce,
                'loss_bbox': total_loss_bbox,
                'loss_giou': total_loss_giou
            }

        except Exception as e:
            return {
                'loss_ce': torch.tensor(0.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }

    def hungarian_matching(self, outputs, targets):
        """Perform Hungarian matching silently."""
        cost_matrices = self.get_hungarian_matching_loss(outputs, targets)
        indices = []

        for cost_matrix in cost_matrices:
            if cost_matrix.shape[1] == 0:
                indices.append(([], []))
                continue

            # Move to CPU for Hungarian algorithm
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            pred_indices, tgt_indices = linear_sum_assignment(cost_matrix_np)

            # Move back to device
            indices.append((
                torch.as_tensor(pred_indices, dtype=torch.long, device=outputs['pred_logits'].device),
                torch.as_tensor(tgt_indices, dtype=torch.long, device=outputs['pred_logits'].device)
            ))

        return indices

    def get_hungarian_matching_loss(self, outputs, targets):
        """
        Calculate matching cost matrix silently.

        Args:
            outputs (dict): Model outputs containing 'pred_logits' and 'pred_boxes'
            targets (list): List of target dictionaries

        Returns:
            torch.Tensor: Cost matrix for each batch
        """
        bs = outputs["pred_logits"].shape[0]
        num_queries = outputs["pred_logits"].shape[1]
        device = outputs["pred_logits"].device

        # Convert predictions to probabilities
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        # Initialize cost matrix for each batch
        cost_matrices = []

        # Process each batch item separately
        for batch_idx in range(min(bs, len(targets))):
            try:
                tgt_ids = targets[batch_idx]["labels"]
                tgt_bbox = targets[batch_idx]["boxes"]
                num_tgt = len(tgt_ids)

                if num_tgt == 0:
                    cost_matrices.append(torch.zeros((num_queries, 0), device=device))
                    continue

                # Classification cost
                cost_class = -out_prob[batch_idx, :, tgt_ids]

                # L1 box cost
                cost_bbox = torch.cdist(
                    out_bbox[batch_idx],
                    tgt_bbox,
                    p=1
                )

                # GIoU cost
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox[batch_idx]),
                    box_cxcywh_to_xyxy(tgt_bbox)
                )

                # Combine costs
                C = (self.loss_weights['loss_ce'] * cost_class +
                     self.loss_weights['loss_bbox'] * cost_bbox +
                     self.loss_weights['loss_giou'] * cost_giou)

                cost_matrices.append(C)

            except Exception as e:
                cost_matrices.append(torch.zeros((num_queries, 0), device=device))

        # If we have fewer cost matrices than batch size, pad with zeros
        while len(cost_matrices) < bs:
            cost_matrices.append(torch.zeros((num_queries, 0), device=device))

        return cost_matrices
    def compute_matched_loss(self, outputs, targets, indices):
        """Compute loss for matched predictions and targets."""
        device = outputs['pred_logits'].device

        # Initialize matched tensors
        src_logits_matched = []
        src_boxes_matched = []
        target_classes = []
        target_boxes = []

        # Process each batch item
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            src_logits_matched.append(outputs['pred_logits'][batch_idx, pred_idx])
            src_boxes_matched.append(outputs['pred_boxes'][batch_idx, pred_idx])
            target_classes.append(targets[batch_idx]['labels'][tgt_idx])
            target_boxes.append(targets[batch_idx]['boxes'][tgt_idx])

        # Skip if no matches
        if len(src_logits_matched) == 0:
            return {
                'loss_ce': torch.tensor(0.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }

        # Stack matched predictions and targets
        src_logits = torch.cat(src_logits_matched)
        src_boxes = torch.cat(src_boxes_matched)
        target_classes = torch.cat(target_classes)
        target_boxes = torch.cat(target_boxes)

        # Calculate losses
        loss_ce = nn.CrossEntropyLoss()(src_logits, target_classes)
        loss_bbox = nn.L1Loss()(src_boxes, target_boxes)
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))).mean()

        return {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }

    def save_model(self, filename):
        """Save model checkpoint."""
        save_path = self.config.SAVED_MODELS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'best_val_loss': self.best_val_loss
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, filename):
        """Load model checkpoint."""
        load_path = self.config.SAVED_MODELS_DIR / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.info(f"Model loaded from {load_path}")

def main():
    parser = argparse.ArgumentParser(description='DETR Training Script')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'inference', 'save'],
                        help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    if args.config_path:
        config = Config.load(args.config_path)
    else:
        config = Config()

    # Setup logging
    logger = setup_logging(config)

    # Verify dataset structure
    logger.info("Verifying dataset structure...")
    from utils.dataset_utils import verify_dataset_structure, print_dataset_info

    if not verify_dataset_structure(config):
        logger.error("Dataset verification failed. Please check the dataset structure and files.")
        return

    # Print dataset information
    print_dataset_info(config)

    # Create trainer
    trainer = DETRTrainer(config, logger)

    # Load model if specified
    if args.model_path:
        trainer.load_model(args.model_path)

    # Execute specified mode
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        val_loss, val_metrics = trainer.validate()
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info("Validation Metrics:")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")
    elif args.mode == 'inference':
        trainer.inference()
    elif args.mode == 'save':
        if args.model_path is None:
            raise ValueError("Model path must be specified for save mode")
        trainer.save_model(args.model_path)


if __name__ == '__main__':
    main()