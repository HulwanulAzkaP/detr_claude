# data/dataloader.py
import torch
from torch.utils.data import DataLoader
from .dataset import FireSmokeDataset
from utils.transforms import make_detr_transforms


# In data/dataloader.py

def collate_fn(batch):
    """
    Custom collate function for DETR data.

    Args:
        batch: List of tuples (image, target)

    Returns:
        tuple: (images, targets) where:
            - images is a tensor of shape [batch_size, C, H, W]
            - targets is a list of dictionaries
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)

        # Ensure all required keys exist
        processed_target = {
            'labels': target.get('labels', torch.tensor([])),
            'boxes': target.get('boxes', torch.tensor([])),
            'image_id': target.get('image_id', torch.tensor([0])),
            'area': target.get('area', torch.tensor([])),
            'iscrowd': target.get('iscrowd', torch.tensor([]))
        }
        targets.append(processed_target)

    # Stack images
    images = torch.stack(images)

    return images, targets


def build_dataloader(config, split='train'):
    """
    Build data loader for DETR.

    Args:
        config: Configuration object
        split: Dataset split ('train', 'valid', or 'test')

    Returns:
        tuple: (dataloader, num_classes)
    """
    try:
        dataset = FireSmokeDataset(
            root_dir=config.DATASET_DIR,
            split=split,
            transform=make_detr_transforms(split)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE if split == 'train' else 1,
            shuffle=split == 'train',
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=config.PIN_MEMORY,
            drop_last=split == 'train'
        )

        return dataloader, len(dataset.cat_ids)

    except Exception as e:
        print(f"Error building dataloader for {split} split: {str(e)}")
        raise