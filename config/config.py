# config/config.py
import os
from pathlib import Path


class Config:
    """
    Configuration class for DETR Fire-Smoke Detection
    """

    def __init__(self):
        # Project structure paths
        self.ROOT_DIR = Path(__file__).parent.parent
        self.DATASET_DIR = self.ROOT_DIR / 'dataset'
        self.OUTPUT_DIR = self.ROOT_DIR / 'output'
        self.SAVED_MODELS_DIR = self.OUTPUT_DIR / 'saved_models'
        self.PREDICTIONS_DIR = self.OUTPUT_DIR / 'predictions'
        self.METRICS_DIR = self.OUTPUT_DIR / 'metrics'

        # Create directories if they don't exist
        self.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)

        # Model parameters
        self.HIDDEN_DIM = 256  # Transformer hidden dimension
        self.NUM_QUERIES = 100  # Number of object queries
        self.DROPOUT = 0.1  # Dropout rate
        self.NUM_HEADS = 8  # Number of attention heads
        self.NUM_ENCODER_LAYERS = 6  # Number of encoder layers
        self.NUM_DECODER_LAYERS = 6  # Number of decoder layers
        self.DIM_FEEDFORWARD = 2048  # Dimension of feedforward network

        # Training parameters
        self.BATCH_SIZE = 16  # Batch size for training
        self.LEARNING_RATE = 1e-4  # Initial learning rate
        self.WEIGHT_DECAY = 1e-4  # Weight decay for regularization
        self.EPOCHS = 300  # Number of training epochs
        self.CLIP_MAX_NORM = 0.1  # Gradient clipping max norm
        self.LR_DROP = 200  # Learning rate drop epoch
        self.SCHEDULER_STEP_SIZE = 100  # LR scheduler step size

        # Dataset parameters
        self.IMG_SIZE = 640  # Input image size
        self.NUM_WORKERS = 4  # Number of workers for data loading
        self.PIN_MEMORY = True  # Pin memory for faster data transfer

        # Device configuration
        self.DEVICE = 'cuda'  # Use 'cuda' or 'cpu'

        # Model specific parameters
        self.POSITION_EMBEDDING = 'sine'  # Position embedding type
        self.ENC_LAYERS = 6  # Number of encoder layers
        self.DEC_LAYERS = 6  # Number of decoder layers
        self.PRE_NORM = False  # Whether to use pre-normalization

        # Loss parameters
        self.AUX_LOSS = True  # Whether to use auxiliary decoding losses
        self.BBOX_LOSS_COEF = 5  # Coefficient for bbox loss
        self.GIOU_LOSS_COEF = 2  # Coefficient for giou loss
        self.CLASS_LOSS_COEF = 2  # Coefficient for classification loss
        self.MASK_LOSS_COEF = 1  # Coefficient for mask loss
        self.DICE_LOSS_COEF = 1  # Coefficient for dice loss

        # Matcher parameters
        self.SET_COST_CLASS = 1  # Class coefficient in the matching cost
        self.SET_COST_BBOX = 5  # L1 box coefficient in the matching cost
        self.SET_COST_GIOU = 2  # giou box coefficient in the matching cost

        # Dataset specific parameters
        self.CLASSES = {
            0: 'api-asap-R5wA',  # General category
            1: 'api',  # Fire
            2: 'asap'  # Smoke
        }
        self.NUM_CLASSES = len(self.CLASSES)  # Number of classes including background

        # Inference parameters
        self.CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detection
        self.NMS_THRESHOLD = 0.5  # Non-maximum suppression threshold

        # Validation parameters
        self.VALIDATION_FREQUENCY = 1  # Validate every N epochs

        # Checkpoint parameters
        self.SAVE_FREQUENCY = 10  # Save checkpoint every N epochs
        self.CHECKPOINT_PATH = self.SAVED_MODELS_DIR / 'checkpoint.pth'
        self.BEST_MODEL_PATH = self.SAVED_MODELS_DIR / 'best_model.pth'

        # Logging parameters
        self.LOG_FREQUENCY = 10  # Log every N batches

        # Data augmentation parameters
        self.TRAIN_TRANSFORMS = {
            'HORIZONTAL_FLIP_PROB': 0.5,
            'VERTICAL_FLIP_PROB': 0.0,
            'RANDOM_BRIGHTNESS': 0.2,
            'RANDOM_CONTRAST': 0.2,
            'RANDOM_SATURATION': 0.2,
            'RANDOM_HUE': 0.2,
            'SCALE_MIN': 0.8,
            'SCALE_MAX': 1.2,
        }

        # Early stopping parameters
        self.EARLY_STOPPING_PATIENCE = 20  # Number of epochs to wait for improvement
        self.EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum change to qualify as an improvement

        # Visualization parameters
        self.VISUALIZATION = {
            'BBOX_COLOR': (255, 0, 0),  # Red for bounding boxes
            'FONT_SIZE': 1,
            'THICKNESS': 2,
            'FONT_FACE': 'FONT_HERSHEY_SIMPLEX'
        }

    def update(self, config_dict):
        """
        Update configuration parameters from a dictionary
        Args:
            config_dict (dict): Dictionary containing configuration updates
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")

    def save(self, path):
        """
        Save configuration to a file
        Args:
            path (str): Path to save configuration
        """
        import json

        config_dict = {k: v for k, v in self.__dict__.items()
                       if not k.startswith('__') and not callable(v)}

        # Convert Path objects to strings
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        """
        Load configuration from a file
        Args:
            path (str): Path to configuration file
        Returns:
            Config: Configuration object
        """
        import json

        config = cls()
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Convert string paths back to Path objects
        path_keys = ['ROOT_DIR', 'DATASET_DIR', 'OUTPUT_DIR',
                     'SAVED_MODELS_DIR', 'PREDICTIONS_DIR', 'METRICS_DIR',
                     'CHECKPOINT_PATH', 'BEST_MODEL_PATH']

        for k, v in config_dict.items():
            if k in path_keys:
                config_dict[k] = Path(v)

        config.update(config_dict)
        return config

    def __str__(self):
        """String representation of the configuration"""
        config_str = "DETR Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(value):
                config_str += f"{key}: {value}\n"
        return config_str