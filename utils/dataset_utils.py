# utils/dataset_utils.py
import os
import json
from pathlib import Path
import logging


def verify_dataset_structure(config):
    """
    Verify the dataset structure and files existence.

    Args:
        config: Configuration object containing dataset paths

    Returns:
        bool: True if dataset structure is valid, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Check main dataset directory
    if not os.path.exists(config.DATASET_DIR):
        logger.error(f"Dataset directory not found: {config.DATASET_DIR}")
        return False

    # Required splits
    splits = ['train', 'valid', 'test']

    for split in splits:
        split_dir = os.path.join(config.DATASET_DIR, split)

        # Check split directory
        if not os.path.exists(split_dir):
            logger.error(f"{split} directory not found: {split_dir}")
            return False

        # Check annotation file
        ann_file = os.path.join(split_dir, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            logger.error(f"Annotation file not found: {ann_file}")
            return False

        # Verify images from annotation file
        try:
            with open(ann_file, 'r') as f:
                coco = json.load(f)

            for img in coco['images']:
                img_path = os.path.join(split_dir, img['file_name'])
                if not os.path.exists(img_path):
                    logger.error(f"Image not found: {img_path}")
                    return False

            logger.info(f"Found {len(coco['images'])} images in {split} set")
            logger.info(f"Found {len(coco['categories'])} categories: {[cat['name'] for cat in coco['categories']]}")
            logger.info(f"Found {len(coco['annotations'])} annotations in {split} set")

        except Exception as e:
            logger.error(f"Error reading annotation file {ann_file}: {str(e)}")
            return False

    return True


def setup_dataset_structure(config):
    """
    Setup the dataset directory structure.

    Args:
        config: Configuration object
    """
    # Create main directories if they don't exist
    os.makedirs(config.DATASET_DIR, exist_ok=True)

    # Create split directories
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(config.DATASET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)


def print_dataset_info(config):
    """
    Print dataset information.

    Args:
        config: Configuration object
    """
    logger = logging.getLogger(__name__)

    for split in ['train', 'valid', 'test']:
        ann_file = os.path.join(config.DATASET_DIR, split, '_annotations.coco.json')
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                coco = json.load(f)

            logger.info(f"\n{split} set information:")
            logger.info(f"Number of images: {len(coco['images'])}")
            logger.info(f"Number of categories: {len(coco['categories'])}")
            logger.info(f"Categories: {[cat['name'] for cat in coco['categories']]}")
            logger.info(f"Number of annotations: {len(coco['annotations'])}")

            # Count annotations per category
            cat_counts = {}
            for ann in coco['annotations']:
                cat_id = ann['category_id']
                cat_name = next(cat['name'] for cat in coco['categories'] if cat['id'] == cat_id)
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

            logger.info("Annotations per category:")
            for cat_name, count in cat_counts.items():
                logger.info(f"  {cat_name}: {count}")