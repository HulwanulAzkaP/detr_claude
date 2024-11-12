# data/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.transforms import make_detr_transforms


class FireSmokeDataset(Dataset):
    """
    Dataset class for Fire and Smoke detection using COCO format from Roboflow
    Categories:
    - api-asap-R5wA (id: 0): general category
    - api (id: 1): fire
    - asap (id: 2): smoke
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the FireSmokeDataset.

        Args:
            root_dir (str): Root directory containing the dataset
            split (str): Dataset split ('train', 'valid', or 'test')
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform if transform else make_detr_transforms(split)

        # Load and parse the COCO format annotations
        self.ann_file = os.path.join(self.root_dir, '_annotations.coco.json')
        with open(self.ann_file, 'r') as f:
            self.coco = json.load(f)

        # Create image id to image data mapping
        self.img_dict = {img['id']: img for img in self.coco['images']}

        # Create category mapping
        self.cat_dict = {cat['id']: cat for cat in self.coco['categories']}
        self.cat_ids = sorted([cat['id'] for cat in self.coco['categories']])
        self.cat_id_to_continuous = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Create final image list
        self.img_ids = sorted(list(self.img_dict.keys()))

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of annotations
        """
        # Get image info
        img_id = self.img_ids[idx]
        img_info = self.img_dict[img_id]

        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Get image size
        w, h = img_info['width'], img_info['height']

        # Get annotations
        anno = self.img_to_anns.get(img_id, [])

        # Prepare target
        boxes = []
        classes = []
        area = []
        iscrowd = []

        for ann in anno:
            bbox = ann['bbox']  # [x, y, width, height] format

            # Convert to normalized center coordinates
            cx = (bbox[0] + bbox[2] / 2) / w
            cy = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            boxes.append([cx, cy, bw, bh])

            # Convert category id to continuous index
            cat_id = ann['category_id']
            classes.append(self.cat_id_to_continuous[cat_id])

            # Area and crowd flag
            area.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # Create target dictionary
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(classes, dtype=torch.long) if classes else torch.zeros(0, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor(area, dtype=torch.float32) if area else torch.zeros(0),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.uint8) if iscrowd else torch.zeros(0),
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([h, w])
        }

        # Apply transforms
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_ids)

    def get_img_info(self, idx):
        """Get image info by index."""
        img_id = self.img_ids[idx]
        return self.img_dict[img_id]

    @property
    def get_categories(self):
        """Get category mapping."""
        return {v: k for k, v in self.cat_id_to_continuous.items()}