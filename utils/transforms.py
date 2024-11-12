# utils/transforms.py
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import random


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Randomly horizontally flips the image and target with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            # Flip image
            image = F.hflip(image)

            if "boxes" in target:
                boxes = target["boxes"]
                boxes = torch.as_tensor(boxes)
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return image, target


class ToTensor:
    """Convert PIL image and target to tensor."""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std."""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image to a given size."""

    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)

    def __call__(self, image, target):
        # Original size
        orig_size = image.size

        # Resize image
        image = F.resize(image, self.size)

        if "boxes" in target:
            # Compute scale factors
            scale_w = self.size[0] / orig_size[0]
            scale_h = self.size[1] / orig_size[1]

            # Scale boxes
            boxes = target["boxes"]
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_w
            scaled_boxes[:, [1, 3]] *= scale_h
            target["boxes"] = scaled_boxes

            # Update size
            target["size"] = torch.tensor(self.size)

        return image, target


def make_detr_transforms(split):
    """
    Create transforms for DETR.

    Args:
        split (str): Dataset split ('train', 'valid', or 'test')

    Returns:
        Compose: Composed transforms
    """
    normalize = Normalize()
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if split == 'train':
        return Compose([
            RandomHorizontalFlip(),
            Resize(640),
            ToTensor(),
            normalize,
        ])

    if split in ['valid', 'test']:
        return Compose([
            Resize(640),
            ToTensor(),
            normalize,
        ])

    raise ValueError(f'Unknown split {split}')