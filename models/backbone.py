# models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone with frozen BatchNorm.
    """

    def __init__(self, train_backbone=True, dilation=False):
        super().__init__()

        backbone = models.resnet50(pretrained=True)

        # Freeze BatchNorm layers
        for name, parameter in backbone.named_parameters():
            if 'bn' in name:
                parameter.requires_grad_(False)

        # Get the layers we want from ResNet
        return_layers = {'layer4': "0"}  # We only need the output of layer4

        if dilation:
            # Modify ResNet for dilation
            backbone.layer3.apply(lambda m: self._replace_stride_with_dilation(m))
            backbone.layer4.apply(lambda m: self._replace_stride_with_dilation(m))

        # Remove fully connected layer and pooling
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Set output channels
        self.num_channels = 2048

        # Freeze backbone if not training
        if not train_backbone:
            for name, parameter in self.body.named_parameters():
                parameter.requires_grad_(False)

    def _replace_stride_with_dilation(self, module):
        """Replace stride with dilation in the given module."""
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                child.stride = (1, 1)
                child.dilation = (2, 2)
                child.padding = (2, 2)

    def forward(self, x):
        """Forward pass of the backbone."""
        return self.body(x)
