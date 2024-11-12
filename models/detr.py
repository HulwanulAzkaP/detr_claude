# models/detr.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Simple multi-layer perceptron """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    DETR model for fire and smoke detection
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super().__init__()

        # Import components here to avoid circular imports
        from .backbone import ResNet50Backbone
        from .transformer import Transformer, PositionEmbeddingSine

        # Create ResNet backbone
        self.backbone = ResNet50Backbone()

        # Create position embedding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        # Create transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=dropout
        )

        # Create input projection layer
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        # Create query embeddings
        self.query_embed = nn.Embedding(100, hidden_dim)

        # Create output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples):
        """
        Forward pass of DETR.
        """
        # Extract features from backbone
        features = self.backbone(samples)["0"]

        # Create position embeddings
        pos = self.position_embedding(features)

        # Project features
        src = self.input_proj(features)

        # Flatten spatial dimensions and permute
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        # Forward pass through transformer
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        hs = self.transformer(src, query_embed, pos)

        # Predict classes and boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        return out