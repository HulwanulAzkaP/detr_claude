# models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position embedding for DETR.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x):
        """
        Create positional embeddings for the input tensor.

        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Position embeddings [batch_size, pos_feat*2, height, width]
        """
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                           device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for encoder layer.
        """
        # Self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for decoder layer.
        """
        # Self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class Transformer(nn.Module):
    """
    Full transformer encoder-decoder architecture.
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Create encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])

        # Create decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        """
        Forward pass of transformer.

        Args:
            src (torch.Tensor): Source tokens [sequence_length, batch_size, embedding_dim]
            query_embed (torch.Tensor): Object queries [num_queries, batch_size, embedding_dim]
            pos_embed (torch.Tensor): Position embeddings [sequence_length, batch_size, embedding_dim]

        Returns:
            torch.Tensor: Decoder output for all layers [num_layers, batch_size, num_queries, embedding_dim]
        """
        # Add positional embeddings to source
        src = src + pos_embed

        # Encoder
        memory = src
        for layer in self.encoder:
            memory = layer(memory)

        # Decoder
        tgt = torch.zeros_like(query_embed)
        output = tgt

        intermediate = []
        for layer in self.decoder:
            output = layer(output, memory)
            intermediate.append(output)

        return torch.stack(intermediate)