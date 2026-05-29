import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layer.layers import CrossAttentionBlock
from layer.jscc_decoder import RateAdaptionDecoder


class JSCCDecoderCross(nn.Module):
    """Cross-attention JSCC Decoder: rate/SNR embeddings as KV, received tokens as Q."""

    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, rate_choice=[0, 128, 256],
                 input_resolution=(16, 16), depths=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))

        if depths is not None:
            depth = sum(depths)
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]

        num_patches = input_resolution[0] * input_resolution[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim=embed_dim, num_heads=num_heads,
                                mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.rate_adaption = RateAdaptionDecoder(embed_dim, rate_choice)

    def forward(self, x, indexes, cond):
        """
        Args:
            x: channel output (B, max_rate, H, W)
            indexes: rate indexes (B*H*W,)
            cond: conditioning tokens (B, K, D)
        Returns:
            y_hat (B, C, H, W)
        """
        B, _, H, W = x.size()
        x = self.rate_adaption(x, indexes)
        x_BLC = x.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.pos_embed[:, :H * W, :]
        for layer in self.layers:
            x_BLC = layer(x_BLC, cond)
        x_BLC = self.norm(x_BLC)
        x_BCHW = x_BLC.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        return x_BCHW

    def update_resolution(self, H, W):
        pass
