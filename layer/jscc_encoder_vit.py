import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layer.layers import Mlp, ViTBlock
from layer.jscc_encoder import RateAdaptionEncoder


class JSCCEncoderViT(nn.Module):
    """ViT-based JSCC Encoder with global self-attention.

    Drop-in replacement for JSCCEncoder — same interface and attribute names.
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=nn.LayerNorm,
                 rate_choice=[0, 128, 256], input_resolution=(16, 16),
                 depths=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))

        # Handle list-style args from config (fe_kwargs uses depths=[4], num_heads=[8])
        if depths is not None:
            depth = sum(depths)
        if isinstance(num_heads, (list, tuple)):
            num_heads = num_heads[0]

        num_patches = input_resolution[0] * input_resolution[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.layers = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.rate_adaption = RateAdaptionEncoder(embed_dim, rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)
        self.refine = Mlp(embed_dim * 2, embed_dim * 8, embed_dim)

    def forward(self, x, px, eta):
        B, C, H, W = x.size()
        hx = torch.clamp_min(-torch.log(px) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        x_BLC = x.flatten(2).permute(0, 2, 1)
        px_BLC = px.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes).reshape(B, H * W, C)
        x_BLC = x_BLC + rate_token + self.pos_embed[:, :H * W, :]
        for layer in self.layers:
            x_BLC = layer(x_BLC)
        x_BLC = self.norm(x_BLC)
        x_BCHW = x_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
        s_masked, mask = self.rate_adaption(x_BCHW, indexes)
        return s_masked, mask, indexes

    def update_resolution(self, H, W):
        pass
