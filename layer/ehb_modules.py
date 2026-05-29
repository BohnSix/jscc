import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertHead(nn.Module):
    """Project one hierarchical feature scale to the transmit latent resolution.

    Uses strided convolutions for spatial downsampling to preserve more
    structural information than adaptive average pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, target_size: int = 16,
                 input_size: int = 16, use_ib: bool = False):
        super().__init__()
        self.use_ib = use_ib
        self.target_size = target_size
        final_channels = 2 * out_channels if use_ib else out_channels

        # Build strided conv layers to downsample from input_size to target_size
        layers = []
        current_size = input_size
        ch = in_channels
        while current_size > target_size:
            layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            layers.append(nn.PReLU())
            current_size //= 2
        layers.append(nn.Conv2d(ch, final_channels, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        if not self.use_ib:
            return out, {}
        mu, logvar = torch.chunk(out, 2, dim=1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            u = mu + torch.randn_like(std) * std
        else:
            u = mu
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return u, {"mu": mu, "logvar": logvar, "kl": kl}


class EHBExperts(nn.Module):
    """Four expert branches, one per hierarchical scale."""

    def __init__(self, embed_dim: int = 256, target_size: int = 16, use_ib: bool = False):
        super().__init__()
        # Input sizes: F1=128, F2=64, F3=32, F4=16 (for 256x256 input)
        self.expert1 = ExpertHead(embed_dim, embed_dim, target_size, input_size=128, use_ib=use_ib)
        self.expert2 = ExpertHead(embed_dim, embed_dim, target_size, input_size=64, use_ib=use_ib)
        self.expert3 = ExpertHead(embed_dim, embed_dim, target_size, input_size=32, use_ib=use_ib)
        self.expert4 = ExpertHead(embed_dim, embed_dim, target_size, input_size=16, use_ib=use_ib)

    def forward(self, f1, f2, f3, f4):
        u1, ib1 = self.expert1(f1)
        u2, ib2 = self.expert2(f2)
        u3, ib3 = self.expert3(f3)
        u4, ib4 = self.expert4(f4)
        return u1, u2, u3, u4, [ib1, ib2, ib3, ib4]


class EntropyEstimator(nn.Module):
    """Lightweight CNN proxy for image complexity from hierarchical features."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 64, target_size: int = 16):
        super().__init__()
        self.target_size = target_size
        self.proj1 = nn.Conv2d(embed_dim, hidden_dim, 1)
        self.proj2 = nn.Conv2d(embed_dim, hidden_dim, 1)
        self.proj3 = nn.Conv2d(embed_dim, hidden_dim, 1)
        self.proj4 = nn.Conv2d(embed_dim, hidden_dim, 1)
        self.net = nn.Sequential(
            nn.Conv2d(4 * hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _align(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.target_size:
            return x
        return F.adaptive_avg_pool2d(x, self.target_size)

    def forward(self, f1, f2, f3, f4) -> torch.Tensor:
        h1 = self._align(self.proj1(f1))
        h2 = self._align(self.proj2(f2))
        h3 = self._align(self.proj3(f3))
        h4 = self._align(self.proj4(f4))
        return self.net(torch.cat([h1, h2, h3, h4], dim=1))


class GateNet(nn.Module):
    """Generate four expert fusion weights from complexity, SNR, and eta."""

    def __init__(self, hidden_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, r_map: torch.Tensor, snr: float, eta: float) -> torch.Tensor:
        B = r_map.size(0)
        device = r_map.device
        dtype = r_map.dtype
        r_global = r_map.mean(dim=(2, 3))
        snr_norm = torch.full((B, 1), snr / 20.0, device=device, dtype=dtype)
        eta_norm = torch.full((B, 1), eta, device=device, dtype=dtype)
        cond = torch.cat([r_global, snr_norm, eta_norm], dim=1)
        return self.mlp(cond)


class ExpertFusion(nn.Module):
    """Fuse four expert outputs into the transmit latent."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.fuse = nn.Conv2d(4 * embed_dim, embed_dim, 1)
        self.stage1_fuse = nn.Conv2d(embed_dim, embed_dim, 1)
        self.stage2_fuse = nn.Conv2d(2 * embed_dim, embed_dim, 1)
        self.stage3_fuse = nn.Conv2d(3 * embed_dim, embed_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, u1, u2, u3, u4, alpha=None):
        if alpha is not None:
            a = alpha.to(device=u1.device, dtype=u1.dtype)
            u1 = a[:, 0].view(-1, 1, 1, 1) * u1
            u2 = a[:, 1].view(-1, 1, 1, 1) * u2
            u3 = a[:, 2].view(-1, 1, 1, 1) * u3
            u4 = a[:, 3].view(-1, 1, 1, 1) * u4
        return self.fuse(torch.cat([u1, u2, u3, u4], dim=1))

    def forward_stages(self, u1, u2, u3, u4, alpha=None):
        if alpha is not None:
            a = alpha.to(device=u1.device, dtype=u1.dtype)
            u1 = a[:, 0].view(-1, 1, 1, 1) * u1
            u2 = a[:, 1].view(-1, 1, 1, 1) * u2
            u3 = a[:, 2].view(-1, 1, 1, 1) * u3
            u4 = a[:, 3].view(-1, 1, 1, 1) * u4
        z1 = self.stage1_fuse(u1)
        z2 = self.stage2_fuse(torch.cat([u1, u2], dim=1))
        z3 = self.stage3_fuse(torch.cat([u1, u2, u3], dim=1))
        z4 = self.fuse(torch.cat([u1, u2, u3, u4], dim=1))
        return z1, z2, z3, z4


class RateAllocator(nn.Module):
    """Predict per-patch symbol count from R_map + SNR + eta."""

    def __init__(self, hidden_dim: int = 32, max_rate: int = 256):
        super().__init__()
        self.max_rate = max_rate
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, r_map: torch.Tensor, snr: float, eta: float):
        B, _, H, W = r_map.shape
        device, dtype = r_map.device, r_map.dtype
        snr_map = torch.full((B, 1, H, W), snr / 20.0, device=device, dtype=dtype)
        eta_map = torch.full((B, 1, H, W), eta, device=device, dtype=dtype)
        cond = torch.cat([r_map, snr_map, eta_map], dim=1)
        rate_map = self.net(cond)
        symbol_num = rate_map.flatten() * self.max_rate
        return symbol_num, rate_map


class RatePredictor(nn.Module):
    """Predict per-image continuous rate from fused latent Z."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 64, max_rate: int = 256):
        super().__init__()
        self.max_rate = max_rate
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feat = z.mean(dim=(2, 3))
        return self.net(feat).squeeze(-1) * self.max_rate

