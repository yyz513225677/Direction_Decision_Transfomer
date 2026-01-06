import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTokenEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=4):
        super().__init__()
        self.in_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):  # (B,T,4)
        h = self.in_mlp(tokens) + self.pos_mlp(tokens[..., :3])
        h = self.enc(h)
        h = self.norm(h)
        return h.mean(dim=1)  # (B,C)

class DirectionTransformer(nn.Module):
    def __init__(self, num_tokens=256, d_model=256, nhead=8, depth=4):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.backbone = PointTokenEncoder(d_model=d_model, nhead=nhead, depth=depth)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def _sample_tokens(self, x):  # x: (B,N,4)
        B, N, _ = x.shape
        T = self.num_tokens
        if N >= T:
            idx = torch.randperm(N, device=x.device)[:T]
            idx = idx.unsqueeze(0).expand(B, T)
        else:
            idx = torch.randint(0, N, (B, T), device=x.device)
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))

    def forward(self, pc0, pc1):  # (B,N,4)
        t0 = self._sample_tokens(pc0)
        t1 = self._sample_tokens(pc1)

        g0 = self.backbone(t0)
        g1 = self.backbone(t1)

        fuse = torch.cat([g0, g1, g1 - g0, g0 * g1], dim=-1)
        out = self.head(fuse)
        out = F.normalize(out, dim=-1, eps=1e-6)  # unit direction
        return out
