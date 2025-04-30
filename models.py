from typing import List
import numpy as np, torch
from torch import nn
import torch.nn.functional as F


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class SpatialEncoder(nn.Module):
    def __init__(self, in_ch=2, ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, ch, 3, 1, 1), nn.ReLU(True),
        )

    def forward(self, x):               # (B,2,64,64) → (B,64,64,64)
        return self.conv(x)

class ConvGRUPredictor(nn.Module):
    def __init__(self, ch=64, act_dim=2, hidden_ch=128):
        super().__init__()
        self.act2flow = nn.Linear(act_dim, 2)            # (dx,dy)→flow
        self.gru = nn.GRUCell(ch + 2, hidden_ch)
        self.out = nn.Conv2d(hidden_ch, ch, 3, padding=1)

    def forward(self, f0, actions):                      # f0 (B,C,H,W)
        B, T, _ = actions.shape; H = W = f0.size(-1)
        h = torch.zeros(B, self.gru.hidden_size, device=f0.device)
        feats, f = [], f0
        for t in range(T):
            flow = self.act2flow(actions[:, t]).view(B, 2, 1, 1).expand(-1, -1, H, W)
            z = torch.cat([f, flow], 1)                  # (B,C+2,H,W)
            h = self.gru(z.flatten(2).mean(-1), h)
            f = self.out(h[:, :, None, None].expand(-1, -1, H, W))
            feats.append(f)
        return torch.stack(feats, 1)                     # (B,T,C,H,W)

#### JEPA model ####

class JEPAModel(nn.Module):
    def __init__(self, in_ch=2, act_dim=2,
                ch=32, hidden_ch=128, repr_dim=64,
                ema_tau=0.996, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.ch = ch
        self.encoder = SpatialEncoder(in_ch, ch).to(self.device)
        self.target_encoder = SpatialEncoder(in_ch, ch).to(self.device)
        self.predictor = ConvGRUPredictor(ch, act_dim, hidden_ch).to(self.device)

        self.soft_T   = nn.Parameter(torch.tensor(0.0))   # log-temperature
        self.repr_dim = self.ch + 2                       # 64 + 2 = 66

        self._ema_tau = ema_tau
        self._sync_target()

    # ------------- helpers ----------------------------------------- #
    @torch.no_grad()
    def _sync_target(self):
        for p, tp in zip(self.encoder.parameters(),
                         self.target_encoder.parameters()):
            tp.data.copy_(p.data)

    @torch.no_grad()
    def update_target(self, tau=None):
        tau = tau or self._ema_tau
        for p, tp in zip(self.encoder.parameters(),
                         self.target_encoder.parameters()):
            tp.data.lerp_(p.data, 1. - tau)

    # ------------- forward paths ----------------------------------- #
    def _encode(self, frames):
        return self.encoder(frames)

    def _to_vec(self, fm):                               # (B,C,H,W)
        B,C,H,W = fm.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=fm.device),
            torch.linspace(-1, 1, W, device=fm.device),
            indexing="ij")
        coords = torch.stack([xx, yy], 0)                # (2,H,W)

        T = self.soft_T.exp()                            # learned log-T
        w = F.softmax(fm.mean(1) / T, dim=(-2,-1))       # (B,H,W)
        mu = (w.unsqueeze(1) * coords).flatten(2).sum(-1)     # (B,2)

        g = F.adaptive_avg_pool2d(fm, 1).flatten(1)      # (B,C)
        return torch.cat([mu, g], 1)                     # (B, C+2)

    def _teacher_force(self, states, actions, *, return_maps=False):
        """
        Encode every frame, then predict the next (T-1) feature maps with the
        Conv-GRU, finally return either the full map sequence (B,T,C,64,64) or
        the usual latent vectors (B,T,D).
        """
        B, T, *_ = states.shape                       # states: (B,T,2,64,64)

        # encode each frame, then reshape to (B,T,C,64,64)
        enc = self._encode(states.flatten(0, 1)).view(B, T, self.ch, 64, 64)

        preds = self.predictor(enc[:, 0], actions)    # (B,T-1,C,64,64)
        full  = torch.cat([enc[:, :1], preds], 1)     # (B,T,C,64,64)

        if return_maps:
            return full                               # ----- maps for loss -----

        # convert each map to a (C+2)-d vector
        vecs = self._to_vec(full.flatten(0,1)         # (B·T,C+2)
                        ).view(B, T, -1)           # (B,T,D)
        return vecs


    @torch.no_grad()
    def _rollout(self, states, actions):
        """Inference path used by the evaluator (states has length 1)."""
        B = states.size(0)

        f0    = self._encode(states[:, 0])            # (B,C,64,64)
        preds = self.predictor(f0, actions)           # (B,T,C,64,64)
        full  = torch.cat([f0.unsqueeze(1), preds], 1)   # (B,T+1,C,64,64)

        vecs = self._to_vec(full.flatten(0,1))        # (B·(T+1),C+2)
        return vecs.view(B, -1, self.repr_dim)        # (B,T+1,D)


    def forward(self, states, actions):
        if states.size(1) == 1:      # inference
            return self._rollout(states, actions)
        return self._teacher_force(states, actions)

    # ------------- VicReg loss ------------------------------------- #
    def jepa_loss(self, online_fm, target_fm,
                  λpix=1.0, λvar=1., λcov=0.1):
        B,T,C,H,W = online_fm.shape
        pix = F.mse_loss(online_fm, target_fm)

        o = online_fm.flatten(0,1).mean([-1,-2])
        t = target_fm.flatten(0,1).mean([-1,-2])
        mse = F.mse_loss(o, t)

        std = torch.sqrt(o.var(0) + 1e-4)
        var = torch.clamp(1.-std, 0).mean()

        zc = o - o.mean(0)
        cov = (zc.T @ zc) / (o.size(0)-1)
        off = cov.flatten()[1:].view(C-1, C+1)[:, :-1]
        cov = (off**2).mean()

        return λpix*pix + mse + λvar*var + λcov*cov


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
