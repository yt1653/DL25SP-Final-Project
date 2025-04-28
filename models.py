####  ####


from typing import List, Tuple, Optional
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

#### Helper ####
class ConvEncoder(nn.Module):
    """Keep 16×16 spatial map so we don’t quantise away sub-pixel detail."""
    def __init__(self, in_ch: int, state_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(                # 65×65 → 16×16
            nn.Conv2d(in_ch, 16, 3, 2, 1), nn.ReLU(True),   # 33×33
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),      # 17×17
            nn.Conv2d(32, 32, 3, 1, 0), nn.ReLU(True),      # 15×15
        )
        self.head = nn.Linear(32 * 16 * 16, state_dim)

    def forward(self, x):
        f = self.conv(x)
        return self.head(f.flatten(1))

    

#### Predictor ####
class GRUPredictor(nn.Module):
    """tiny predictor – 64-d hidden, 2 k params total"""
    def __init__(self, d=64, act=2, h=64):
        super().__init__()
        self.rnn = nn.GRUCell(d + act, h)
        self.out = nn.Linear(h, d)

    def forward(self, s0, a):
        B, T, _ = a.shape
        h = s0.new_zeros(B, self.rnn.hidden_size)
        outs = []
        for t in range(T):
            h = self.rnn(torch.cat([s0, a[:, t]], -1), h)
            s0 = self.out(h)
            outs.append(s0)
        return torch.stack(outs, 1)

#### JEPA model ####

class JEPAModel(nn.Module):
    def __init__(
        self,
        in_ch:      int  = 2,
        act_dim:    int  = 2,
        state_dim:  int  = 512,
        hidden_dim: int  = 1024,
        ema_tau:    float= 0.996,
        device:     str  = "cuda",
    ):
        super().__init__()
        self.repr_dim = state_dim
        self.device   = torch.device(device)

        self.encoder        = ConvEncoder(in_ch, state_dim)
        self.target_encoder = ConvEncoder(in_ch, state_dim)
        self.predictor      = GRUPredictor(state_dim, act_dim, hidden_dim)

        self._ema_tau = ema_tau
        self._sync_target()
        self.to(self.device)

    # ------------ public forward (used only by evaluator) -----------
    def forward(self, states, actions):         # see evaluator contract
        if states.size(1) == 1:
            return self._rollout(states, actions)
        else:
            return self._teacher_force(states, actions)

    # ------------ teacher-forcing (for pre-training) ----------------
    def _teacher_force(self, states, actions):
        B, T, *_ = states.shape
        s_all = self.encoder(states.flatten(0,1)).view(B, T, -1)
        preds = self.predictor(s_all[:,0], actions)      # (B,T-1,D)
        return torch.cat([s_all[:, :1], preds], 1)       # (B,T,D)

    # ------------ inference / rollout (evaluator uses this) ---------
    @torch.no_grad()
    def _rollout(self, states, actions):
        B = states.size(0)
        s0 = self.encoder(states[:,0])
        preds = self.predictor(s0, actions)              # (B,T,D)
        return torch.cat([s0.unsqueeze(1), preds], 1)    # (B,T+1,D)

    # ------------ VicReg-style loss (call from your train loop) -----
    def jepa_loss(self, online, target, λvar=25., λcov=1.):
        mse = F.mse_loss(online, target)

        z = online.reshape(-1, online.size(-1))
        std = torch.sqrt(z.var(0) + 1e-4)
        var_loss = torch.clamp(1.-std, min=0).mean()

        zc = z - z.mean(0)
        cov = (zc.T @ zc)/(z.shape[0]-1)
        off = cov.flatten()[1:].view(cov.size(0)-1, cov.size(1)+1)[:,:-1]
        cov_loss = (off**2).mean()

        return mse + λvar*var_loss + λcov*cov_loss

    # ------------ momentum update helpers ---------------------------
    @torch.no_grad()
    def update_target(self, tau=None):
        tau = tau or self._ema_tau
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.lerp_(p.data, 1. - tau)

    @torch.no_grad()
    def _sync_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.copy_(p.data)


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
