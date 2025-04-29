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
    def __init__(self, in_ch=2, out_ch=32):
        super().__init__()
        self.conv = nn.Sequential(               # 65 → 33 → 17 → 16
            nn.Conv2d(in_ch, 16, 5, 2, 2), nn.ReLU(True),  # 65→33
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True),     # 33→17
            nn.Conv2d(32, out_ch, 3, 1, 0), nn.ReLU(True), # 17→15
            nn.ZeroPad2d((0,1,0,1)),                      # 15→16 (square)
        )

    def forward(self, x):            # (B,C,H,W) → (B,32,16,16)
        return self.conv(x)

class ConvGRUPredictor(nn.Module):
    def __init__(self, ch=32, act_dim=2, hidden_ch=32):
        super().__init__()
        # project action to a 16×16 plane and concat to feature map
        self.act_proj = nn.Linear(act_dim, 16 * 16, bias=False)
        self.gru = nn.GRUCell(ch + 1, hidden_ch)          # +1 action plane
        self.out = nn.Conv2d(hidden_ch, ch, 1)

    def step(self, feat, act):
        B = feat.size(0)
        act_plane = self.act_proj(act).view(B, 1, 16, 16)
        x = torch.cat([feat, act_plane], 1)               # (B,33,16,16)
        h = self.gru(x.flatten(2).transpose(1,2),         # (B,256,33)
                     torch.zeros(B, self.gru.hidden_size,
                                 device=feat.device))
        h_map = h.transpose(1,2).view(B, self.gru.hidden_size, 16, 16)
        return self.out(h_map)                            # (B,32,16,16)

    def forward(self, s0, actions):
        """
        s0      : (B,32,16,16)  initial feature map from encoder
        actions : (B,T,2)       Δx,Δy for each step
        returns : (B,T,32,16,16)
        """
        B, T, _ = actions.shape
        h = torch.zeros(B, self.gru.hidden_size, device=s0.device)  # hidden state
        f = s0
        outs = []
        for t in range(T):
            act_plane = self.act_proj(actions[:, t]).view(B, 1, 16, 16)
            x = torch.cat([f, act_plane], 1)            # (B,33,16,16)

            # ---- convert spatial map to a 2-D vector (B,33) ----
            z = x.flatten(2).mean(2)                    # (B,33)

            # ---- recurrent update --------------------------------
            h = self.gru(z, h)                          # (B,hidden)

            # ---- broadcast hidden back to spatial map ------------
            f = self.out(h.unsqueeze(-1).unsqueeze(-1)  # (B,hidden,1,1)
                           .expand(-1, -1, 16, 16))     # (B,32,16,16)

            outs.append(f)

        return torch.stack(outs, 1)                     # (B,T,32,16,16)

#### JEPA model ####

class JEPAModel(nn.Module):
    def __init__(self, in_ch=2, act_dim=2,
                 ch=32, hidden_ch=32, repr_dim=64,
                 ema_tau=0.996, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = SpatialEncoder(in_ch, ch).to(self.device)
        self.target_encoder = SpatialEncoder(in_ch, ch).to(self.device)
        self.predictor = ConvGRUPredictor(ch, act_dim, hidden_ch).to(self.device)

        # compress spatial map to vector for evaluator / prober
        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, repr_dim),
        ).to(self.device)
        self.repr_dim = repr_dim

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

    def _to_vec(self, feat_map):                 # (B,32,16,16) → (B,D)
        return self.readout(feat_map)

    def _teacher_force(self, states, actions):   # used in training
        B, T, *_ = states.shape
        enc_all = self._encode(states.flatten(0,1)).view(B, T, -1, 16, 16)
        preds = self.predictor(enc_all[:,0], actions)      # (B,T-1,32,16,16)
        full = torch.cat([enc_all[:, :1], preds], 1)       # (B,T,32,16,16)
        return self._to_vec(full.view(-1,32,16,16)).view(B, T, -1)

    @torch.no_grad()
    def _rollout(self, states, actions):         # evaluator inference
        B = states.size(0)
        f0 = self._encode(states[:,0])
        preds = self.predictor(f0, actions)      # (B,T,32,16,16)
        full = torch.cat([f0.unsqueeze(1), preds], 1)      # (B,T+1,32,16,16)
        return self._to_vec(full.view(-1,32,16,16)).view(B, -1, self.repr_dim)

    def forward(self, states, actions):
        if states.size(1) == 1:      # inference
            return self._rollout(states, actions)
        return self._teacher_force(states, actions)

    # ------------- VicReg loss ------------------------------------- #
    def jepa_loss(self, online, target, λvar=25., λcov=1.):
        mse = F.mse_loss(online, target)

        z = online.reshape(-1, online.size(-1))
        std = torch.sqrt(z.var(0) + 1e-4)
        var_loss = torch.clamp(1.-std, min=0).mean()

        zc = z - z.mean(0)
        cov = (zc.T @ zc)/(z.shape[0]-1)
        off = cov.flatten()[1:].view(z.size(1)-1, z.size(1)+1)[:,:-1]
        cov_loss = (off**2).mean()
        return mse + λvar*var_loss + λcov*cov_loss


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
