#### MiniConv GRU-JEPA (VicReg - 256)####


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
    """
    Flexible CNN encoder.  Accepts arbitrary C,H,W and returns a
    fixed-dim embedding via adaptive pooling.
    """
    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  32, 3, 2, 1),   # now uses in_ch, not hard-coded 3
            nn.ReLU(True),
            nn.Conv2d(32,     64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64,    128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128,   256, 3, 2, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    

#### Predictor ####
class GRUPredictor(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = state_dim + act_dim
        self.gru = nn.GRUCell(self.input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        prev_state: torch.Tensor,   # (B, D)
        action_seq: torch.Tensor,   # (B, T, A)
    ) -> torch.Tensor:
        """
        Roll the GRU for T steps and return predicted states:
            out[:,t] = Ŝ_{t}
        Shapes:
            prev_state : (B,D)   given S_0   (encoder on o_0)
            action_seq : (B,T,A)  u_0 … u_{T-1}
            returns    : (B,T,D)
        """
        B, T, _ = action_seq.shape
        h = prev_state.new_zeros(B, self.gru.hidden_size)
        z = []
        for t in range(T):
            inp = torch.cat([prev_state, action_seq[:, t]], dim=-1)
            h = self.gru(inp, h)
            prev_state = self.lin(h)          # Ŝ_{t+1}
            z.append(prev_state)
        return torch.stack(z, dim=1)          # (B,T,D)


#### JEPA model ####

class JEPAModel(nn.Module):
    def __init__(
        self,
        in_ch:   int = 2,   # <-- new argument
        act_dim: int = 2,
        state_dim: int = 256,
        hidden_dim: int = 512,
        ema_tau: float = 0.996,
        device:   str = "cuda",
    ):
        super().__init__()
        self.repr_dim = state_dim
        self.device   = torch.device(device)

        # pass in_ch here, not the default 3:
        self.encoder        = ConvEncoder(in_ch, state_dim)
        self.target_encoder = ConvEncoder(in_ch, state_dim)
        self._ema_tau       = ema_tau
        self._sync_target()

        self.predictor = GRUPredictor(state_dim, act_dim, hidden_dim)
        self.to(self.device)

    # -------------------------- public API -------------------------- #
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        See docstring above for expected behaviour.
        """
        if states.size(1) > 1:                                # === training ===
            return self._forward_train(states, actions)
        else:                                                 # === inference ===
            return self._forward_rollout(states, actions)

    # ---------------------- training (teacher-forcing) -------------- #
    def _forward_train(self, states: torch.Tensor, actions: torch.Tensor):
        """
        states  : (B,T,C,H,W)   with T ≥ 2
        actions : (B,T-1,A)
        return  : (B,T,D)       [ŝ₀ (= enc(o₀)), Ŝ₁ … Ŝ_{T-1}]
        """
        B, T, *_ = states.shape
        # encode all observations with *online* encoder (teacher forcing):
        s_online = self.encoder(states.flatten(0,1)).view(B, T, -1)  # (B,T,D)

        # first state is ground-truth embedding of o₀
        s0 = s_online[:, 0]                          # (B,D)
        preds = self.predictor(s0, actions)          # (B,T-1,D)
        preds = torch.cat([s0.unsqueeze(1), preds], dim=1)  # prepend ŝ₀
        return preds

    # -------------------- inference (single obs + actions) ---------- #
    @torch.no_grad()
    def _forward_rollout(self, states: torch.Tensor, actions: torch.Tensor):
        """
        states  : (B,1,C,H,W)   (only o₀)
        actions : (B,T,A)
        return  : (B,T+1,D) = [ŝ₀, Ŝ₁ … Ŝ_T]
        """
        B = states.size(0)
        s0 = self.encoder(states[:,0]).view(B,-1)      # (B,D)
        preds = self.predictor(s0, actions)            # (B,T,D)
        return torch.cat([s0.unsqueeze(1), preds], dim=1)

    # ------------------------ anti-collapse loss -------------------- #
    def jepa_loss(
        self,
        online_preds: torch.Tensor,   # (B,T,D) from encoder+predictor (online)
        target_states: torch.Tensor,  # (B,T,D) from *target* encoder
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
    ):
        """
        VicReg-style: invariance + variance + covariance.
        """
        B, T, D = online_preds.shape
        # invariance (MSE)
        loss_inv = F.mse_loss(online_preds, target_states)

        # reshape to (B·T , D)
        z = online_preds.reshape(-1, D)
        std = torch.sqrt(z.var(dim=0) + 1e-04)
        loss_var = torch.clamp(1.0 - std, min=0).mean()

        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (B*T - 1)          # (D,D)
        off_diag = cov.flatten()[
            1:].view(D - 1, D + 1)[:, :-1].flatten()
        loss_cov = (off_diag**2).mean()

        return loss_inv + lambda_var*loss_var + lambda_cov*loss_cov

    # ------------------- EMA target encoder utilities --------------- #
    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data = tp.data * self._ema_tau + p.data * (1.0 - self._ema_tau)

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
