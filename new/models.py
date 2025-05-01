from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)



class ResidualPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return x[..., :self.output_dim] + self.net(x)  


class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        # x: [B, 1, input_dim]
        # h: [1, B, hidden_dim] (GRU hidden state)
        out, h_next = self.gru(x, h)              # out: [B, 1, hidden_dim]
        pred = self.linear(out.squeeze(1))        # [B, output_dim]
        return pred, h_next


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_c):
        # x: [B, 1, input_dim]
        # h_c: (h, c), each of shape [1, B, hidden_dim]
        out, (h_next, c_next) = self.lstm(x, h_c)     # out: [B, 1, hidden_dim]
        pred = self.linear(out.squeeze(1))            # pred: [B, output_dim]
        return pred, (h_next, c_next)



class ActionEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, action):
        return self.encoder(action)




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



class JEPAAgent(nn.Module):
    def __init__(self, repr_dim=256, action_emb_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_emb_dim = action_emb_dim
        self.hidden_dim = 256 #for GRU
        self.aux_position_head = nn.Linear(self.repr_dim, 2)


        # Encoder: 2-channel image -> representation
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # [B, 2, 64, 64] -> [B, 16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 16, 32, 32] -> [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 16, 16] -> [B, 64, 8, 8] 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder_projector = nn.Sequential(
            nn.Flatten(), # [B, 64, 8, 8] -> [B, 4096]
            nn.Linear(64 * 8 * 8, repr_dim),    # [B, 4096] -> [B, 256]
        )

        self.action_encoder = ActionEncoder(input_dim=2, hidden_dim=32, output_dim=self.action_emb_dim)

        # Spatial predictor
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )

        # Predictor: (s_prev, action) -> s_next_pred
        #self.predictor = build_mlp([repr_dim + 2, 512, repr_dim])
        #self.predictor = ResidualPredictor(input_dim=self.repr_dim + 2, hidden_dim=512, output_dim=self.repr_dim)
        #self.predictor = ResidualPredictor(input_dim=self.repr_dim + self.action_emb_dim, hidden_dim=512, output_dim=self.repr_dim)
        #self.predictor = build_mlp([repr_dim + action_emb_dim, 512, repr_dim])
        '''
        self.predictor = GRUPredictor(
            input_dim=self.repr_dim + self.action_emb_dim,
            hidden_dim=256,
            output_dim=self.repr_dim
        )
        '''
        self.predictor = LSTMPredictor(
            input_dim=self.repr_dim + self.action_emb_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.repr_dim
        )

    # JEPA rollout for T steps:
    # At t = 0: use encoder to get s_0 from o_0
    # At t = 1 to T-1: use predictor(s_{t-1}, u_{t-1}) to get s̃_t
    # Output: a sequence of T representations [s_0, s̃_1, ..., s̃_{T-1}]
    def forward(self, states, actions): 
        # actions: [B, T-1, 2] 
        # states: [num_trajectories, trajectory_length, 2, 64, 64] ([B, T, 2, 64, 64])
        # return the representation of states: [B, T, repr_dim] 
        """
        states: [B, T, 2, 64, 64]
        actions: [B, T-1, 2]
        returns: [B, T, D]
        """
        B, T, _, H, W = states.shape
        reprs = []

        # Encode first observation step-by-step with shape printing
        x = states[:, 0]  # [B, 2, 64, 64]
        #print(f"Actual input size: {states[:, 0].shape}") 
        feat = self.encoder_backbone(x)            # [B, 64, 8, 8]
        for i, layer in enumerate(self.encoder_projector):
            feat = layer(feat)
            #print(f"After encoder layer {i} ({layer.__class__.__name__}): {feat.shape}")
        s_prev = feat
        reprs.append(s_prev)

        h = torch.zeros(1, B, self.hidden_dim).to(states.device) 
        c = torch.zeros(1, B, self.hidden_dim, device=states.device)
        for t in range(T - 1):
            a_t = actions[:, t]  # [B, 2]
            a_embed = self.action_encoder(a_t)  # [B, action_emb_dim]
            inp = torch.cat([s_prev, a_embed], dim=-1)  # [B, repr_dim + action_emb_dim]
            inp = inp.unsqueeze(1)  #for GRU
            #inp = torch.cat([s_prev, a_t], dim=-1)  # [B, repr_dim+2]
            #s_pred = self.predictor(inp)  # [B, repr_dim]
            #s_pred, h = self.predictor(inp, h) # for GRU
            s_pred, (h, c) = self.predictor(inp, (h, c))    # for LSTM 
            reprs.append(s_pred)
            s_prev = s_pred

        return torch.stack(reprs, dim=1)  # [B, T, repr_dim]

    def forward_spatial(self, states):
        """
        Predict spatial feature map from encoder features.
        Args:
            states: [B, 2, 64, 64]
        Returns:
            feat_map: [B, 64, 8, 8] - encoder output
            pred_map: [B, 64, 8, 8] - predicted feature map
        """
        feat_map = self.encoder_backbone(states)         # [B, 64, 8, 8]
        pred_map = self.spatial_predictor(feat_map)      # [B, 64, 8, 8]
        return feat_map, pred_map

  
