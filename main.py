from tqdm.auto import tqdm
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
# from models import MockModel
from models import JEPAModel
import glob
import math


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def _cosine_tau(step, total, base=0.996, final=0.9995):
    """Smooth EMA schedule for the momentum (BYOL style)."""
    p = step / total
    return final - (final - base) * 0.5 * (1 + math.cos(math.pi * p))

def load_data(device):
    data_path = "/scratch/DL25SP"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds

ROLL   = 3           # rollout steps
STEPS  = 10_000      # pre-train iterations
LR     = 3e-4

def load_model(device):
    # ───────── 1. build tiny-JEPA ───────────────────────────────────
    model = JEPAModel(
        in_ch=2, act_dim=2,
        state_dim=64, hidden_dim=64,
        ema_tau=0.995, device=device,
    )

    # ───────── 2. data loader (exploration set) ─────────────────────
    loader    = create_wall_dataloader(
                    "/scratch/DL25SP/train",
                    probing=False, device=device, train=True)
    iterator  = iter(loader)
    opt       = torch.optim.Adam(model.parameters(), LR)
    model.train()

    # ───────── 3. short VicReg pre-train loop ───────────────────────
    for step in tqdm(range(STEPS), desc="pre-train"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch    = next(iterator)

        s = batch.states.to(device)         # (B,T,C,H,W)
        a = batch.actions.to(device)        # (B,T-1,2)

        # teacher-forcing for t ≥ ROLL, rollout first ROLL steps
        online_tf = model._teacher_force(s[:, :-ROLL], a[:, :-ROLL])
        roll      = model(s[:, :1], a[:, :ROLL])           # ROLL-step rollout
        online    = torch.cat([roll, online_tf[:, ROLL:]], 1)   # (B,T_out,4)

        # -------- target encodings with SAME time length -------------
        B, T_out, _ = online.shape                                    # ← EDIT
        tgt = model.target_encoder(s[:, :T_out].flatten(0, 1))        # ← EDIT
        tgt = tgt.view(B, T_out, -1).detach()                         # ← EDIT

        loss = model.jepa_loss(online, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()

    model.eval()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    model = load_model(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
