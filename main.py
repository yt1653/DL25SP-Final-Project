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


def load_model(device):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    # model = MockModel()
    model = JEPAModel(in_ch=2, act_dim=2, state_dim=64,
                      hidden_dim=64, device=device, ema_tau=0.995)

    loader = create_wall_dataloader("/scratch/DL25SP/train",
                                    probing=False, device=device, train=True)
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    iterator = iter(loader)
    model.train()

    STEPS   = 10_000          # ← bump
    ROLL    = 3               # ← predict 3 steps before teacher forcing

    for step in tqdm(range(STEPS), desc="pre-train"):
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)

        s, a = batch.states.to(device), batch.actions.to(device)

        # teacher forcing for t ≥ 1, rollout 1 step:
        online0 = model._teacher_force(s[:, :-ROLL], a[:, :-ROLL])
        roll = model(s[:, :1], a[:, :ROLL])     # 3-step rollout
        online = torch.cat([roll, online0[:, ROLL:]], 1)

        with torch.no_grad():
            tgt = model.target_encoder(s.flatten(0,1)).view_as(online)

        loss = model.jepa_loss(online, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
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
