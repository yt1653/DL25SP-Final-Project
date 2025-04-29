import os, math, torch
from tqdm.auto import tqdm
from dataset   import create_wall_dataloader
from evaluator import ProbingEvaluator
from models    import JEPAModel

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", dev)
    return dev

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
    """
    • If jepa.ckpt exists → load and return the model.
    • Else  → run a short self-supervised JEPA training loop,
              save jepa.ckpt, then return the trained model.
    All helper code lives *inside* this function, so the rest of
    main.py (load_data, evaluate_model, __main__) is unchanged.
    """
    import os, math, torch
    from tqdm.auto import tqdm
    from models import JEPAModel
    from dataset import create_wall_dataloader

    CKPT_FILE   = "jepa.ckpt"
    TRAIN_STEPS = 10_000
    LR          = 3e-4
    ROLL        = 3
    PRINT_EVERY = 800

    # ---------------------------------------------------------------
    def cosine_tau(step, total, base=0.996, final=0.9995):
        p = step / total
        return final - (final - base) * 0.5 * (1 + math.cos(math.pi * p))

    # build tiny model
    model = JEPAModel(in_ch=2, act_dim=2,
                      state_dim=64, hidden_dim=64,
                      ema_tau=0.995, device=device)

    # if checkpoint exists – just load it
    if os.path.exists(CKPT_FILE):
        model.load_state_dict(torch.load(CKPT_FILE, map_location=device))
        print("✓ loaded pretrained weights")
        return model

    # ---------------------------------------------------------------
    #                self-supervised pre-training
    # ---------------------------------------------------------------
    print("⏳  checkpoint not found – training JEPA …")

    train_loader = create_wall_dataloader(
        "/scratch/DL25SP/train",
        probing=False, device=device, train=True,
    )
    iterator = iter(train_loader)
    opt      = torch.optim.Adam(model.parameters(), LR)

    model.train()
    for step in tqdm(range(TRAIN_STEPS), desc="JEPA pre-train"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch    = next(iterator)

        s = batch.states.to(device)
        a = batch.actions.to(device)

        tf   = model._teacher_force(s[:, :-ROLL], a[:, :-ROLL])
        roll = model(s[:, :1], a[:, :ROLL])
        online = torch.cat([roll, tf[:, ROLL:]], 1)   # (B,T_out,64)

        B, T_out, _ = online.shape
        tgt = model.target_encoder(s[:, :T_out].flatten(0,1)) \
                    .view(B, T_out, -1).detach()

        loss = model.jepa_loss(online, tgt)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        model.update_target(cosine_tau(step, TRAIN_STEPS))

        if step % PRINT_EVERY == 0:
            print(f"[step {step:>5}] VicReg loss = {loss.item():.4f}", flush=True)

    torch.save(model.state_dict(), CKPT_FILE)
    print(f"✅  saved checkpoint to {CKPT_FILE}")
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
