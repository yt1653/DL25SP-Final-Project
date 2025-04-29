"""
Self-supervised JEPA training script.
Run once to produce jepa.ckpt, then run main.py (which will load it).

$ python train.py --steps 15000 --ckpt jepa.ckpt
"""
import argparse, math, torch
from tqdm.auto import tqdm

from dataset import create_wall_dataloader
from models  import JEPAModel

# ---------------- utils -------------------------------------------------
def cosine_tau(step, total, base=0.996, final=0.9995):
    p = step / total
    return final - (final - base) * 0.5 * (1 + math.cos(math.pi * p))

# ---------------- main routine -----------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JEPAModel(
        in_ch=2, act_dim=2,
        state_dim=64, hidden_dim=64,
        ema_tau=args.tau, device=device,
    )

    loader = create_wall_dataloader(
        "/scratch/DL25SP/train",
        probing=False, device=device, train=True,
    )
    opt     = torch.optim.Adam(model.parameters(), args.lr)
    iterator= iter(loader)

    model.train()
    pbar = tqdm(range(args.steps), desc="JEPA training")

    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        s  = batch.states.to(device)      # (B,T,C,H,W)
        a  = batch.actions.to(device)     # (B,T-1,2)

        # teacher forcing only (simplest)
        online = model._teacher_force(s, a)

        with torch.no_grad():
            tgt = model.target_encoder(
                    s.flatten(0,1)).view_as(online)

        loss = model.jepa_loss(online, tgt)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tau = cosine_tau(step, args.steps)
        model.update_target(tau)

        if step % args.print_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), args.ckpt)
    print(f"✓ saved checkpoint to {args.ckpt}")

# ---------------- argparse ---------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr",    type=float, default=3e-4)
    ap.add_argument("--tau",   type=float, default=0.996)
    ap.add_argument("--ckpt",  type=str, default="jepa.ckpt")
    ap.add_argument("--print-every", type=int, default=800)
    main(ap.parse_args())
