"""Stage 2, Step 5 — retrain the SAME CNN on real noise.

Stage 1's recipe, unchanged: same `Stage1CNN` architecture (251k params — the
architecture is deliberately NOT a Stage 2 variable), BCEWithLogitsLoss, Adam, model
selection on val AUC, test split untouched until stage2_eval.py. The only difference is
the dataset underneath — real O3 noise, time-ordered splits.

One Stage 2 wrinkle worth naming: VAL IS LATER IN TIME THAN TRAIN. Val AUC is therefore
measuring generalisation across both the injection distribution AND the detector's
drift — which is what a deployed model would face, and is exactly the number model
selection should be optimising here.

Expectations, so a surprise is recognisable as one:
  - val AUC should land BELOW Stage 1's 0.9872 (real noise is simply harder), but well
    above the transferred Stage-1 model's AUC on the same data (that gap is what
    retraining recovers). stage2_eval.py measures both properly on test.
  - The >0.999 SUSPICIOUS rule stands. On real noise it is if anything more alarming.

Run (from WSL2, ~1 min on the 3070):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 2"
    python stage2_train.py                  # the real thing
    python stage2_train.py --epochs 1       # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_model import Stage1CNN
from stage1_train import run_val  # val loss + AUC; eval-mode discipline lives there

from stage2_data import Stage2Dataset

DEFAULT_CKPT = Path.home() / "ligo-data" / "stage2_cnn.pt"
OUT = Path(__file__).parent / "outputs"

BATCH = 256
LR = 1e-3
SEED = 20260712
PATIENCE = 6


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 Step 5 — train the CNN on real noise")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("device: CPU — this will be ~10-20x slower; the plan expects the 3070")

    train = Stage2Dataset("train")
    val = Stage2Dataset("val")
    print(f"{train}\n{val}\n")
    train_loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val, batch_size=1024, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    model = Stage1CNN().to(device)  # the SAME architecture — that is the point
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage1CNN (unchanged, 1 variable per stage): {n_params:,} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch = -1.0, -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / n_batches
        val_loss, val_auc = run_val(model, val_loader, device)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_auc"].append(val_auc)

        star = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            args.ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "seed": SEED,
                    "trained_on": "stage2 (real O3 noise)",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                args.ckpt,
            )
            star = "  * saved"
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val AUC {val_auc:.4f}  ({time.time()-te:.0f}s){star}")

        if epoch - best_epoch >= PATIENCE:
            print(f"\nearly stop: no val-AUC improvement in {PATIENCE} epochs")
            break

    print(f"\ndone in {(time.time()-t0)/60:.1f} min — best val AUC {best_auc:.4f} "
          f"(epoch {best_epoch}) -> {args.ckpt}")

    ok = best_auc >= 0.85  # real noise is harder; Stage 1's 0.90 gate was for Gaussian
    print(f"\n  {'PASS' if ok else 'FAIL'}: best val AUC {best_auc:.4f} "
          f"{'>=' if ok else '<'} 0.85 (real-noise gate; Stage 1's Gaussian gate was 0.90)")
    if best_auc > 0.999:
        print("  SUSPICIOUS: AUC > 0.999 on real noise with half-weak injections. Assume "
              "leakage before genius — stage2_eval's low-SNR efficiency is the arbiter.")

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))

    axl.plot(epochs, hist["train_loss"], "o-", color="tab:blue", label="train")
    axl.plot(epochs, hist["val_loss"], "o-", color="tab:red", label="validation (later in time)")
    axl.set_xlabel("epoch")
    axl.set_ylabel("BCE loss")
    axl.set_title("Loss — real O3 noise")
    axl.legend()
    axl.grid(alpha=0.3)

    axa.plot(epochs, hist["val_auc"], "o-", color="tab:red")
    axa.axvline(best_epoch, color="k", lw=1, ls="--", alpha=0.6,
                label=f"checkpoint: epoch {best_epoch}, AUC {best_auc:.4f}")
    axa.axhline(0.9872, color="tab:gray", lw=1, ls=":",
                label="Stage 1 val AUC on Gaussian noise (0.9872)")
    axa.set_xlabel("epoch")
    axa.set_ylabel("validation AUC")
    axa.set_title("Model selection on val AUC — val is strictly later in time")
    axa.legend(loc="lower right", fontsize=8)
    axa.grid(alpha=0.3)

    fig.suptitle(f"Stage 2 CNN training — same architecture, real noise "
                 f"({n_params:,} params, batch {args.batch}, Adam {args.lr})")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "5_training.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
