"""Stage 3, Step 4 — train the same CNN on Stage2-train + Stage3-train.

The mix: ~80k Stage 2 crops (injections + plain real noise) + ~9k Stage 3 crops
(glitch negatives + same-block injections + plain). The glitches are ~3% of training —
deliberately NOT oversampled in v1: Stage 2 showed the model learns from ~28 unlabelled
glitches/h, so the first question is what a realistic exposure of labelled ones buys.
Model selection on the combined val (both vals are later-in-time than their trains).

Run (from WSL2, ~1-2 min on the 3070):
    python stage3_train.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 2"))
from stage1_model import Stage1CNN
from stage1_train import run_val

from stage2_data import Stage2Dataset

from stage3_data import Stage3Dataset

DEFAULT_CKPT = Path.home() / "ligo-data" / "stage3_cnn.pt"
OUT = Path(__file__).parent / "outputs"
BATCH = 256
LR = 1e-3
SEED = 20260713
PATIENCE = 6


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 Step 4 — train on the glitch mix")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    s2tr, s3tr = Stage2Dataset("train"), Stage3Dataset("train")
    s2va, s3va = Stage2Dataset("val"), Stage3Dataset("val")
    print(f"{s2tr}\n{s3tr}\n{s2va}\n{s3va}\n")
    train = ConcatDataset([s2tr, s3tr])
    val = ConcatDataset([s2va, s3va])

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=1024, num_workers=2,
                            pin_memory=True, persistent_workers=True)

    model = Stage1CNN().to(device)  # STILL the same architecture — data is the variable
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        running, nb = 0.0, 0
        for x, yy in train_loader:
            x, yy = x.to(device, non_blocking=True), yy.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), yy)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        val_loss, val_auc = run_val(model, val_loader, device)
        hist["train_loss"].append(running / nb)
        hist["val_loss"].append(val_loss)
        hist["val_auc"].append(val_auc)
        star = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            args.ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_auc": val_auc, "seed": SEED,
                        "trained_on": "stage2+stage3 (real noise + labelled glitches)",
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, args.ckpt)
            star = "  * saved"
        print(f"epoch {epoch:3d}  train loss {running/nb:.4f}  val loss {val_loss:.4f}  "
              f"val AUC {val_auc:.4f}  ({time.time()-te:.0f}s){star}")
        if epoch - best_epoch >= PATIENCE:
            print(f"\nearly stop after {PATIENCE} flat epochs")
            break

    print(f"\ndone in {(time.time()-t0)/60:.1f} min — best val AUC {best_auc:.4f} "
          f"(epoch {best_epoch}) -> {args.ckpt}")
    ok = best_auc >= 0.85
    print(f"\n  {'PASS' if ok else 'FAIL'}: best val AUC {best_auc:.4f} >= 0.85")

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))
    axl.plot(epochs, hist["train_loss"], "o-", color="tab:blue", label="train")
    axl.plot(epochs, hist["val_loss"], "o-", color="tab:red", label="val (combined)")
    axl.set_xlabel("epoch"), axl.set_ylabel("BCE loss"), axl.legend(), axl.grid(alpha=0.3)
    axl.set_title("Loss — Stage2 + Stage3 mix")
    axa.plot(epochs, hist["val_auc"], "o-", color="tab:red")
    axa.axvline(best_epoch, color="k", lw=1, ls="--", alpha=0.6,
                label=f"checkpoint: epoch {best_epoch}, AUC {best_auc:.4f}")
    axa.set_xlabel("epoch"), axa.set_ylabel("val AUC"), axa.grid(alpha=0.3)
    axa.set_title("Model selection on combined val")
    axa.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "5_training.png", dpi=130)
    print(f"  plot -> {OUT / '5_training.png'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
