"""Stage 5, Step 4 — train the SAME CNN on one ablated arm.

stage3_train.py, with the dataset swapped for an arm of the ablation and nothing else
touched: same Stage1CNN (251,361 params), same BCEWithLogitsLoss, same Adam at 1e-3,
same batch 256, same patience 6, same model selection on val AUC, same mix
(Stage2-train + Stage3-train). The seed is the same too — if `raw` loses, it will not be
because it drew a worse initialisation.

    python stage5_train.py --arm raw
    python stage5_train.py --arm raw --no-clip     # if the check says the clip bites raw
    python stage5_train.py --arm bp
    python stage5_train.py --arm wh

The 0.85 val-AUC gate from Stages 2-3 is DELIBERATELY NOT a failure here. This stage's
whole purpose is to find out whether an arm fails; a script that exits non-zero when the
experiment answers "yes, it fails" would be a script that cannot report its own result.
The gate is printed, not enforced.
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

from stage5_condition import BUILD_ARMS
from stage5_data import Stage5Dataset

DATA = Path.home() / "ligo-data"
OUT = Path(__file__).parent / "outputs"
BATCH = 256
LR = 1e-3
SEED = 20260713  # Stage 3's seed, unchanged — same init, same shuffle
PATIENCE = 6


def ckpt_path(arm: str, clip: bool) -> Path:
    return DATA / f"stage5_cnn_{arm}{'' if clip else '_noclip'}.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 4 — train one ablated arm")
    ap.add_argument("--arm", choices=BUILD_ARMS, required=True)
    ap.add_argument("--no-clip", action="store_true", help="disable +-20 sigma saturation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    clip = not args.no_clip
    ckpt = ckpt_path(args.arm, clip)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"arm: {args.arm}  clip: {'+-20 sigma' if clip else 'OFF'}  -> {ckpt}\n")

    s2tr = Stage5Dataset(args.arm, "s2", "train", clip=clip)
    s3tr = Stage5Dataset(args.arm, "s3", "train", clip=clip)
    s2va = Stage5Dataset(args.arm, "s2", "val", clip=clip)
    s3va = Stage5Dataset(args.arm, "s3", "val", clip=clip)
    print(f"{s2tr}\n{s3tr}\n{s2va}\n{s3va}\n")

    train_loader = DataLoader(ConcatDataset([s2tr, s3tr]), batch_size=BATCH, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(ConcatDataset([s2va, s3va]), batch_size=1024, num_workers=2,
                            pin_memory=True, persistent_workers=True)

    model = Stage1CNN().to(device)  # STILL the same architecture. The data is the variable.
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
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_auc": val_auc, "seed": SEED, "arm": args.arm, "clip": clip,
                        "trained_on": f"stage5 arm {args.arm} (stage2+stage3 rows)",
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ckpt)
            star = "  * saved"
        print(f"epoch {epoch:3d}  train loss {running/nb:.4f}  val loss {val_loss:.4f}  "
              f"val AUC {val_auc:.4f}  ({time.time()-te:.0f}s){star}")
        if epoch - best_epoch >= PATIENCE:
            print(f"\nearly stop after {PATIENCE} flat epochs")
            break

    print(f"\ndone in {(time.time()-t0)/60:.1f} min — best val AUC {best_auc:.4f} "
          f"(epoch {best_epoch}) -> {ckpt}")
    print(f"  (Stages 2-3 gate for reference: 0.85. Arm {args.arm} "
          f"{'clears' if best_auc >= 0.85 else 'MISSES'} it — reported, not enforced: "
          f"an arm failing is a result, not a crash.)")

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))
    axl.plot(epochs, hist["train_loss"], "o-", color="tab:blue", label="train")
    axl.plot(epochs, hist["val_loss"], "o-", color="tab:red", label="val")
    axl.set_xlabel("epoch"), axl.set_ylabel("BCE loss"), axl.legend(), axl.grid(alpha=0.3)
    axl.set_title(f"Loss — arm {args.arm}")
    axa.plot(epochs, hist["val_auc"], "o-", color="tab:red")
    axa.axvline(best_epoch, color="k", lw=1, ls="--", alpha=0.6,
                label=f"checkpoint: epoch {best_epoch}, AUC {best_auc:.4f}")
    axa.axhline(0.9839, color="tab:gray", lw=1, ls=":",
                label="control (full conditioning): stage2-test AUC 0.9839")
    axa.set_xlabel("epoch"), axa.set_ylabel("val AUC"), axa.grid(alpha=0.3)
    axa.set_title(f"Model selection on val AUC — arm {args.arm}")
    axa.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"4_training_{args.arm}{'' if clip else '_noclip'}.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
