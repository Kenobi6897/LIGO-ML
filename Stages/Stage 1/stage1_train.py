"""Stage 1, Step 5 — train the 1D CNN.

Binary cross-entropy (`BCEWithLogitsLoss`), Adam, the 80k train split, model selection on
**validation AUC** — not accuracy, and not train loss. Accuracy is the metric this project
explicitly refuses ([[Stage 1#Step 6]]); train loss keeps falling long after the model has
started memorising. The checkpoint written to disk is whichever epoch scored the best AUC
on the 10k validation split, and the test split is never touched here — it belongs to
`stage1_eval.py` and is read exactly once, at the end, by a model that has already been
chosen.

THE CHECK THAT CAN FAIL: best val AUC must clear 0.90 (a coin-flip model scores 0.5; a
model that only finds loud signals scores ~0.8 on our SNR 4-20 mix). And — the project's
standing rule — if it clears 0.999 the run is flagged as SUSPICIOUS, not celebrated: on
half-weak injections that number means leakage until proven otherwise. The proof either
way is Step 6's efficiency curve, which must *fail* at low SNR like physics says it must.

The checkpoint goes to ~/ligo-data/ with the dataset, NOT the vault: model weights are
data, `.gitignore` already refuses `*.pt`, and the note's rule is code in git, data on
the PC.

Run (from WSL2, ~1 min/epoch, GPU):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_train.py                  # the real thing
    python stage1_train.py --epochs 1       # smoke test
"""

from __future__ import annotations

import argparse
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

from stage1_data import Stage1Dataset
from stage1_model import Stage1CNN

DEFAULT_CKPT = Path.home() / "ligo-data" / "stage1_cnn.pt"
OUT = Path(__file__).parent / "outputs"

BATCH = 256
LR = 1e-3
SEED = 20260712
PATIENCE = 6  # epochs without a val-AUC improvement before stopping


def run_val(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    """Validation loss and AUC. `model.eval()` matters twice here: dropout off, and
    BatchNorm using its running statistics instead of the batch's own."""
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    total, n = 0.0, 0
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            total += loss_fn(out, y).item()
            n += len(y)
            logits.append(out.cpu().numpy())
            labels.append(y.cpu().numpy())
    logits, labels = np.concatenate(logits).ravel(), np.concatenate(labels).ravel()
    return total / n, float(roc_auc_score(labels, logits))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 Step 5 — train the 1D CNN")
    ap.add_argument("--epochs", type=int, default=30, help="max epochs (early stop on val AUC)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed shapes every batch; let cudnn tune
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("device: CPU — this will be ~10-20x slower; the plan expects the 3070")

    train = Stage1Dataset("train")
    val = Stage1Dataset("val")
    print(f"{train}\n{val}\n")
    # Lazy HDF5 + workers is exactly the pattern stage1_data.py is built for: each worker
    # opens its own handle on first read; nothing is pickled, nothing shared over a fork.
    train_loader = DataLoader(
        train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val, batch_size=1024, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    model = Stage1CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage1CNN: {n_params:,} parameters\n")

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
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                args.ckpt,
            )
            star = "  * saved"
        print(
            f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val AUC {val_auc:.4f}  ({time.time()-te:.0f}s){star}"
        )

        if epoch - best_epoch >= PATIENCE:
            print(f"\nearly stop: no val-AUC improvement in {PATIENCE} epochs")
            break

    print(f"\ndone in {(time.time()-t0)/60:.1f} min — best val AUC {best_auc:.4f} "
          f"(epoch {best_epoch}) -> {args.ckpt}")

    # The check that can fail — and the standing rule about results that look too good.
    ok = best_auc >= 0.90
    print(f"\n  {'PASS' if ok else 'FAIL'}: best val AUC {best_auc:.4f} "
          f"{'>=' if ok else '<'} 0.90")
    if best_auc > 0.999:
        print("  SUSPICIOUS: AUC > 0.999 on half-weak injections. Assume leakage before "
              "genius — Step 6's low-SNR efficiency is the arbiter.")

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (axl, axa) = plt.subplots(1, 2, figsize=(13, 5))

    axl.plot(epochs, hist["train_loss"], "o-", color="tab:blue", label="train")
    axl.plot(epochs, hist["val_loss"], "o-", color="tab:red", label="validation")
    axl.set_xlabel("epoch")
    axl.set_ylabel("BCE loss")
    axl.set_title("Loss")
    axl.legend()
    axl.grid(alpha=0.3)

    axa.plot(epochs, hist["val_auc"], "o-", color="tab:red")
    axa.axvline(best_epoch, color="k", lw=1, ls="--", alpha=0.6,
                label=f"checkpoint: epoch {best_epoch}, AUC {best_auc:.4f}")
    axa.set_xlabel("epoch")
    axa.set_ylabel("validation AUC")
    axa.set_title("Model selection is on val AUC — never accuracy")
    axa.legend(loc="lower right")
    axa.grid(alpha=0.3)

    fig.suptitle(f"Stage 1 CNN training — {n_params:,} params, batch {args.batch}, Adam {args.lr}")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "5_training.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
