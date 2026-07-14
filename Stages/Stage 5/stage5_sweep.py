"""Stage 5, Step 6 — is `bp` beating the control, or is it beating a random seed?

The single-seed run said: bandpass-only matches full conditioning on detection
(AUC 0.9829 vs 0.9839) and rejects glitches 3.5x better (0.048 vs 0.166 @ FAP 1e-2).
The second claim is the surprising one, and a surprising result from ONE seed is a
hypothesis, not a finding. So: retrain both arms across N seeds and look at the spread.

THE CONTROL IS RETRAINED HERE TOO, THROUGH THE IDENTICAL CODE PATH — same reader, same
clip, same loader, same optimiser, same early stopping — so the comparison cannot be an
artefact of `full` having come from stage3_train.py and `bp` from stage5_train.py. (Its
seed-20260713 run reproduces Stage 3's published checkpoint's numbers, which is the
evidence that the two paths agree: see stage5_eval.py's CHECK 1.)

Nothing here overwrites stage3_cnn.pt: ckpt_path() gives every non-default seed its own
file, and the control's own sweep checkpoints are `stage5_cnn_full_s*.pt`.

Run (~1.5 min per arm-seed on the 3070):
    python stage5_sweep.py --arms full bp --seeds 20260713 1 2 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage 1"))
from stage1_model import load_checkpoint

from stage5_data import Stage5Dataset
from stage5_eval import FAPS, SNR_EDGES, score
from stage5_train import ckpt_path

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 6 — seed sweep")
    ap.add_argument("--arms", nargs="+", default=["full", "bp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260713, 1, 2, 3])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- train whatever is missing -----------------------------------------------------
    print(f"training {len(args.arms)} arms x {len(args.seeds)} seeds "
          f"(skipping any checkpoint already on disk)\n")
    for arm in args.arms:
        for seed in args.seeds:
            p = ckpt_path(arm, True, seed)
            if p.exists():
                print(f"  arm {arm:4s} seed {seed}  (already trained)")
                continue
            subprocess.run(
                [sys.executable, str(HERE / "stage5_train.py"), "--arm", arm,
                 "--seed", str(seed), "--quiet"],
                check=True, cwd=HERE,
            )

    # --- score them all, exactly as stage5_eval does -----------------------------------
    print("\nscoring — thresholds at each run's own FAP quantile on stage2-val negatives\n")
    res: dict[str, dict[str, list[float]]] = {}
    for arm in args.arms:
        s2val = Stage5Dataset(arm, "s2", "val")
        s2test = Stage5Dataset(arm, "s2", "test")
        gtest = Stage5Dataset(arm, "s3", "test")
        gmask = gtest.is_glitch == 1
        pos = s2test.y == 1
        weak = pos & (s2test.snr >= 6) & (s2test.snr < 8)

        res[arm] = {"auc": [], "glitch_1e2": [], "glitch_1e3": [], "eff68_1e3": []}
        for seed in args.seeds:
            model, _ = load_checkpoint(ckpt_path(arm, True, seed))
            model.to(device)
            v = score(model, s2val, device)
            thr = {f: float(np.quantile(v[s2val.y == 0], 1 - f)) for f in FAPS}
            t2, t3 = score(model, s2test, device), score(model, gtest, device)
            res[arm]["auc"].append(float(roc_auc_score(s2test.y, t2)))
            res[arm]["glitch_1e2"].append(float((t3[gmask] > thr[1e-2]).mean()))
            res[arm]["glitch_1e3"].append(float((t3[gmask] > thr[1e-3]).mean()))
            res[arm]["eff68_1e3"].append(float((t2[weak] > thr[1e-3]).mean()))

    # --- the table ----------------------------------------------------------------------
    metrics = [("stage2-test AUC", "auc", "higher better"),
               ("glitch FA @1e-2", "glitch_1e2", "LOWER better"),
               ("glitch FA @1e-3", "glitch_1e3", "LOWER better"),
               ("eff SNR 6-8 @1e-3", "eff68_1e3", "higher better")]
    print(f"  {'metric':20s} {'arm':>5}  {'mean':>8} {'std':>8}  "
          f"{'min':>8} {'max':>8}   per-seed")
    for name, key, sense in metrics:
        print(f"  {name:20s} ({sense})")
        for arm in args.arms:
            v = np.array(res[arm][key])
            print(f"  {'':20s} {arm:>5}  {v.mean():8.4f} {v.std():8.4f}  "
                  f"{v.min():8.4f} {v.max():8.4f}   "
                  + " ".join(f"{x:.4f}" for x in v))

    # --- the verdict, as a check that can fail ------------------------------------------
    if set(args.arms) >= {"full", "bp"}:
        print()
        for key, name in (("glitch_1e2", "@1e-2"), ("glitch_1e3", "@1e-3")):
            b, f = np.array(res["bp"][key]), np.array(res["full"][key])
            sep = b.max() < f.min()
            gap = f.mean() - b.mean()
            pooled = np.sqrt((b.std() ** 2 + f.std() ** 2) / 2) + 1e-12
            print(f"  {'SEPARATED' if sep else 'OVERLAPPING'}: glitch FA {name} — "
                  f"bp {b.mean():.3f}+-{b.std():.3f} vs full {f.mean():.3f}+-{f.std():.3f}, "
                  f"gap {gap:+.3f} ({gap/pooled:.1f} pooled sd)")
        a_b, a_f = np.array(res["bp"]["auc"]), np.array(res["full"]["auc"])
        print(f"  detection: AUC bp {a_b.mean():.4f}+-{a_b.std():.4f} vs "
              f"full {a_f.mean():.4f}+-{a_f.std():.4f} — "
              f"gap {a_f.mean()-a_b.mean():+.4f}")
        print("\n  'SEPARATED' means every bp seed rejected glitches better than every full")
        print("  seed — the claim survives. 'OVERLAPPING' means the single-seed result was")
        print("  a seed, and the writeup must say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
