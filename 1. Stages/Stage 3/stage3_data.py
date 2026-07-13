"""Stage 3, Step 3 — reading the dataset. Same lazy-handle contract as Stages 1-2,
plus one new rule this stage's data forces: crops SATURATE at +-CLIP_SIGMA.

Whitened glitch crops peak at up to ~6,000 sigma (Extremely_Loud); Stage 2 never
exceeds 18. Trained on the raw mix, BCE punishes a near-linear CNN so hard for a
6,000-sigma negative that the optimizer shrinks the network's overall gain until
every logit fits in +-7 — glitch rejection learned, weak-signal sensitivity gone
(measured: stage2-test AUC 0.979 -> 0.934, SNR 6-8 efficiency 0.89 -> 0.37). The
clip is applied at READ time, identically for training and eval, so both arms see
the same inputs; the h5 stays raw and the bit-exact rebuild check still holds. At
+-20 sigma it is a no-op for every Stage 2 crop and for the injections; a rail at
20 sigma is still unmistakably a glitch (physical analogue: sensor saturation).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from stage3_dataset import DEFAULT_OUT

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
CLIP_SIGMA = 20.0


class Stage3Dataset(Dataset):
    """One split of `stage3.h5`: injected positives, plain negatives, glitch negatives."""

    def __init__(self, split: str = "train", path: Path | str = DEFAULT_OUT):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {list(SPLIT_IDS)}, got {split!r}")
        self.path = Path(path)
        self.split = split
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} — run stage3_dataset.py first")

        with h5py.File(self.path, "r") as f:
            self.rows = np.flatnonzero(f["split"][:] == SPLIT_IDS[split])
            self.y = f["y"][:][self.rows].astype(np.float32)
            self.snr = f["snr"][:][self.rows].astype(np.float32)
            self.is_glitch = f["is_glitch"][:][self.rows]
            self.glitch_label = f["glitch_label"][:][self.rows]
            self.glitch_snr = f["glitch_snr"][:][self.rows]
            self.gps = f["gps"][:][self.rows]
            self.glitch_names = list(f.attrs["glitch_labels"])
        self._f: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = np.asarray(self._file()["X"][int(self.rows[i])], dtype=np.float32)
        np.clip(x, -CLIP_SIGMA, CLIP_SIGMA, out=x)
        return torch.from_numpy(x), torch.tensor([self.y[i]])

    def __getstate__(self):
        return {**self.__dict__, "_f": None}

    def __repr__(self) -> str:
        pos = int(self.y.sum())
        ng = int(self.is_glitch.sum())
        return (f"Stage3Dataset(split={self.split!r}, n={len(self)}, pos={pos}, "
                f"glitch={ng}, plain_neg={len(self)-pos-ng}, path={self.path})")


if __name__ == "__main__":
    for s in SPLIT_IDS:
        ds = Stage3Dataset(s)
        x, y = ds[0]
        print(f"{ds}\n  x{tuple(x.shape)} std={x.std():.3f}")
