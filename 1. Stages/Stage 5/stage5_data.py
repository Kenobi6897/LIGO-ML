"""Stage 5, Step 3 — reading an arm. Same lazy-handle contract as Stages 1-3.

One class, parameterised by the arm, so the training script and the eval script cannot
accidentally read a different arm's crops than they think they are reading. The +-20 sigma
read-time saturation is Stage 3's, applied here to every arm in that arm's own units —
identical read-time treatment is the price of "one variable per stage". `--no-clip` exists
because stage5_dataset_check.py may find that for some arm the clip is NOT a no-op, and an
arm that the clip bites is an arm the clip is a variable for.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from stage5_condition import BUILD_ARMS, CLIP_SIGMA

DATA = Path.home() / "ligo-data"
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class Stage5Dataset(Dataset):
    """One (arm, source, split) of the ablation: `X_{arm}` from stage5_{source}.h5."""

    def __init__(self, arm: str, source: str = "s2", split: str = "train",
                 clip: bool = True, path: Path | None = None):
        if arm not in BUILD_ARMS:
            raise ValueError(f"arm must be one of {BUILD_ARMS}, got {arm!r}")
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {list(SPLIT_IDS)}, got {split!r}")
        self.arm, self.source, self.split, self.clip = arm, source, split, clip
        self.path = Path(path) if path else DATA / f"stage5_{source}.h5"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} — run stage5_dataset.py --source {source}")
        self.key = f"X_{arm}"

        with h5py.File(self.path, "r") as f:
            self.rows = np.flatnonzero(f["split"][:] == SPLIT_IDS[split])
            self.y = f["y"][:][self.rows].astype(np.float32)
            self.snr = f["snr"][:][self.rows].astype(np.float32)
            self.gps = f["gps"][:][self.rows]
            if "is_glitch" in f:
                self.is_glitch = f["is_glitch"][:][self.rows]
                self.glitch_label = f["glitch_label"][:][self.rows]
                self.glitch_names = list(f.attrs["glitch_labels"])
            else:
                self.is_glitch = np.zeros(len(self.rows), dtype=np.int8)
                self.glitch_label = np.full(len(self.rows), -1, dtype=np.int16)
                self.glitch_names = []
        self._f: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = np.asarray(self._file()[self.key][int(self.rows[i])], dtype=np.float32)
        if self.clip:
            np.clip(x, -CLIP_SIGMA, CLIP_SIGMA, out=x)
        return torch.from_numpy(x), torch.tensor([self.y[i]])

    def __getstate__(self):
        return {**self.__dict__, "_f": None}

    def hours_of_noise(self) -> float:
        return float((self.y == 0).sum()) / 3600.0

    def __repr__(self) -> str:
        pos = int(self.y.sum())
        ng = int(self.is_glitch.sum())
        return (f"Stage5Dataset(arm={self.arm!r}, {self.source}/{self.split}, n={len(self)}, "
                f"pos={pos}, glitch={ng}, plain_neg={len(self)-pos-ng}, "
                f"clip={'20s' if self.clip else 'off'})")


if __name__ == "__main__":
    for arm in BUILD_ARMS:
        for src in ("s2", "s3"):
            ds = Stage5Dataset(arm, src, "train")
            x, y = ds[0]
            print(f"{ds}\n  x{tuple(x.shape)} std={x.std():.3f} max={x.abs().max():.1f}")
