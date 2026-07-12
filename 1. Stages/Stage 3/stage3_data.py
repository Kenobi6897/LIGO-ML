"""Stage 3, Step 3 — reading the dataset. Same lazy-handle contract as Stages 1-2."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from stage3_dataset import DEFAULT_OUT

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


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
        x = self._file()["X"][int(self.rows[i])]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor([self.y[i]])

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
