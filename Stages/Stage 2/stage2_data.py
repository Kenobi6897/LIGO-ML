"""Stage 2, Step 3 — reading the dataset. The torch `Dataset` both models eval/train on.

Byte-for-byte the Stage 1 reader pattern (lazy per-process HDF5 handle, nothing opened
in __init__, nothing pickled across a fork — see stage1_data.py for the full argument),
pointed at stage2.h5 and carrying the Stage 2 metadata (block, gps) that eval needs to
report false alarms per hour of real noise.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from stage2_dataset import DEFAULT_OUT

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class Stage2Dataset(Dataset):
    """One split of `stage2.h5`, read lazily, one segment per __getitem__."""

    def __init__(self, split: str = "train", path: Path | str = DEFAULT_OUT):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {list(SPLIT_IDS)}, got {split!r}")
        self.path = Path(path)
        self.split = split
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} — run stage2_dataset.py first")

        with h5py.File(self.path, "r") as f:
            self.rows = np.flatnonzero(f["split"][:] == SPLIT_IDS[split])
            self.y = f["y"][:][self.rows].astype(np.float32)
            self.snr = f["snr"][:][self.rows].astype(np.float32)
            self.m1 = f["m1"][:][self.rows].astype(np.float32)
            self.m2 = f["m2"][:][self.rows].astype(np.float32)
            self.block = f["block"][:][self.rows]
            self.gps = f["gps"][:][self.rows]

        self._f: h5py.File | None = None  # opened per-process, on first read

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        x = self._file()["X"][row]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor([self.y[i]])

    def __getstate__(self):
        return {**self.__dict__, "_f": None}

    def hours_of_noise(self) -> float:
        """Hours of real detector noise in this split's NEGATIVES — the honest
        denominator for a false-alarms-per-hour claim."""
        return float((self.y == 0).sum()) / 3600.0

    def __repr__(self) -> str:
        pos = int(self.y.sum())
        return (
            f"Stage2Dataset(split={self.split!r}, n={len(self)}, pos={pos}, "
            f"neg={len(self)-pos}, noise={self.hours_of_noise():.2f}h, path={self.path})"
        )


if __name__ == "__main__":
    for s in SPLIT_IDS:
        ds = Stage2Dataset(s)
        x, y = ds[0]
        print(f"{ds}\n  x{tuple(x.shape)} {x.dtype}  y={y.item():.0f}  "
              f"gps={ds.gps[0]:.0f}  mean={x.mean():+.3f} std={x.std():.3f}")
