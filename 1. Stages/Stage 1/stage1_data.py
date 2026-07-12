"""Stage 1, Step 4 — reading the dataset. The torch `Dataset` the CNN trains on.

    from stage1_data import Stage1Dataset
    train = Stage1Dataset(split="train")
    loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=4)

LAZY, BY REQUIREMENT — NOT BY TASTE
The full X is ~820 MB and the PC has 16 GB shared with Windows (WSL2 is capped at 10 GB by
`.wslconfig`). `np.load`/`f["X"][:]` would pull the entire array into RAM in every worker
process, and 4 workers would want 3.3 GB before a single batch was assembled. So
`__getitem__` reads one segment at a time — the file is chunked one-segment-per-chunk
precisely so that a random read costs exactly one 8 KB chunk.

THE h5py + DataLoader FOOTGUN
An open HDF5 handle cannot be pickled, and a handle inherited across a fork is *not* safe
to read from concurrently. Opening the file in `__init__` therefore either crashes on
spawn or silently corrupts reads on fork — the classic symptom being garbage or duplicated
batches that look like a bad model rather than a bad reader.

So: open NOTHING in `__init__`. The handle is created on first access *inside* whichever
process ends up doing the reading, and cached there. This is the standard pattern and the
reason this class looks more defensive than it needs to.

Labels are float32 of shape (1,) to match `BCEWithLogitsLoss` against a Dense(1) head.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from stage1_dataset import DEFAULT_OUT

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class Stage1Dataset(Dataset):
    """One split of `stage1.h5`, read lazily, one segment per __getitem__."""

    def __init__(self, split: str = "train", path: Path | str = DEFAULT_OUT):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {list(SPLIT_IDS)}, got {split!r}")
        self.path = Path(path)
        self.split = split
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} — run stage1_dataset.py first")

        # Metadata only: a few MB, and worth having in RAM. X is NOT touched here.
        with h5py.File(self.path, "r") as f:
            self.rows = np.flatnonzero(f["split"][:] == SPLIT_IDS[split])
            self.y = f["y"][:][self.rows].astype(np.float32)
            self.snr = f["snr"][:][self.rows].astype(np.float32)
            self.m1 = f["m1"][:][self.rows].astype(np.float32)
            self.m2 = f["m2"][:][self.rows].astype(np.float32)

        self._f: h5py.File | None = None  # opened per-process, on first read

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        x = self._file()["X"][row]  # (1, 2048) float32 — one chunk off disk
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor([self.y[i]])

    def __getstate__(self):
        # Never pickle the handle across a process boundary; the child reopens its own.
        return {**self.__dict__, "_f": None}

    def __repr__(self) -> str:
        pos = int(self.y.sum())
        return (
            f"Stage1Dataset(split={self.split!r}, n={len(self)}, "
            f"pos={pos}, neg={len(self)-pos}, path={self.path})"
        )


if __name__ == "__main__":
    for s in SPLIT_IDS:
        ds = Stage1Dataset(s)
        x, y = ds[0]
        print(f"{ds}\n  x{tuple(x.shape)} {x.dtype}  y={y.item():.0f}  "
              f"snr={ds.snr[0]:.1f}  mean={x.mean():+.3f} std={x.std():.3f}")
