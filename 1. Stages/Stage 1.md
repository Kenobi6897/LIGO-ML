---
tags: [ligo, machine-learning, stage-1, plan]
status: todo
created: 2026-07-11
updated: 2026-07-11
parent: "[[GW Signal Classifier - Brainstorm]]"
---

# Stage 1 — Injections + the first 1D CNN

**Machine:** 🖥️ **Desktop PC** (Ryzen 7 3700X / RTX 3070 / 16 GB), inside WSL2
**Depends on:** [[Setup - Desktop PC]] — **all four parts done**
**Previous:** [[Stage 0]] ✅

> **The goal:** on *simulated Gaussian noise*, train a 1D CNN to detect injected chirps, and plot detection efficiency vs. SNR. Reproduce the Gabbard figure.
>
> **Success = we land near matched filtering.** That's the "our pipeline is correct" signal. Not beating it — [[GW Signal Classifier - Brainstorm|that's a theorem]].

---

## 0. Setup — what you need on the PC

### 🚧 Gate: [[Setup - Desktop PC]] must be finished first
Stage 1 cannot start until all four of these are true. **Verify them, don't assume them:**

```bash
nvidia-smi                                              # reports RTX 3070
python -c "import torch; print(torch.cuda.is_available())"   # True
python -c "import lal, lalsimulation; print('ok')"      # lalsuite imports
cat /proc/meminfo | head -1                             # ~10GB, i.e. .wslconfig applied
```

**`import lal` is the whole reason WSL2 exists.** If it fails, nothing below is possible — Windows has no `lalsuite` wheels.

### ➕ Extra packages beyond the setup note
[[Setup - Desktop PC]] Part 3 installs `gwpy pycbc numpy scipy matplotlib h5py torch`. Stage 1 also needs:

```bash
source ~/venvs/ligo/bin/activate
pip install scikit-learn tqdm
```

- **`scikit-learn`** — `roc_curve` / `roc_auc_score`. We are explicitly *not* evaluating on accuracy ([[GW Signal Classifier - Brainstorm|see metrics]]), so this isn't optional.
- **`tqdm`** — dataset generation is a long loop; you want a progress bar rather than a silent terminal.

*(Optional, later: `tensorboard` if training runs get long enough to want curves. They probably won't — this model is small.)*

### 📁 Where things live
| | |
|---|---|
| **Code** | `~/ligo-ml/` inside WSL2 — clone from git, edit from either machine |
| **Data** | `~/ligo-data/` inside WSL2 — **never in the vault, never in git** |
| **Notes** | the vault (Windows side). WSL2 can reach it at `/mnt/c/Users/tmloc/Documents/LIGO-ML` |

- [ ] `mkdir -p ~/ligo-data`
- [ ] Port [[Stage 0]]'s `condition()` logic across — but **fix the PSD leak first** (see below).

### 🖥️ What actually uses the GPU
**Almost nothing.** Be clear-eyed about this:

| Work | Hardware | Notes |
|---|---|---|
| Waveform generation (`pycbc`) | **CPU** (3700X) | Embarrassingly parallel — use `multiprocessing`, all 16 threads |
| Noise generation, whitening | **CPU** | Same |
| Dataset write (HDF5) | **Disk** | |
| **CNN training** | **GPU** (3070) | Minutes, not hours. ~2 MB per batch. |

The 3070 is **wildly overkill** for this model. **Don't let it shape the architecture.** The 3700X will be doing the heavy lifting during dataset generation, and that's the part that'll take real time.

---

## 1. Decisions to pin before writing code

Pin these now — changing them later means regenerating the dataset.

| Decision | Recommendation | Why |
|---|---|---|
| **Segment length** | **1 s** | Long enough to hold the audible inspiral; matches the literature. |
| **Sample rate** | **2048 Hz** | Downsample from 4096. Our band is 30–350 Hz, so 2048 is ample (Nyquist 1024) and **halves the dataset size** — which matters at 16 GB. |
| **Channels** | **Start with H1 only (1 channel)** | Get the pipeline correct first. Add L1 as a 2nd channel *after* the baseline works — [[Stage 0]]'s overlay is the argument for it, but it's a Stage 2 refinement. |
| **Waveform** | `IMRPhenomD` | Fast, inspiral-merger-ringdown, standard. |
| **Mass range** | m1, m2 ∈ **[10, 50] M☉**, sampled uniformly | Brackets GW150914 (36 + 29). Keeps the template bank small. |
| **SNR range** | **train across SNR 4–20**, evaluate per-SNR | The interesting regime is **low** SNR, where MF starts to struggle. Training only on loud signals produces a model that can only find loud signals. |
| **Dataset size** | **100k segments**, 50/50 | ≈ 820 MB at 2048 Hz float32. Fits. |
| **Noise** | **Simulated Gaussian**, coloured by `aLIGOZeroDetHighPower` | Stage 1 is the *clean-room* experiment. Real noise is **Stage 2** — that's the whole point of the staging. |

---

## 2. The build

### Step 1 — Noise generator
```python
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
```
Coloured Gaussian noise at the aLIGO design sensitivity. This is the **same** generator for positives and negatives — a positive is just a negative with a waveform added.

- [ ] Generate noise segments
- [ ] Sanity check: PSD of generated noise should match the design curve

### Step 2 — Waveform generator + SNR scaling
```python
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma          # optimal SNR of a waveform vs a PSD
```
For each positive:
1. Sample masses, generate the waveform
2. Compute its **optimal SNR** against the PSD (`sigma`)
3. **Rescale** it to hit a target SNR drawn from [4, 20]
4. Add it into a noise segment

- [ ] ⚠️ **Randomise where the merger sits in the 1 s window.** See [[#Footguns]].

### Step 3 — Condition (whiten → bandpass → crop)
Port from [[Stage 0]] — **but with the PSD fix.** See [[#Footguns]].

Apply **identically** to positives and negatives. Same function, same PSD, no branches.

### Step 4 — Write the dataset
- [ ] HDF5 at `~/ligo-data/stage1.h5`, arrays `X` (N, 1, 2048) and `y` (N,)
- [ ] **Lazy load in `Dataset.__getitem__`.** Do NOT `np.load` the whole thing into RAM — 16 GB, shared with Windows.
- [ ] Also store **per-segment SNR** and masses — you need SNR at eval time to plot efficiency curves, and you cannot recover it later.
- [ ] Parallelise generation with `multiprocessing` — 16 threads on the 3700X.

### Step 5 — The 1D CNN
Modest architecture — this is the Gabbard-class problem, not ImageNet:
```
Conv1d → BatchNorm → ReLU → MaxPool     (×3–4 blocks, widening)
→ Flatten → Dense → Dropout → Dense(1)
```
- [ ] Binary cross-entropy, Adam
- [ ] **Hold out real GW150914** as a sanity check — if the model can't fire on it, the model is broken

### Step 6 — Evaluate (NOT on accuracy)
- [ ] **ROC / AUC** — `sklearn.metrics`
- [ ] **The money plot: detection efficiency (TPR) vs. injected SNR, at fixed false-alarm rate.** This is the Gabbard figure. Accuracy is meaningless here; a model that says "noise" always scores 50% and has learned nothing.
- [ ] **Report FAR as false positives per hour of held-out noise** — *not* per year. We don't have the background to support a per-year claim, and faking one is worse than not making one.

### Step 7 — The payoff figure 🎁
- [ ] **Plot the first-layer conv kernels.**

A 1D convolution **is** a matched filter — it slides a kernel along the signal computing a dot product at each lag, which is exactly what correlating against a template does. So layer 1 is, functionally, **a learned template bank**.

**You should see chirp-shaped filters the network invented on its own**, without ever being told what a black hole is. See [[1D vs 2D - decision explained]] — this is the single best figure this project will produce, and it's the reason we went 1D.

---

## 3. Footguns

### ⚠️⚠️ The PSD leak — this is THE one
[[Stage 0]] used gwpy's `.whiten()`, which **estimates the PSD from the very segment it is whitening.**

That was fine there. **It is catastrophic here.** A segment with a loud injection gets whitened *differently* than a noise segment — the whitening operation itself **carries a fingerprint of the signal**. The CNN learns the fingerprint, not the chirp. You get **99% AUC and a model that has learned nothing.**

**Fix:** estimate the PSD from a **separate stretch of data** (or, in Stage 1, use the *known* design PSD directly — we generated the noise, so we know its spectrum exactly). Apply the **same** PSD to every segment.

### ⚠️ Merger position
If every positive has the merger at the same sample index, **the CNN learns the index, not the shape.** It will score brilliantly and generalise to nothing.

**Fix:** randomise the merger time within the window (e.g. uniformly in the last 0.3 s).

### ⚠️ Per-segment normalisation
Normalising each segment to unit variance **leaks the injection** — adding a signal changes the variance. Same class of bug as the PSD leak.

**Fix:** don't. Whitening already sets the scale.

### 🚩 The tell
> **If AUC is suspiciously high, assume leakage before assuming genius.**

A correct Stage 1 model should be **roughly comparable to matched filtering — and clearly *fail* at low SNR.** If it detects SNR-4 signals perfectly, that is not a triumph. That is a bug.

---

## Done when

- [ ] Detection efficiency vs. SNR curve exists, and it **degrades at low SNR like it should**
- [ ] ROC/AUC computed on a held-out set
- [ ] FAR quoted in false positives **per hour**, honestly
- [ ] The model fires on real **GW150914**
- [ ] First-layer kernels plotted, and **they look like chirps**
- [ ] Result is in the same ballpark as the [[GW Signal Classifier - Brainstorm|Gabbard reference]]

→ Then **Stage 2**: swap simulated Gaussian noise for **real O3 noise**. Expect performance to drop. Understanding *why* is the actual project.
