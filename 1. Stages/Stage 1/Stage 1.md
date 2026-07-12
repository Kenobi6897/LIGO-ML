---
tags: [ligo, machine-learning, stage-1, plan]
status: in-progress
created: 2026-07-11
updated: 2026-07-12
progress: "Steps 1-3 done and checked. Step 4 (dataset) in progress: writer + lazy Dataset + check written, writer runs, but 2 of the check's 4 properties are RED — bit-exact rebuild, and SNR calibration. Not yet generated the real 100k."
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

### ✅ Gate: [[Setup - Desktop PC]] — **PASSED 2026-07-12. Stage 1 is unblocked.**

> [!success] Gate run on the PC (`DESKTOP-P6POAN4`), 2026-07-12 — **all checks green**
> | Check | Result |
> |---|---|
> | RTX 3070 visible | ✅ driver 596.49 |
> | WSL2 platform + **distro** | ✅ **Ubuntu 26.04 LTS**, WSL 2.7.10, kernel 6.18.33.2 |
> | `nvidia-smi` inside WSL2 | ✅ RTX 3070 — **with no CUDA toolkit installed** |
> | `torch.cuda.is_available()` | ✅ `True` — torch **2.13.0+cu130** |
> | GPU `conv1d`, Stage-1 shapes | ✅ `(256, 1, 2048) → (256, 16, 1985)` ran on the 3070 |
> | `import lal, lalsimulation` | ✅ lal **7.7.1** · pycbc **2.11.0** · gwpy **4.0.1** |
> | `.wslconfig` 10 GB cap | ✅ `MemTotal` ≈ 9.7 GiB |
>
> **Two deviations from the setup plan — the plan was wrong, the environment is right:**
> - **The venv is Python 3.13.14, not the system 3.14.** Ubuntu 26.04 ships 3.14 and **`pycbc` has no
>   3.14 wheel** (lalsuite and torch both do — pycbc is the lone holdout). The venv is pinned to 3.13
>   via `uv`. **Don't "upgrade" it to the system interpreter** — that silently breaks pycbc.
> - **No CUDA toolkit was installed, and none is needed.** Torch's wheels bundle their own CUDA
>   runtime; the toolkit only supplies `nvcc`, and we compile no custom kernels.

```bash
source ~/venvs/ligo/bin/activate                        # CPython 3.13.14, via uv
nvidia-smi                                              # reports RTX 3070
python -c "import torch; print(torch.cuda.is_available())"   # True
python -c "import lal, lalsimulation; print('ok')"      # lalsuite imports
head -1 /proc/meminfo                                   # ~10GB, i.e. .wslconfig applied
```

**`import lal` is the whole reason WSL2 exists.** If it ever stops working, nothing below is possible — Windows has no `lalsuite` wheels.

### ➕ Extra packages beyond the setup note
✅ **Already installed** — `scikit-learn` (1.9.0) and `tqdm` went in with the Part 3 stack. Nothing to do here.

- **`scikit-learn`** — `roc_curve` / `roc_auc_score`. We are explicitly *not* evaluating on accuracy ([[GW Signal Classifier - Brainstorm|see metrics]]), so this isn't optional.
- **`tqdm`** — dataset generation is a long loop; you want a progress bar rather than a silent terminal.

*(Optional, later: `tensorboard` if training runs get long enough to want curves. They probably won't — this model is small.)*

### 📁 Where things live
| | |
|---|---|
| **Code** | **`1. Stages/Stage 1/` — in the vault, in git.** Scripts sit in the same folder as this note. |
| **Data** | `~/ligo-data/` inside WSL2 — **never in the vault, never in git** |
| **Notes** | the vault (Windows side). WSL2 reaches it at `/mnt/c/Users/locke/Documents/LIGO-ML` |

Scripts are run from WSL2 against the Windows-side vault — one copy of the code, no syncing:
```bash
source ~/venvs/ligo/bin/activate
cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
python stage1_noise_check.py
```

- [ ] `mkdir -p ~/ligo-data`
- [x] Port [[Stage 0]]'s `condition()` logic across — but **fix the PSD leak first** (see below).

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

> [!tip] Every step ships with a check that can fail
> Each of Steps 1–3 is a `*_check.py` that compares the code against something we **independently know** to be true. This is not ceremony. Steps 2 and 3 each caught a real bug that produced *plausible numbers* rather than a crash — see [[#🐛 Bugs the checks caught]]. On this project, a silent wrong answer is the default failure mode, not the exception.

### ✅ Step 1 — Noise generator — `stage1_noise_check.py`
```python
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
```
Coloured Gaussian noise at aLIGO design sensitivity. The **same** generator makes positives and negatives — a positive is just a negative with a waveform added. If its spectrum is wrong, every injected SNR downstream is wrong with it.

- [x] Generate noise segments
- [x] Sanity check: PSD of generated noise matches the design curve

**Result: PASS.** Measured (Welch) / design PSD ratio over 30–350 Hz: **median 0.999** (5th–95th pct 0.805–1.215 — that spread is Welch's χ² scatter, not bias; the *median* is the test).

![[1_psd_check.png]]

### ✅ Step 2 — Waveform generator + SNR scaling — `stage1_injection_check.py`
```python
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma          # optimal SNR of a waveform vs a PSD
```
1. Sample masses, generate the waveform (`IMRPhenomD`)
2. Compute its **optimal SNR** against the PSD (`sigma`)
3. **Rescale** to hit a target SNR from [4, 20]
4. Add it into a noise segment

- [x] ⚠️ **Randomise where the merger sits in the 1 s window** — `rng.uniform(0.7, 0.95)`. See [[#Footguns]].

**Result: PASS.** Request an SNR, then measure it back with an independent matched filter:

| target | recovered (mean ± sd) | bias |
|---|---|---|
| 4 | 4.49 ± 0.81 | 1.12× |
| 8 | 7.72 ± 0.81 | 0.97× |
| 12 | 12.03 ± 1.02 | 1.00× |
| 20 | 20.00 ± 0.99 | 1.00× |

Two things worth noticing. **σ ≈ 1.0 at every SNR** — matched-filter SNR has unit variance by construction, so getting it back is strong evidence the whitening and normalisation are right. And the **high bias at low SNR is physics, not a bug**: with a weak signal the filter peaks on a noise fluctuation near it. That is *precisely why low-SNR detection is hard*, and it is what the money plot in Step 6 is measuring.

![[2_injection_check.png]]

### ✅ Step 3 — Condition (whiten → bandpass → crop) — `stage1_condition.py`
The most dangerous function in the project, so it is the smallest. **The rule: `condition()` never looks at the segment's own content.** It whitens with the *known* design PSD and one global scale constant — so it is the same fixed filter for every segment, positive or negative. No branches.

**A 4 s buffer for a 1 s segment.** Whitening and band-passing are convolutions, so they corrupt both ends of whatever you hand them. Generate 4 s, condition it, crop the central 1 s → 2048 samples, safely 1.5 s clear of either edge. *Crop from a longer buffer; never try to fix the edges of a short one.* (`inverse_spectrum_truncation` bounds the whitening filter to 0.5 s so the corruption is a length we can afford to throw away.)

- [x] Whiten with the **known** design PSD — never a per-segment estimate
- [x] Bandpass 30–350 Hz (`highpass_fir` / `lowpass_fir`, zero-phase)
- [x] Crop the corrupted edges away
- [x] Global scale constant, measured on **noise only** — *not* per-segment ([[#Footguns|the unit-variance leak]])

#### The leak test — `stage1_condition_check.py`
How do you *prove* a whitener isn't leaking? State it as a property that can fail:

> **A signal-blind operator applies the same fixed filter to every segment — so the change a signal makes cannot depend on the noise it landed in.**
> $$C(n_1 + h) - C(n_1) \;=\; C(n_2 + h) - C(n_2)$$

Same signal, two different noise realisations. If the whitener is built *from the segment*, the two responses differ — and that difference is a signal-dependent stamp on the noise floor, far easier for a CNN to learn than a chirp.

| | disagreement |
|---|---|
| `condition()` — known PSD | **1×10⁻⁷** — machine precision. Signal-blind. |
| `condition_leaky()` — per-segment PSD ([[Stage 0]]'s `.whiten()`) | **36%** — the same signal is conditioned *differently* depending on its noise. |

**Result: PASS** on all four checks — noise whitened flat in band (tilt 4.5× → 0.86×), conditioned noise std 0.978 from one global constant, `condition()` signal-blind, and `condition_leaky()` demonstrably not.

In the right-hand panel the two chirps (green, black-dashed) lie exactly on top of each other — same signal, different noise, identical response. The purple trace is the leaky whitener's residue, and it visibly **tracks the chirp**. That purple line is what a 99%-AUC model would actually be detecting.

![[3_condition_check.png]]

> [!note] The demo has to survive a hostile reading
> The first version of this test compared against `C(h)` — the whitener applied to a **pure signal with no noise**. It "failed" the leaky whitener by a factor of 10¹⁰, which was flattering and meaningless: no detector ever produces a noiseless signal, and a per-segment whitener is undefined on one. The version above only ever feeds both operators inputs the real pipeline actually sees. It is a weaker-looking number and a much stronger claim.

### 🚧 Step 4 — Write the dataset — **IN PROGRESS**
Per segment: generate a **4 s** noise buffer → (if positive) inject a waveform scaled to a target SNR → `condition()` → store the central **2048** samples. Positives and negatives go through the identical call.

**Code written, writer runs, check is red.** Three files:

| file | what it is |
|---|---|
| `stage1_dataset.py` | the writer — `multiprocessing` over 16 workers, streams into HDF5 |
| `stage1_data.py` | `Stage1Dataset` — the lazy torch `Dataset` the CNN will train on |
| `stage1_dataset_check.py` | the check that can fail. **Currently failing — see below.** |

- [x] HDF5 at `~/ligo-data/stage1.h5`, arrays `X` (N, 1, 2048) and `y` (N,)
- [x] **Lazy load in `Dataset.__getitem__`.** Handle opened per-process on first read, never in `__init__` — an open h5py handle can't be pickled to a `DataLoader` worker, and one inherited across a fork is unsafe to read concurrently.
- [x] Also store **per-segment SNR** and masses — plus `merger_pos`, the noise `seed`, and `split`. You need SNR at eval time and cannot recover it later.
- [x] Parallelise generation with `multiprocessing` — 16 threads on the 3700X.
- [ ] ⚠️ **Get the check green.** Two open failures.
- [ ] Run the real 100k.

**Throughput:** 19 seg/s on the smoke run → **100k is ~90 min**, not the few minutes the plan assumed. Worth a look before committing to it; ~20 s of that is per-worker warm-up (each worker recomputes `_norm()` and its PSDs).

#### Two decisions this step had to make that Steps 1–3 didn't
Both follow from the crop, and both would have been **silent** errors:

1. **The merger is placed relative to the CROP, not the buffer.** `merger_pos ∈ [0.7, 0.95)` is a fraction of the 1 s window the CNN sees, so it maps to buffer sample `CROP_START + merger_pos·CROP_N`. Placing it at 0.7–0.95 of the *4 s buffer* would put the merger at ~3 s — **outside the crop.** Every "positive" would be a segment whose signal had been cropped away. It would train, and it would learn nothing.
	- The early inspiral **is** still injected into the buffer outside the crop, deliberately. A real 1 s segment cut from a detector stream has the earlier inspiral present on either side of it, and the 0.5 s whitening filter reaches into it. We discard those samples; we don't pretend they were never there.
2. **SNR is integrated over 30–350 Hz** — the band the CNN actually sees. `condition()` band-passes, so power outside the band is filtered away before the network is handed the segment. Step 2's check integrated from 30 Hz with **no upper cutoff**, which counts ringdown power above 350 Hz that never reaches the model — overstating the SNR of exactly the *lightest* systems, whose merger frequency is highest. The money plot's x-axis has to mean "the SNR available in the data the model is given", so `sigma` now takes `high_frequency_cutoff=350`.

#### 🔴 The check is failing — two open items
Structure passes clean (shapes, dtypes, 50/50, finite, negatives carry no waveform metadata, merger randomised, **noise seeds unique and no realisation shared across splits**, 80/10/10). The two that matter are red:

**(a) Rebuild is not bit-exact.** Rebuilding a stored row from its metadata alone (seed, masses, SNR, merger position) reproduces it to `max|Δ| = 1.3×10⁻⁵`, not `0`. On a segment whose std is 1.0 that's a **relative 10⁻⁵ — far too big for float32 round-off** (~10⁻⁷). Suspicion is FFTW planning differing between a Pool worker and the main process, but **that is a guess and it has not been demonstrated.** Do not relax the tolerance to make this pass until the cause is known — a writer that shuffled rows, or a parameter that never made it into the segment, would look exactly like this.

**(b) The SNR calibration check is measuring the wrong thing.** It asserted `‖C(h)‖₂ ≈ requested SNR`, and got a median **4360×** with a **6× spread** across segments. The spread is the tell: a pure scale error would be a *constant* ratio, so this is mass-dependent and the identity itself is wrong.
	- **Why it's wrong:** `‖C(h)‖₂ = SNR` only holds for **white, unit-variance** noise. Conditioned noise here is unit-variance but **band-limited** (30–350 Hz of a 0–1024 Hz band), so its samples are correlated and the identity doesn't apply.
	- **The fix (not yet written):** normalise against the **empirical background**. Correlate the conditioned template `C(h)` against a few hundred conditioned *negatives from the file itself* to get σ_bg, then the optimal SNR is `‖C(h)‖² / σ_bg`. Self-normalising, assumption-free, and it measures the SNR the way a detection statistic actually would. Expect it to come back ≈ requested, possibly biased high at low SNR — which would be [[#✅ Step 2 — Waveform generator + SNR scaling — `stage1_injection_check.py`|the same physics Step 2 saw]], not a bug.
	- Conditioning the **pure waveform** to get `C(h)` is legal, and only because [[#✅ Step 3 — Condition (whiten → bandpass → crop) — `stage1_condition.py`|`condition()` is signal-blind]]. Hand a noiseless signal to `condition_leaky()` and it isn't even well-defined.

> [!warning] Neither failure is cosmetic
> **(b) makes the x-axis of the money plot unverified.** Until it's green we do not know that a segment labelled "SNR 8" contains an SNR-8 signal — and that axis is the entire result of Stage 1.

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

## 🐛 Bugs the checks caught

Both of these were found by Step 2's check. **Neither one crashed.** Both produced confident, plausible-looking numbers — which is the entire argument for writing a check that compares against something you independently know.

### 1. `resize()` threw the merger away
At 10–50 M☉ from 30 Hz, an `IMRPhenomD` waveform is **longer than the 1 s window** — often several seconds. The obvious `hp.resize(n)` keeps the **first** *n* samples: the quiet early inspiral. It silently discards the merger — the loudest part, and the only part the window actually sees.

So `sigma()` was normalising against a waveform that wasn't in/ the data. Recovered SNR came back biased 0.42–0.95×, *worsening with SNR*.

**Fix:** place the waveform by aligning its **amplitude peak** to the target merger time, and let the early inspiral fall off the front of the window. Compute `sigma` on the **placed snippet** — so "SNR 12" means *the SNR of the signal actually present in the segment*, which is the only definition the money plot's x-axis can be held to.

### 2. The crop deleted the answer
`recover_snr()` cropped the edges of the SNR time series — an innocent, standard-looking `[edge:-edge]`. But the template is placed at the **same alignment as the signal**, so the matched filter peaks at **lag zero — index 0**. The crop removed precisely the peak being measured.

It returned a flat **~3.6 for every target SNR**: the pure-noise background, reported with total confidence. Not an error, not a NaN — just a wrong number that looked like a number.

> [!warning] This is the project's stated failure mode wearing different clothes
> Neither bug was a leak, but both were the *same shape* of problem: code that runs clean and lies. The only reason either was caught is that the check compared against a value known independently. **Write the check that can fail.**

---

## Done when

- [ ] Detection efficiency vs. SNR curve exists, and it **degrades at low SNR like it should**
- [ ] ROC/AUC computed on a held-out set
- [ ] FAR quoted in false positives **per hour**, honestly
- [ ] The model fires on real **GW150914**
- [ ] First-layer kernels plotted, and **they look like chirps**
- [ ] Result is in the same ballpark as the [[GW Signal Classifier - Brainstorm|Gabbard reference]]

→ Then **Stage 2**: swap simulated Gaussian noise for **real O3 noise**. Expect performance to drop. Understanding *why* is the actual project.
