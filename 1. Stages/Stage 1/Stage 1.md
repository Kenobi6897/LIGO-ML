---
tags: [ligo, machine-learning, stage-1, plan]
status: done
created: 2026-07-11
updated: 2026-07-12
progress: "COMPLETE — all 7 steps done, every check green. Test AUC 0.9877; efficiency degrades at low SNR exactly as it must (0.50 at SNR 4-6 @ FAP 1e-2, below the Neyman-Pearson ceiling everywhere); fires on real GW150914 above all 112 background segments; 15/16 first-layer kernels peak in the analysis band. Best pre-paid Stage 2 fact: real O1 noise scores median logit +17.5 vs -3 on simulated negatives."
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

- [x] `mkdir -p ~/ligo-data` — the writer creates it; `stage1.h5` lives there.
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

### ✅ Step 4 — Write the dataset — `stage1_dataset_check.py`
Per segment: generate a **4 s** noise buffer → (if positive) inject a waveform scaled to a target SNR → `condition()` → store the central **2048** samples. Positives and negatives go through the identical call.

**Result: PASS — all 17 checks.** Both of the failures logged here earlier turned out to be real, and **neither was the thing the note said it was.** Written up below, because being wrong twice in the same predictable direction is the most useful thing in this file. Three files:

| file | what it is |
|---|---|
| `stage1_dataset.py` | the writer — `multiprocessing` over 16 workers, streams into HDF5 |
| `stage1_data.py` | `Stage1Dataset` — the lazy torch `Dataset` the CNN will train on |
| `stage1_dataset_check.py` | the check that can fail. **Green: structure, bit-exact rebuild, SNR calibration, no variance leak.** |

- [x] HDF5 at `~/ligo-data/stage1.h5`, arrays `X` (N, 1, 2048) and `y` (N,)
- [x] **Lazy load in `Dataset.__getitem__`.** Handle opened per-process on first read, never in `__init__` — an open h5py handle can't be pickled to a `DataLoader` worker, and one inherited across a fork is unsafe to read concurrently.
- [x] Also store **per-segment SNR** and masses — plus `merger_pos`, the noise `seed`, and `split`. You need SNR at eval time and cannot recover it later.
- [x] Parallelise generation with `multiprocessing` — 16 threads on the 3700X.
- [x] **SNR calibration green** — the x-axis of the money plot is verified. See below.
- [x] ⚠️ **Get the check green.** **All 17 checks pass**, including the bit-exact rebuild.
- [x] Run the real 100k.

**The real run: 100,000 segments in 68 min** (24 seg/s, 16 workers on the 3700X) → **829 MB** at `~/ligo-data/stage1.h5`, 50,000 + / 50,000 −, split 80,000 / 10,000 / 10,000. Not the few minutes the plan assumed — the CPU was always going to be the bottleneck here, and it was.

> [!tip] The parent must not touch FFTW before it forks
> `multiprocessing` forks, and a `Pool` forked from a parent that has *already run an FFT* **deadlocks** — the child inherits FFTW/lal planner state that is not safe across a fork. `stage1_dataset.py` gets this right by accident of structure (it draws specs, then forks, and only ever does DSP inside workers). Anything added to `main()` before the `Pool` must keep it that way. A debug script that built one segment first, then forked, hung solid.

#### Two decisions this step had to make that Steps 1–3 didn't
Both follow from the crop, and both would have been **silent** errors:

1. **The merger is placed relative to the CROP, not the buffer.** `merger_pos ∈ [0.7, 0.95)` is a fraction of the 1 s window the CNN sees, so it maps to buffer sample `CROP_START + merger_pos·CROP_N`. Placing it at 0.7–0.95 of the *4 s buffer* would put the merger at ~3 s — **outside the crop.** Every "positive" would be a segment whose signal had been cropped away. It would train, and it would learn nothing.
	- The early inspiral **is** still injected into the buffer outside the crop, deliberately. A real 1 s segment cut from a detector stream has the earlier inspiral present on either side of it, and the 0.5 s whitening filter reaches into it. We discard those samples; we don't pretend they were never there.
2. **SNR is integrated over 30–350 Hz** — the band the CNN actually sees. `condition()` band-passes, so power outside the band is filtered away before the network is handed the segment. Step 2's check integrated from 30 Hz with **no upper cutoff**, which counts ringdown power above 350 Hz that never reaches the model — overstating the SNR of exactly the *lightest* systems, whose merger frequency is highest. The money plot's x-axis has to mean "the SNR available in the data the model is given", so `sigma` now takes `high_frequency_cutoff=350`.

#### ✅ (b) SNR calibration — **FIXED, and it was hiding a real bug**
Structure passes clean (shapes, dtypes, 50/50, finite, negatives carry no waveform metadata, merger randomised, **noise seeds unique and no realisation shared across splits**, 80/10/10).

The check asserted `‖C(h)‖₂ ≈ requested SNR` and got a median **4360×** with a **6× spread**. Two separate things were wrong, and the spread was the tell that told us so — a pure scale error would have been a *constant* ratio.

**1. The check never applied the SNR rescale.** `signal_only()` conditioned the **raw** `IMRPhenomD` waveform, at pycbc's default distance. But `build_segment` stores `noise + placed × (snr/σ)`, and `condition()` is linear, so the signal term of a stored row is `C(placed × snr/σ)`, **not** `C(placed)`. The check was measuring a signal that is not in the file — one whose optimal SNR is in the *thousands*. That is the 4360×, and because σ depends on the masses, that is also the 6× spread. **A bug in the check, not in the writer** — but a check that measures the wrong signal is worth exactly nothing, and it had been reported as a *dataset* failure.

**2. `‖C(h)‖₂ = SNR` was the wrong identity anyway.** It needs the conditioned noise to be **white and unit-variance in the same basis**. Ours is neither: `condition()` band-passes (samples are correlated) and applies a global scale (`_norm()`), which `‖C(h)‖` inherits and the true SNR does not. Left alone, this would have been a quiet **1.79× = 1/√(320/1024)** bias — the reciprocal root of the band fraction. Small, plausible, and fatal to the x-axis.

**The fix — normalise against the empirical background.** Correlate the conditioned template against a few hundred conditioned *negatives from the file itself* to measure σ_bg, then

$$\rho \;=\; \frac{\lVert C(h)\rVert^2}{\sigma_{bg}}, \qquad \sigma_{bg} \;=\; \operatorname{std}\big(\,C(h)\cdot C(n)\,\big)\ \text{over negatives}$$

Numerator and denominator carry the same arbitrary scale and see the same band, so **both cancel**. It assumes nothing about the noise that we did not measure — and it is how a detection statistic would actually be normalised.

**Result: PASS. Median 0.995×** (5th–95th 0.943–1.055, sd 0.034) on the real 100k.

**And it buys a second check for free.** The same background, correlated against the **stored row** instead of the pure template, gives `(C(h)·x)/σ_bg = ρ + N(0,1)`. Getting **unit variance** back (measured: **−0.018 ± 0.984**) is what proves σ_bg is the right normalisation — and because a misaligned or absent waveform would drive the recovered value to ~0, it *also* proves the row on disk contains the signal its metadata claims, at the sample offset we think. Two of Step 4's silent failure modes, closed by one statistic.

> [!note] Check 4's prediction had the same crack in it
> It predicted a positive's variance as `1 + ρ²/N` — which is the same white-noise assumption wearing a different hat. It now uses the **measured** `‖C(h)‖²` from the statistic above (`1 + ‖C(h)‖²/N`), so the leak test no longer rests on an assumption the pipeline doesn't satisfy. Measured/predicted = **0.976×**.

![[4_dataset_check.png]]
*The real 100k. Top-right is the one that matters. Blue = the signal in the CNN's input, sitting on the ideal line. Orange = the same signal measured off the stored row, scattering about it by exactly 1 — that scatter **is** the noise, and it is why SNR 4 is hard.*

#### ✅ (a) Rebuild is not bit-exact — **FIXED. The file was recording the wrong number.**
Rebuilding a stored row from its metadata alone reproduced it to `max|Δ| = 6.1×10⁻⁶`, not `0`.

**The FFTW-planning theory was wrong.** It was never tested, and it does not survive being tested: forked `Pool` workers and the main process agree **bit-for-bit** on the noise, the waveform, `_norm()`, `condition()`, and the whole segment. Zero, not 10⁻¹⁵. That theory had been sitting in this note as the explanation for a day.

**The actual cause: the metadata was stored as `float32`.** The writer draws parameters in float64 and builds from them — but wrote `m1/m2/snr/merger_pos` to HDF5 as `f4`. So the check rebuilt from a mass of `44.02118682861328` where the writer had used `44.0211878409138`. **That is a different waveform**, and it reproduces the row to ~10⁻⁵ rather than to zero.

The bisect is what named it — the error was **exactly zero on every negative and nonzero on every positive.** Negatives have no waveform parameters, so nothing to round. An FFT or a threading difference could not possibly know the label.

**The fix: store the parameters as `float64`** — at the precision they were *used*. X stays `float32`, because X is *data*; the parameters are the **inputs to a function we intend to rerun**, and 1.6 MB across 100k segments is nothing. `stage1_dataset_check.py` now also asserts the on-disk dtype, so a regression to `f4` says so in one line instead of coming back as an unexplained 10⁻⁵.

**Result: PASS — `max|rebuilt - stored| = 0.000e+00`, exactly.**

> [!tip] The lesson is about the tolerance, not the dtype
> The tempting fix was `assert worst < 1e-4`. It would have passed, it would have looked reasonable, and it would have **thrown away the only check that can tell you your rows and your labels have come apart** — to hide a bug that was real. `10⁻⁵` was not round-off. It was the file honestly reporting that it had recorded a number the writer never used.

### ✅ Step 5 — The 1D CNN — `stage1_model.py` + `stage1_train.py`
Modest architecture — this is the Gabbard-class problem, not ImageNet. **251,361 parameters:**
```
Conv1d → BatchNorm → ReLU → MaxPool(4)     (×3, widening 16 → 32 → 64)
→ Flatten → Dense(128) → Dropout(0.5) → Dense(1)      # raw logit, no sigmoid
```
The one architectural decision that wasn't generic: **the first kernel is 64 samples (31 ms)** — long enough to hold real oscillations (~1 cycle at 30 Hz, ~11 at 350 Hz), because Step 7 wants to read layer 1 as a template bank. Conv layers carry no bias (BatchNorm would silently absorb it), and there is **no per-segment input normalisation** — that's the variance leak; `condition()` already set the scale globally.

- [x] Binary cross-entropy (`BCEWithLogitsLoss`), Adam — **model selection on val AUC, never accuracy.** The test split is untouched until Step 6.
- [x] **Hold out real GW150914** as a sanity check — **it fires.** See below.

**Result: PASS — best val AUC 0.9872 (epoch 7; early-stopped at 13).** ~4 s/epoch on the 3070, **one minute of training total** — "minutes, not hours" was right. Textbook overfit curve after epoch 7 (train loss keeps falling, val loss turns); the checkpoint on disk is the epoch-7 model, at `~/ligo-data/stage1_cnn.pt` — weights are data, not code, and `.gitignore` refuses `*.pt`.

![[5_training.png]]

#### ✅ The GW150914 check — `stage1_gw150914.py`
Real H1 strain from GWOSC (cached to `~/ligo-data/`), merger at GPS **1126259462.423** ([[Stage 0]]'s number). Conditioned with the exact Stage-1 recipe, with the one forced substitution: the whitening PSD is **Welch-measured on an off-source stretch** ([merger−256 s, merger−128 s]) — *never* the segment's own — and the same fixed filter is applied to the event and to 112 disjoint off-source background segments alike. The check: the event must outscore **every** background segment. An absolute p > 0.5 would be meaningless — the background sets the scale.

**Result: PASS — event logit +33.7, above all 112 background segments (max +24.0).** A model trained purely on synthetic injections in synthetic noise fires on the first real black-hole merger ever observed.

> [!warning] The number that matters is the background, and it's a Stage 2 fact measured early
> On simulated test negatives the median logit is ≈ **−3**. On real O1 noise it is **+17.5** — the model finds *real detector noise* dramatically more signal-like than anything it trained on, and it's the event's ~10-logit margin over that inflated floor that saves the check. That one number is Stage 2's thesis measured in advance: swap Gaussian noise for real noise and the false-alarm floor rises before the signals get any louder. "Expect performance to drop" now has a mechanism attached.

![[8_gw150914.png]]

### ✅ Step 6 — Evaluate (NOT on accuracy) — `stage1_eval.py`
- [x] **ROC / AUC** — `sklearn.metrics`. **Test AUC 0.9877** (10k held-out segments).
- [x] **The money plot: detection efficiency (TPR) vs. injected SNR, at fixed false-alarm probability.** This is the Gabbard figure. Accuracy is meaningless here; a model that says "noise" always scores 50% and has learned nothing.
- [x] **Report FAR as false positives per hour of held-out noise** — *not* per year. We don't have the background to support a per-year claim, and faking one is worse than not making one.

**Thresholds are set on the val negatives and *measured* on the test negatives** — setting the threshold on the same segments you then report FAP on makes the number true by construction. They transfer cleanly (measured 1.08e-1 / 9.5e-3 / 6.0e-4 against targets 1e-1 / 1e-2 / 1e-3), and per hour of held-out noise (1.38 h of it): **389 / 34 / 2.2 false alarms per hour**.

**Result: PASS on all four checks.** Efficiency at fixed FAP:

| SNR bin | FAP 1e-1 | FAP 1e-2 | FAP 1e-3 |
|---|---|---|---|
| **4–6** | 0.775 | **0.499** | 0.308 |
| 6–8 | 0.973 | 0.907 | 0.794 |
| 8–10 | 0.997 | 0.992 | 0.969 |
| 10–12 | 1.000 | 1.000 | 0.998 |
| 12–20 | 1.000 | 1.000 | 1.000 |

The curve **degrades at low SNR exactly like it should** — falling off a cliff below SNR 8, half-blind at SNR 4–6 — and sits **below the Neyman–Pearson ceiling everywhere** (the dashed lines: TPR = Φ(ρ − z), the optimal statistic for a *fully known* signal — no search beats it, so sitting above it would have been the leak alarm, per [[#🚩 The tell]]). Same ballpark as the [[GW Signal Classifier - Brainstorm|Gabbard figure]]: their efficiency collapse happens in the same SNR ≲ 8 regime. The honest apples-to-apples matched-filter *baseline* — template bank, unknown time — is [[GW Signal Classifier - Brainstorm|Stage 4]]'s job.

![[6_eval.png]]

### ✅ Step 7 — The payoff figure 🎁 — `stage1_kernels.py`
- [x] **Plot the first-layer conv kernels.**

A 1D convolution **is** a matched filter — it slides a kernel along the signal computing a dot product at each lag, which is exactly what correlating against a template does. So layer 1 is, functionally, **a learned template bank**.

**Result: PASS — 15 of 16 kernels have their spectral peak inside the analysis band** (random init would average ~6/16). The network moved its template bank into the band the signals live in, from labels alone, without ever being told what a black hole is. See [[1D vs 2D - decision explained]] — this figure is the reason we went 1D.

Honesty about what the eye sees: the kernels are **band-limited oscillatory wavelets — chirp *snippets*, not whole chirps.** A 31 ms window physically cannot hold a 30→350 Hz sweep (~2 cycles fit at 60 Hz), so no single kernel can be "a chirp"; the bank covers the band *collectively*, low to high, the way a template bank covers a mass range, and the deeper layers assemble the sweep from these pieces. The figure ends with the same 31 ms of an actual conditioned chirp at merger for comparison — same character as the kernels: band-limited oscillation, rising frequency. The plot shows each kernel's 30–350 Hz component as the dark line: **the input is band-passed, so the out-of-band part of the weights is functionally inert** — data with no power there cannot excite it.

![[7_kernels.png]]

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

**None of these crashed.** All produced confident, plausible-looking numbers — which is the entire argument for writing a check that compares against something you independently know.

### 1. `resize()` threw the merger away
At 10–50 M☉ from 30 Hz, an `IMRPhenomD` waveform is **longer than the 1 s window** — often several seconds. The obvious `hp.resize(n)` keeps the **first** *n* samples: the quiet early inspiral. It silently discards the merger — the loudest part, and the only part the window actually sees.

So `sigma()` was normalising against a waveform that wasn't in/ the data. Recovered SNR came back biased 0.42–0.95×, *worsening with SNR*.

**Fix:** place the waveform by aligning its **amplitude peak** to the target merger time, and let the early inspiral fall off the front of the window. Compute `sigma` on the **placed snippet** — so "SNR 12" means *the SNR of the signal actually present in the segment*, which is the only definition the money plot's x-axis can be held to.

### 2. The crop deleted the answer
`recover_snr()` cropped the edges of the SNR time series — an innocent, standard-looking `[edge:-edge]`. But the template is placed at the **same alignment as the signal**, so the matched filter peaks at **lag zero — index 0**. The crop removed precisely the peak being measured.

It returned a flat **~3.6 for every target SNR**: the pure-noise background, reported with total confidence. Not an error, not a NaN — just a wrong number that looked like a number.

### 3. The check measured a signal that wasn't in the file
Step 4's SNR check conditioned the **unscaled** waveform — it dropped the `snr/σ` factor that `build_segment` applies. It was therefore measuring the SNR of a signal ~1000× louder than the one on disk, and reporting it as a *dataset* failure. **The check was broken, and the dataset was fine.**

Two things saved it. The number was **absurd** (4360×) rather than merely wrong — a subtler slip in the same place would have been believed. And it had a **6× spread** when a pure scale error must be a *constant* ratio: the spread was the fingerprint of `σ(m1, m2)`, and it is what said "you are dividing by something mass-dependent" out loud.

### 4. The dataset recorded a number it had never used
Step 4's writer drew masses in float64, built the waveform from them, and then wrote them to disk as **float32**. Rebuilding a row from its own metadata therefore rebuilt it from a *different waveform* — reproducing it to 10⁻⁵ instead of to 0.

The failure was **exactly zero on every negative and nonzero on every positive**, and that is what named it: negatives carry no waveform parameters, so they have nothing to round. The FFTW theory this note had been carrying could not have known the label.

**It looked exactly like float round-off, and the fix that "worked" was a looser tolerance.** That fix would have deleted the one check that can tell you a writer has come apart from its labels — in order to hide a bug that was really there. Store parameters at the precision you used them.

> [!warning] This is the project's stated failure mode wearing different clothes
> None of these was a leak, but all four were the *same shape* of problem: code that runs clean and lies. The only reason any was caught is that the check compared against a value known independently. **Write the check that can fail** — and then, when it does, **read the shape of the failure before you believe your first theory about it.** The first theory here (band-limited noise breaks the identity) was *true*, and was not the bug.

---

## Done when — ✅ ALL DONE, 2026-07-12

- [x] Detection efficiency vs. SNR curve exists, and it **degrades at low SNR like it should** — 0.50 at SNR 4–6 @ FAP 1e-2, below the known-signal ceiling everywhere
- [x] ROC/AUC computed on a held-out set — **test AUC 0.9877**, thresholds set on val, measured on test
- [x] FAR quoted in false positives **per hour**, honestly — 389 / 34 / 2.2 FA/h on 1.38 h of held-out noise
- [x] The model fires on real **GW150914** — logit +33.7, above all 112 off-source background segments
- [x] First-layer kernels plotted, and **they look like chirps** — band-limited chirp *snippets* (31 ms can't hold a whole sweep); 15/16 peak in band vs ~6/16 for random init
- [x] Result is in the same ballpark as the [[GW Signal Classifier - Brainstorm|Gabbard reference]] — efficiency collapse in the same SNR ≲ 8 regime

→ Then **Stage 2**: swap simulated Gaussian noise for **real O3 noise**. Expect performance to drop. Understanding *why* is the actual project — **and we already hold the first clue: real O1 noise scores median logit +17.5 where simulated negatives score −3.** The false-alarm floor rises before the signals get any louder.
