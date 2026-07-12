---
tags: [ligo, machine-learning, stage-0, log]
status: complete
created: 2026-07-11
updated: 2026-07-12
parent: "[[GW Signal Classifier - Brainstorm]]"
---

# Stage 0 — See the Chirp

**Status:** ✅ Complete, 2026-07-11
**Machine:** first run on the laptop (i7-1255U, Iris Xe) — **native Windows, no WSL, no GPU**
**Code:** `4. Code/stage0/stage0_gw150914.py` — **in the repo**, runs on either machine
**Outputs:** `4. Code/stage0/outputs/` — the three plots below, committed

> [!note]- Provenance of the committed plots
> Stage 0 was originally run on the laptop, and its code lived outside the vault at
> `C:\Users\tmloc\ligo-ml\`, so **none of it was ever in git.** On 2026-07-12 the script was
> brought into the repo and re-run **on the PC** (native Windows, no WSL — Stage 0 needs neither).
> The plots below are from that PC run. Same data, same pipeline, same conclusions: GWOSC returned
> the identical 131,072 samples @ 4096 Hz, and the chirp is where it was.
> The `igwn-segments==2.0.0` pin was confirmed a second time — gwpy 4.0.1 installs clean on
> native Windows with it, and cannot without it.

---

## Why this stage exists

Stage 0 contains **no machine learning at all.** That's deliberate.

The whole project rests on a preprocessing pipeline — fetch strain → whiten → bandpass. If any link in that chain is subtly wrong, every downstream result is garbage, and we'd be debugging a *CNN* when the actual bug is in a *filter*. So before building anything, prove the pipeline works on a case where **we already know the right answer**: GW150914, the most scrutinized 0.2 seconds in the history of the field.

The success criterion is deliberately not a number. It's: **can I see the chirp with my own eyes?**

Secondary goal: find out what the toolchain does on Windows *before* committing to an architecture around it. (It bit us — see [[#What went wrong]].)

---

## What GW150914 is

The first direct detection of a gravitational wave — 14 Sept 2015. Two black holes (~36 and ~29 solar masses) spiralling into each other ~1.3 billion light years away, merging into one.

As they spiral in, they orbit **faster and faster**, radiating gravitational waves at rising frequency — sweeping from ~35 Hz up to ~250 Hz in about 0.2 s, then cutting off abruptly at merger. Rising frequency + rising amplitude = **the chirp**. That shape is the thing the whole project is trying to detect.

The strain it produced on Earth is ~1e-21 — a fractional length change of about 1/10,000th the width of a proton across LIGO's 4 km arms.

---

## What I did

### 1. Fetched real strain data from GWOSC
```python
from gwpy.timeseries import TimeSeries
GPS_MERGER = 1126259462.423
h1 = TimeSeries.fetch_open_data("H1", GPS_MERGER - 16, GPS_MERGER + 16, cache=True)
l1 = TimeSeries.fetch_open_data("L1", GPS_MERGER - 16, GPS_MERGER + 16, cache=True)
```
32 s of data around the event, from both detectors — **H1** (Hanford, WA) and **L1** (Livingston, LA). Came back as 131,072 samples @ **4096 Hz**.

This is genuinely the real, public, archival LIGO data. Not a simulation.

### 2. Whitened it
```python
white = strain.whiten(4, 2)     # fftlength=4 s, overlap=2 s
```
**Why:** raw LIGO strain is *completely dominated* by low-frequency seismic noise and narrow instrumental lines. The signal is ~1e-21 and utterly invisible underneath — plot the raw strain and you see a wall of noise, nothing else.

Whitening divides the data by the noise amplitude spectral density, flattening the noise floor so **every frequency contributes equally**. The noise becomes roughly white; anything that isn't noise-shaped pops out. This is not cosmetic — without it there is nothing to see.

### 3. Bandpassed 30–350 Hz
```python
band = white.bandpass(30, 350)
```
**Why:** GW150914's power lives in that band. Below 30 Hz is seismic; above 350 Hz is shot noise and irrelevant to this event. Cutting both raises contrast.

### 4. Cropped the edges
```python
return band.crop(START + 1, END - 1)
```
**Why:** filtering and PSD estimation **corrupt the edges** of a segment — filter transients ring in from the boundaries. The outer second on each side is an artifact, not data. Throwing it away is mandatory, not tidiness.

> This is why we fetched **32 s** to look at **0.3 s**. The margin is fuel for the PSD estimate and the crop.

### 5. Plotted three things
| Output | What it shows |
|---|---|
| `1_chirp_per_detector.png` | Whitened + bandpassed strain, H1 and L1 separately |
| `2_overlay.png` | H1 and L1 laid on top of each other — *the money plot* |
| `3_qtransform.png` | Q-transform time-frequency spectrogram — the chirp as a visible arc |

---

## Results

### Both detectors, separately

![[1_chirp_per_detector.png]]

Amplitude *and* frequency both climb into t=0, then cut off and ring down. That's the chirp in the raw time series — and it's only visible at all because of the whitening.

### The Q-transform is the convincing one

![[3_qtransform.png]]

Both detectors show the **textbook upward sweep** — energy climbing from ~35 Hz to ~250 Hz and terminating abruptly at t=0. That abrupt cutoff *is* the merger. Nothing in instrumental noise looks like that.

> Look at where the two arcs *end*: L1's terminates a few ms **before** H1's. That offset is the light-travel time between the detectors, falling out of the data on its own — and it's the same 6.9 ms we have to put in by hand to make the overlay below work.

> 📖 **Full explainer: [[Q-transform explained]]** — what it is, why a plain spectrogram can't do this, how to read the plot, and the 1D-vs-2D design fork it exposes.

*(Two gotchas: `q_transform` runs on the **raw** strain — it whitens internally, so passing whitened data would whiten twice. And the colorbar plots **√**(normalized energy) for contrast, so displayed 0–7 = real energy 0–49.)*

### The overlay is the physics

![[2_overlay.png]]

To lay L1 on top of H1, two corrections are needed:
1. **Shift L1 by +6.9 ms.** The wave hit Livingston *before* Hanford — it swept across the Earth at *c*, and the detectors are ~3000 km apart.
2. **Invert L1.** The two detectors' arms are oriented differently, so L1's response has opposite sign.

Apply both, and **two independent instruments 3000 km apart trace the same squiggle.**

That agreement is *why GW150914 was believable*. A glitch in one detector is just a glitch. The same waveform in two, separated by exactly the light-travel time, is a signal.

> **→ This is the argument for giving the CNN H1+L1 as a 2-channel input** rather than a single detector. Coincidence is most of the discriminating power, and a single-channel model throws it away.

---

## What went wrong

Three things. All worth having hit now rather than in Stage 3.

### 1. `pip install gwpy` fails on native Windows ⚠️
I'd claimed it would "just work." It doesn't.

gwpy depends on `igwn-segments`. Pip resolves to the newest, **2.1.1**, which ships **source-only** — it tries to compile a C extension and dies:
```
error: Microsoft Visual C++ 14.0 or greater is required.
```
**Fix:** `igwn-segments` **2.0.0** has a prebuilt `cp313-win_amd64` wheel, and gwpy only requires `>=2.0.0`. Pin it. No 6 GB MSVC toolchain needed.
```bash
pip install "igwn-segments==2.0.0" gwpy matplotlib
```
*(Not an issue in WSL2 — Linux gets wheels for current versions.)*

### 2. Merger GPS was 23 ms off
Used `1126259462.4`; the chirp showed up centred at **+0.023 s** instead of 0. The correct value is **`1126259462.423`**. Harmless here, but it would quietly misalign every training segment if carried into Stage 1.

### 3. NaN speckles in the Q-transform
gwpy's normalized energy can go **slightly negative**; I took `sqrt` of it and got NaNs, rendered as white speckle across the plot. Fix: `np.sqrt(np.clip(q.value.T, 0, None))`.

---

## ⚠️ The lesson that matters for Stage 1

**gwpy's `.whiten()` estimates the PSD from the very segment it is whitening.**

That is **fine here.** We're looking at one event in 32 s of data; a 0.2 s signal barely perturbs a PSD estimated over that whole stretch.

It is **catastrophic for the training set.** If a segment containing a loud injection is whitened using *its own* PSD, then the whitening operation itself **carries a fingerprint of the signal** — and the CNN will cheerfully learn *that fingerprint* instead of the chirp. The result is a model with 99% AUC that has learned nothing about gravitational waves and will fail on anything real.

This is the single most likely way this project silently fails.

**Rules for Stage 1:**
- [ ] Estimate the PSD from a **separate stretch of data**, not the segment being whitened.
- [ ] Apply **identical** preprocessing to positives and negatives.
- [ ] Build each positive by injecting into a noise segment **that could equally well have been a negative**.
- [ ] **If AUC is suspiciously high, assume leakage before assuming genius.**

---

## Reproduce
From the vault root, on **either machine** — native Windows is fine, Stage 0 needs no WSL:
```powershell
cd "4. Code\stage0"
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-stage0.txt
.\.venv\Scripts\python stage0_gw150914.py
```
Takes ~1 min (most of it downloading from GWOSC; `cache=True` makes reruns instant). Writes the
three PNGs into `outputs/`, overwriting them.

The `requirements-stage0.txt` pin of **`igwn-segments==2.0.0`** is the whole reason this installs
at all on Windows — see [[#1. `pip install gwpy` fails on native Windows ⚠️]]. Don't "helpfully"
unpin it.

*(`.venv/` is gitignored. The plots are **not** — they're the deliverable, and they're small.)*

---

## Next
→ **Stage 1** ([[GW Signal Classifier - Brainstorm#Staged plan]]): injections + 1D CNN on simulated Gaussian noise.
Needs `lalsuite` → **Linux only** → requires the WSL2 setup on the **PC**.
