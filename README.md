# LIGO-ML — Can a CNN match matched filtering?

A staged, from-scratch attempt to train a **1D convolutional neural network** to pick
gravitational-wave signals out of LIGO strain data — and to benchmark it honestly against
**matched filtering**, the method the field actually uses.

The interesting part isn't beating matched filtering. **You can't** — for a known waveform in
Gaussian noise, matched filtering is provably optimal (Neyman–Pearson), so a CNN that beats it
there has a bug, not a breakthrough. The interesting part is what happens when you leave that
regime: **real detector noise isn't Gaussian, and it's full of glitches.** That's where MF's
optimality guarantee evaporates and a learned model has room to earn its keep.

<p align="center">
  <img src="1.%20Stages/Stage%200/outputs/3_qtransform.png" width="85%" alt="Q-transform of GW150914 in H1 and L1 — the chirp"><br>
  <em>GW150914, whitened and Q-transformed. The upward sweep from ~35 to 250 Hz, in both detectors. This is what we're asking the network to find.</em>
</p>

---

## Results so far

| Stage | What it does | Headline result |
|---|---|---|
| **0** ✅ | Pull GW150914, whiten, bandpass, look at it | Saw the chirp. H1/L1 align at **+6.9 ms** and inverted — the case for a 2-channel input. |
| **1** ✅ | 100k injections into **simulated** design noise → 1D CNN | **Test AUC 0.9877** (251k params, ~1 min to train). Fires on real GW150914 above all 112 off-source background segments. |
| **2** ✅ | Same, but into **29.7 h of real O3a noise** | **AUC only −0.017 — but false alarms/hour ×52.** The damage is all in the tail. Retraining re-opens the low-FAP regime (0.008 → 0.957 detection at SNR 8–10). |
| **3** 🔨 | Add **Gravity Spy glitches** as a labelled negative class | In progress — 2,991 glitch specimens selected, strain fetch running. |
| **4** ⬜ | Matched-filter baseline, compared **at equal false-alarm rate** | The actual benchmark. |

### The Stage 2 result, in one paragraph

Train on simulated Gaussian noise, test on real O3a noise, and the AUC barely moves (0.9877 →
0.9703). That number is a liar. At a false-alarm probability of 1e-3, the same model is
**functionally blind** — false alarms per hour jump 52×, because real noise has a tail that
Gaussian noise doesn't (measured: P(|x| > 5σ) is **70× Gaussian** even in glitch-free stretches,
plus ~28 glitches/hour). Retrain the *identical* architecture on real noise and that regime comes
back. The gap is **learnable structure** — which is exactly the crack in matched filtering's
assumptions that this project exists to drive into.

<p align="center">
  <img src="1.%20Stages/Stage%202/outputs/6_eval.png" width="90%" alt="Stage 2 three-arm evaluation">
</p>

---

## Repository layout

This repo is an **Obsidian vault**. Every stage is a long-form note sitting *beside* the code that
produced it — the note is the lab notebook, including the parts that went wrong.

```
1. Stages/
  GW Signal Classifier - Brainstorm.md   ← start here: the plan, and three corrections to it
  Stage 0/   stage0_*.py   + Stage 0.md  + outputs/   ← GW150914, whitening, Q-transform
  Stage 1/   stage1_*.py   + Stage 1.md  + outputs/   ← injections, dataset, CNN, eval
  Stage 2/   stage2_*.py   + Stage 2.md  + outputs/   ← real O3a noise, transfer, retrain
  Stage 3/   stage3_*.py   + Stage 3.md              ← Gravity Spy glitches (in progress)
2. Explained/                                         ← the physics, worked out properly
  Matched filtering explained.md
  1D vs 2D - decision explained.md
  Q-transform explained.md
3. Setup/                                             ← WSL2 + CUDA, and where the docs lie
```

Each stage ships a `requirements-stageN.txt` and a set of `*_check.py` scripts. **The check
scripts are not optional** — see below.

---

## Two decisions worth explaining

**1D time series, not 2D spectrograms.** A 2D magnitude spectrogram throws away **phase** — and
phase is precisely what matched filtering wins with. Benchmarking a phase-blind model against MF
is a confounded comparison that teaches you nothing. Staying in 1D keeps it fair. The bonus:
**a 1D convolution kernel *is* a matched filter**, so the first layer becomes a learned template
bank you can literally plot. ([Full reasoning](2.%20Explained/1D%20vs%202D%20-%20decision%20explained.md).)

<p align="center">
  <img src="1.%20Stages/Stage%201/outputs/7_kernels.png" width="80%" alt="First-layer kernels">
  <br><em>15 of 16 first-layer kernels peak inside the analysis band. The learned template bank is real.</em>
</p>

**Injections carry the dataset, not real events.** There are ~a couple hundred confirmed events in
the entire catalog — that's barely a test set, let alone a training set. So the positives are
synthetic waveform injections into real noise, and the real events (GW150914 et al.) are a held-out
sanity check: if the model can't fire on GW150914, it's broken.

---

## The failure mode that will actually get you

**Data leakage in preprocessing.** It kills more of these projects than any modeling mistake, and
it's insidious *because it looks like success* — 99% AUC and a great mood. Three ways in:

- **PSD estimated from the segment itself** → a segment with a loud signal whitens differently than
  a noise segment, and the network learns the whitening artifact, not the chirp.
- **Per-segment normalization to unit variance** → the injection changes the variance. Same leak.
- **Positives and negatives drawn from different stretches of data** → the noise floor drifted
  between them, and the network learns the drift.

The rules that prevent it — estimate the PSD from a *separate* stretch, apply *identical*
preprocessing to both classes, and build every positive by injecting into a noise segment that
could equally well have been a negative — are enforced by the `*_check.py` scripts in each stage.

> **If your AUC is suspiciously high, assume leakage before assuming genius.**

---

## Running it

Training happens on a desktop PC (RTX 3070) **inside WSL2** — not by preference, but because
`lalsuite` (which PyCBC needs for waveform generation) ships **Linux wheels only**, and WSL2 is the
only Linux VM that gets CUDA passthrough. Stage 0 is the exception: pure `gwpy`/`scipy`, runs fine
on native Windows.

```bash
# Ubuntu 26.04 ships Python 3.14; pycbc has no cp314 wheel yet. The 3.13 pin is load-bearing.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 ~/venvs/ligo && source ~/venvs/ligo/bin/activate
uv pip install -r "1. Stages/Stage 1/requirements-stage1.txt"

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX 3070
```

> ⚠️ **Never install a Linux NVIDIA driver inside WSL2.** It overwrites the Windows driver stubs
> and breaks the passthrough chain — and it's the natural thing to do, because it's what you'd do
> on real Linux. Install the driver on **Windows only**; WSL2 needs no CUDA toolkit at all (torch's
> wheels bundle their own runtime). [The rest of the setup story](3.%20Setup/Setup%20-%20Desktop%20PC.md),
> including the two places the official docs turned out to be wrong.

Data (strain, HDF5 datasets, checkpoints) is **deliberately not in the repo** — it runs to several
GB. Code in git, data on the PC. Every stage re-fetches what it needs from GWOSC.

---

## On honest metrics

False-alarm rate in gravitational-wave astronomy is conventionally quoted in **events per year**,
and claiming ~1/year requires a background set far larger than anything here. So this project
reports **false positives per hour of held-out noise** and is explicit that extrapolating to a
per-year figure isn't something the test set can support. Being upfront about that is worth more
than a confident fake number.

---

## References

- **Gabbard et al. 2018** — [*Matching Matched Filtering with Deep Learning*](https://arxiv.org/abs/1712.06041) — nearly a line-for-line version of this project; the Stage 1 target.
- **George & Huerta 2018** — [*Deep Learning for Real-time Gravitational Wave Detection*](https://arxiv.org/abs/1701.00008)
- [GWOSC](https://gwosc.org) · [gwpy](https://gwpy.github.io) · [PyCBC](https://pycbc.org) · [Gravity Spy](https://gravityspy.org)
