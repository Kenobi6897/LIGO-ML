---
tags: [project, ligo, machine-learning, brainstorm]
status: in-progress
created: 2026-07-11
updated: 2026-07-12
---

# LIGO Gravitational Wave Signal Classifier

## Objective
Train a **1D CNN** to classify LIGO strain data as *gravitational wave signal* vs. *noise/glitch*, and compare it against **matched filtering**, the method physicists actually use.

## Why this project
LIGO researchers genuinely use ML for noise/signal discrimination in *detector characterization*. This is a scaled-down version of live research, not an invented exercise. It also has a well-trodden reference path — see [[#Reference papers]] — so we can check our work against known results.

---

## Three corrections to the original brief

### 1. Injections carry the dataset, not real events
There are only a couple hundred confirmed GW events in the entire catalog. That is not a training set — it's barely a test set.

**So:** essentially all positives are **synthetic injections** (known waveforms added to real noise at varying SNR). The handful of real events (GW150914 et al.) become a small held-out **sanity check**: if the model can't fire on GW150914, it's broken. This is what the reference papers do.

### 2. We will not beat matched filtering — and that's a theorem, not a failure
For a **known waveform in Gaussian noise**, matched filtering is *provably optimal* (Neyman–Pearson). There is no room above it. **If the CNN beats MF in that regime, we have a bug** — almost certainly leakage (see below).

> 📖 **Full explainer: [[Matched filtering explained]]** — how it works, why whitening is half of it, why phase is the whole ballgame, and — the part that matters — **the two assumptions it rests on, which are exactly the cracks Stages 2–4 drive into.**

So the benchmark needs a sharper question. The real ones:
1. **Speed** — can the CNN approach the optimum in one forward pass, vs. convolving against a whole template bank? (This is why LIGO people actually care: latency.)
2. **Robustness** — does the CNN degrade *more gracefully on real, glitchy, non-Gaussian noise*, where MF's optimality guarantee evaporates? ← **this is our project**
3. **Generalization** — MF can only find what's in the template bank.

### 3. `lalsuite` is Linux-only → we need WSL2
`lalsuite` (which PyCBC needs for waveform generation) publishes **Linux wheels only, no Windows wheels**. Since injections are now load-bearing, this is a hard blocker, not an inconvenience.

---

## Environment (decided)

**Target machine: the desktop PC — RTX 3070 (8 GB), Ryzen 7 3700X (8c/16t), 16 GB 3200 RAM.**
*(Not the laptop: i7-1255U / Iris Xe / no CUDA. Laptop is fine for editing + Stage 0 plotting, but training happens on the PC.)*

**Decision: everything in WSL2 on the PC. No Colab.** WSL2 gets **CUDA passthrough**, so we get `lalsuite` (Linux-only) *and* GPU training in one environment. That's the whole ballgame — it's the one thing the laptop couldn't do.

**Why WSL2 and not a VirtualBox/VMware VM:** WSL2 *is* a VM (lightweight Hyper-V, real Linux kernel) — and crucially, it's the only one that gets the GPU. A full VM would need real PCIe passthrough (painful, typically wants a second GPU), plus static RAM carve-out and fiddly file sharing. Not close.

### ⚠️ The CUDA-on-WSL2 footgun
**NEVER install a Linux NVIDIA driver inside WSL2.** It overwrites the Windows driver stubs and breaks the passthrough chain — and it's the natural thing to do, because it's what you'd do on real Linux.

- Install the NVIDIA driver on **Windows only**.
- Inside WSL2, install the **CUDA toolkit only**, using NVIDIA's `wsl-ubuntu` packages (built to skip the driver).
- `/usr/lib/wsl/lib/` is auto-mounted with `libcuda.so.1` / `nvidia-smi` stubs that forward through `/dev/dxg` to the real Windows driver.
- Ref: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### The real constraint is RAM, not the GPU
The **3070 is wild overkill** for this model — a batch of 256 × 2048 floats is ~2 MB. We will not come close to 8 GB VRAM, and training runs take *minutes*. **Don't let the GPU shape the design.**

**16 GB system RAM is the actual squeeze**, because WSL2 by default claims up to half the host RAM (8 GB), while the dataset is what wants memory:
- 100k segs × 1 s × 2048 Hz × float32 ≈ **820 MB**
- 200k segs @ 4096 Hz ≈ **3.3 GB**

Two rules, applied from the start (retrofitting is annoying):
- [ ] Cap WSL2 memory in `%UserProfile%\.wslconfig` → `memory=10GB`
- [ ] **Store the dataset as HDF5, load lazily in `Dataset.__getitem__`** — do NOT `np.load` the whole array into RAM.

### The Ryzen does more work than you'd think
**Matched filtering is CPU-bound and embarrassingly parallel** across the template bank — as are injection generation and whitening. Stage 4 and the Stage 1 dataset build will lean on those 16 threads far harder than anything leans on the 3070.

Nice side effect for the project's honesty: we're benchmarking **GPU-CNN vs. CPU-matched-filter**, which is exactly the comparison the field cares about — and with respectable hardware on *both* sides, the speed result means something instead of being an artifact of a starved baseline.

### Which machine for which step?
**Only training actually wants the 3070 — and even that's a "wants," not a "needs."** Everything else is CPU work. The split is about *iteration speed*, not capability.

| Step | Needs | Laptop? |
|---|---|---|
| **Stage 0** — pull GW150914, whiten, bandpass, see the chirp | `gwpy` + scipy, CPU | ✅ **Yes — natively on Windows, no WSL at all** |
| **Stage 1a** — injections, dataset build | `lalsuite` → Linux, CPU | ✅ Yes in WSL2, just slower |
| **Stage 1b–3** — train the CNN | GPU strongly preferred | ⚠️ Works on CPU, ~10–20× slower |
| **Stage 4** — matched-filter baseline | CPU, parallel over template bank | ✅ Yes, but the 3700X's 16 threads eat this |

> **Stage 0 needs no `lalsuite`** — whitening/bandpassing GW150914 is pure gwpy/scipy, and gwpy runs on native Windows. No WSL, no reboot, no PC. ✅ **Confirmed working on the laptop.**

#### ⚠️ Windows gotcha: pin `igwn-segments==2.0.0`
`pip install gwpy` **fails on native Windows.** gwpy pulls `igwn-segments`, pip resolves to **2.1.1** — which ships **source-only**, tries to compile a C extension, and dies on `Microsoft Visual C++ 14.0 or greater is required`.

**2.0.0 has a prebuilt `cp313-win_amd64` wheel**, and gwpy only requires `>=2.0.0`. So just pin it — no 6 GB of MSVC Build Tools needed:
```bash
pip install "igwn-segments==2.0.0" gwpy matplotlib
```
*(Not an issue in WSL2 — Linux gets wheels for the current version.)*

**Do NOT split the pipeline across both machines.** You *could* build the dataset on the laptop and train on the PC — but you'd shuttle a 1–3 GB HDF5 between boxes every time you change a preprocessing decision, and you **will** change preprocessing decisions constantly, because that's where the leakage bugs live. The sync tax lands on exactly the loop you iterate hardest.

**Clean division:**
- **Laptop** → Stage 0, plus all code editing/reading.
- **PC** → Stages 1–4, one WSL2 environment. Data and model live together.
- Code in **git** (edit anywhere); **data on the PC only**.

*(Caveat: both machines have 16 GB, so the RAM squeeze + lazy-HDF5 rule applies either way — not a reason to prefer one box.)*

### Setup (PC, Stages 1–4) — ✅ **DONE 2026-07-12**
Full record, with the two places this sketch turned out to be wrong: **[[Setup - Desktop PC]]**.

```bash
wsl --install                      # platform only — then ALSO: wsl --install -d Ubuntu
# NO CUDA toolkit needed: torch's wheels bundle their own CUDA runtime.
# (The toolkit only gives you nvcc, for compiling custom kernels. We compile none.)

# Ubuntu 26.04 ships Python 3.14, and pycbc has NO 3.14 wheel → pin the venv to 3.13:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 ~/venvs/ligo && source ~/venvs/ligo/bin/activate
uv pip install gwpy pycbc numpy scipy matplotlib h5py scikit-learn tqdm torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Result: **Ubuntu 26.04**, CPython **3.13.14**, torch **2.13.0+cu130** → `True NVIDIA GeForce RTX 3070`,
lal **7.7.1**, pycbc **2.11.0**. *(No `--index-url` pin on torch — plain PyPI torch on Linux is already
the CUDA build, and it picks a build that suits driver 596.49 better than cu124 would.)*

(Heavyweight alternative: the official **IGWN conda distribution**, which ships gwpy/pycbc/lalsuite/bilby pre-integrated.)

---

## The failure mode that will actually get us

**Data leakage in preprocessing.** This kills more of these projects than any modeling mistake, and it's sneaky *because it looks like success* — 99% AUC and a great mood.

The mechanisms:
- **PSD estimated from the segment itself** → a segment with a loud signal gets whitened differently than a noise segment, and the network learns *the whitening artifact*, not the chirp.
- **Per-segment normalization to unit variance** → the injection changes the variance. Same leak.
- **Positives and negatives drawn from different stretches of data** → the detector's noise floor drifted between them, and the network learns *the drift*.

Rules that prevent it:
- Estimate the PSD from a **separate** stretch of data.
- Apply **identical** preprocessing to positives and negatives.
- Build each positive by injecting into a noise segment **that could equally well have been a negative**.

> **If AUC is suspiciously high, assume leakage before assuming genius.**

---

## Staged plan

Each stage produces a result we can look at.

- [x] **Stage 0 — DONE (2026-07-11).** ✅ Pulled GW150914, whitened, bandpassed, **saw the chirp.** First run on the *laptop*, native Windows, no WSL. **Full write-up: [[Stage 0]].** Code + plots are **in the repo** at `1. Stages/Stage 0/`, beside the note (added 2026-07-12; re-run on the PC to regenerate them, same result).
    - Q-transform shows the textbook upward sweep (~35 → 250 Hz) cutting off at merger, in **both** detectors.
    - H1/L1 overlay lines up after shifting L1 by **+6.9 ms** and **inverting** it. That inter-detector coincidence is the argument for a **2-channel CNN input** later.
    - Correct merger GPS is **1126259462.423** (not `.4` — that puts the chirp 23 ms off-center).
- [x] **Stage 1 — DONE (2026-07-12).** ✅ 100k injections in simulated design noise → 1D CNN (251k params, 1 min to train) → **test AUC 0.9877**, efficiency vs SNR degrades at low SNR like it must (0.50 at SNR 4–6 @ FAP 1e-2) and sits below the Neyman–Pearson ceiling everywhere — Gabbard ballpark. Fires on real **GW150914** (above all 112 off-source background segments). 15/16 first-layer kernels peak in the analysis band: the learned template bank is real. **Full write-up: [[Stage 1]].** Code + plots in `1. Stages/Stage 1/`, checkpoint + dataset on the PC.
    - **Stage 2's first clue, already measured:** real O1 noise scores median logit **+17.5** where simulated negatives score **−3** — the model finds real detector noise far more signal-like than anything it trained on. The false-alarm floor rises before the signals get louder.
- [x] **Stage 2 — DONE (2026-07-12).** ✅ 100k injections into **29.7 h of real O3a H1 noise** (per-block causal Welch PSDs, time-ordered splits, every leak check green). Performance dropped exactly as predicted — and the *shape* of the drop is the result: AUC only −0.017 (0.9877 → 0.9703), but **FA/h ×52 at FAP 1e-3** — the damage lives in the tail (measured: P(|x|>5σ) = 70× Gaussian even glitch-free, plus ~28 glitches/h, worst crop std 263). **At FAP 1e-3 the Gaussian-trained model is functionally blind on real data; retraining the same CNN re-opens that regime (0.008 → 0.957 at SNR 8–10)** and recovers half the AUC. The gap is *learnable structure* — the crack in MF's optimality theorem, demonstrated. **Full write-up: [[Stage 2]].** Code + plots in `1. Stages/Stage 2/`, strain/dataset/checkpoint on the PC.
    - The O1-based "+17.5 logit floor" clue did **not** reproduce on O3 with proper per-block PSDs — the bulk barely moves; it's the top 0.1% of noise that does the damage. The prediction was right about the mechanism's location (false alarms), wrong about its shape (tail, not shift).
- [x] **Stage 3 — DONE (2026-07-13).** ✅ 1,851 deduplicated Gravity Spy glitches (13 classes) as labelled negatives, both classes from every block (the anti-shortcut rule). Glitch FA halved @1e-2 (0.331 → 0.166), cut 3× @1e-3 (0.159 → 0.050), detection *improved* (AUC 0.9791 → 0.9839). One deep bug: 6,000σ glitch crops made BCE crush the CNN's gain — fixed by ±20σ saturation at read time. Still fooling it: Extremely_Loud, Repeating_Blips, Koi_Fish, Scratchy. **Full write-up: [[Stage 3]].**
- [x] **Stage 4 — DONE (2026-07-13).** ✅ The benchmark: 62-template bank (MM 0.97, crop-truncated, per-block re-whitened, every calibration measured), ρ and χ²-reweighted ρ̃, thresholds at equal FAP on real-noise val negatives. **The theorem held at home** (MF = oracle = 1.000 at SNR ≥ 8 @1e-2, no CNN beat the oracle, gap +0.000); **naked MF died on real noise** (blind at FAP 1e-3 — the glitch tail owns its threshold; 58% glitch FA @1e-2); **the χ² veto rescued it** (F: 0.85–0.999 efficiency at 1e-3, beats both CNNs; glitch FA 0.100/0.047, edging the glitch-trained CNN's 0.166/0.050). The CNN learned the veto instead of engineering it, and wins on speed: ~20× latency, ~3,700× throughput. **Full write-up: [[Stage 4]].**

### Metrics caveat
FAR in GW is conventionally quoted in **events per year**, and claiming ~1/year requires an enormous background set we won't have. **Report false positives per hour of held-out noise**, and be explicit that extrapolating to per-year isn't something our test set can support. Being honest about that is more impressive than a fake number.

---

## Design decisions

### ✅ Decided
- **1D time series, not 2D spectrograms.** Full rationale: **[[1D vs 2D - decision explained]]**.
  Short version: a 2D magnitude image **throws away phase**, which is exactly what matched filtering wins with — so a 2D-vs-MF benchmark is **confounded** and teaches us nothing. 1D keeps the comparison fair. Bonus: **a 1D conv kernel *is* a matched filter**, so the first layer becomes a **learned template bank** we can plot.
- **2D is not discarded** — it becomes a **Stage 3 comparison arm** (glitch rejection is inherently a 2D *shape* problem). The [[Stage 0]] Q-transform code already exists, so this costs almost nothing.

### Still open
- Segment length / sample rate (e.g. 1 s @ 4096 Hz vs. downsampled to 2048).
- Single detector (H1) or **H1 + L1 as 2-channel input**? Inter-detector coincidence is a big part of real detection — and [[Stage 0]]'s overlay plot is the evidence for it.
- Injection SNR range — how weak do we go? The interesting regime is where MF *starts to struggle*.
- Class balance and decision threshold.

## Explicitly out of scope (for now)
- **2D as the *primary* architecture.** Not out of scope entirely — see [[1D vs 2D - decision explained]]; it returns as a Stage 3 comparison arm. But the "YOLO-style object detection" framing in the original brief refers to work on **spectrograms**, not 1D strain — **don't let it pull the 1D CNN design around.**
- **A Gravity Spy multi-class glitch classifier.** A fine *separate* project (easier: ships as labeled images, no injections, no lalsuite, runs natively on Windows) — but it has no matched-filtering benchmark, so it isn't this one.
- **Parameter estimation as regression** (predict chirp mass) — nice follow-on: no class imbalance, no threshold-setting, easy to eyeball.

---

## Reference papers
- **Gabbard et al. 2018 — "Matching Matched Filtering with Deep Learning"** — https://arxiv.org/abs/1712.06041 — nearly a line-for-line version of this project. Our Stage 1 target.
- **George & Huerta 2018 — "Deep Learning for Real-time GW Detection"** — https://arxiv.org/abs/1701.00008

## Links
- GWOSC — https://gwosc.org
- gwpy — https://gwpy.github.io
- PyCBC — https://pycbc.org
- Gravity Spy — https://gravityspy.org
- IGWN conda — https://computing.docs.ligo.org/conda/
