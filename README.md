# LIGO-ML — Can a CNN match matched filtering?

**A completed, from-scratch experiment:** train a **1D convolutional neural network** to pick
gravitational-wave signals out of LIGO strain data, then benchmark it against **matched
filtering** — the method the field actually uses — *at equal false-alarm rate*, on real detector
noise, with a labelled glitch population in the arena.

The interesting part was never beating matched filtering. **You can't** — for a known waveform in
Gaussian noise, matched filtering is provably optimal (Neyman–Pearson), so a CNN that beats it
there has a bug, not a breakthrough. The interesting part is what happens when you leave that
regime: **real detector noise isn't Gaussian, and it's full of glitches.** That's where MF's
optimality guarantee evaporates and a learned model has room to earn its keep.

<p align="center">
  <img src="Stages/Stage%200/outputs/3_qtransform.png" width="85%" alt="Q-transform of GW150914 in H1 and L1 — the chirp"><br>
  <em>GW150914, whitened and Q-transformed. The upward sweep from ~35 to 250 Hz, in both detectors. This is what the network was asked to find.</em>
</p>

## In plain English

*No physics or machine-learning background needed for this section. Everything after it assumes both.*

**What LIGO does.** When two black holes spiral into each other and merge, they shake the shape of
space itself, and the ripple travels outward at the speed of light. LIGO is a pair of enormous
L-shaped rulers, in Louisiana and Washington State, built to feel that ripple as it passes: it
stretches one arm and squeezes the other by a fraction of the width of a proton. The measurement
that comes out is a single wobbling number, recorded thousands of times a second — a sound wave, in
effect. Play it and a black-hole merger goes *whoop*: a rising tone that sweeps up in pitch and cuts
off. Physicists call it a **chirp**. The picture above is one real chirp, from the first detection
ever made, in 2015.

**The problem.** The chirp is far quieter than the noise it's buried in. The detector is a
hair-trigger instrument sitting in a noisy world — trucks, weather, the ocean, its own electronics —
so most of what it records is junk. Finding the chirps in that mess is the whole game.

**How the field does it now.** Because Einstein's equations tell you exactly what shape a merger's
chirp *should* have, you can compute a catalogue of expected chirps ahead of time — one for each
combination of black-hole masses — and then slide each one along the recorded data asking "does this
stretch look like this?" That's **matched filtering**, and there's a theorem saying it is the best
possible method. Nothing can beat it. But the theorem comes with fine print: it only holds if the
noise is *well-behaved* — a uniform, featureless hiss.

**The catch.** Real detector noise is not well-behaved. It hisses, mostly, but it also produces
sudden loud pops and thumps of instrumental origin, called **glitches** — dozens per hour, and some
of them look enough like a chirp to fool the method that was supposed to be unbeatable. The moment
the noise misbehaves, the theorem stops protecting you, and the "best possible method" is only best
in a world that doesn't exist.

**What this project asked.** That gap is where a machine-learning model might have something to
offer: instead of being handed a catalogue of what signals look like, it is shown a large pile of
labelled examples — signal, not-signal, glitch — and left to work out the difference for itself. If
the noise has quirks that nobody wrote an equation for, a model that learns from the actual data has
a chance of picking them up. So: **on real LIGO noise, glitches and all, judged fairly, can a small
neural network keep up with matched filtering — and what does each one cost to run?**

**How it was tested.** In four stages, changing exactly one thing at a time: first in fake, textbook
noise (does the network work at all?), then in 30 hours of real recorded noise (how much does
reality hurt?), then with 1,851 catalogued real glitches added as trick questions (can it learn to
ignore them?), and finally head-to-head against a proper matched-filtering system built to the
standards the field uses. Both sides were tuned to raise the *same number of false alarms* — the
only honest way to compare two detectors, since anything can look sensitive if you let it cry wolf.

**What came back.** Four answers, and the last two are the ones that matter.

1. **In the well-behaved noise the theorem is about, the theorem won — exactly.** The network never
   beat it, anywhere, by any margin. That was the expected outcome, and getting anything else would
   have meant a bug rather than a discovery.
2. **On real noise with glitches, plain matched filtering fell apart** — at strict settings it went
   effectively blind, firing on glitches instead of signals. It survives in practice only because
   physicists bolted on an extra hand-designed test that catches glitches. In other words: the fix
   for the theorem's fine print isn't the theorem. It's a patch.
3. **The network learned its own version of that patch, just from examples** — from about a thousand
   labelled glitches, no equations — and it ends up **better than the hand-designed one at ignoring
   glitches** (it gets fooled about half as often), while the hand-designed one stays **better at
   spotting the faintest real signals**. Honours split. And the network runs **~20× faster per
   candidate and ~3,700× faster in bulk**, needing no catalogue of expected chirps at all.
4. **The preprocessing was doing more of the work than the network was.** Fed the detector's raw
   output with no cleanup, the identical network learns *nothing* — it is a coin flip, and it never
   fires on a real signal at any loudness. The cleanup step isn't housekeeping; it's what makes the
   signal exist as far as the network is concerned. (Discovering that also turned up a bug that had
   been quietly hobbling the network in every previous stage — see below. Fixing it is what moved
   the network ahead of the hand-designed patch in point 3.)

**Why anyone should care.** When a real merger happens, telescopes around the world want to swing
toward it *while the light from the aftermath is still arriving* — a race measured in seconds. That
is the argument for learned detectors: not that they're smarter than the century of physics behind
matched filtering, but that they can be nearly as discerning — better, in one respect — while being
thousands of times cheaper to run. Matched filtering keeps the crown for the careful, offline
analysis. The neural network is making a case for the night shift.

**And the honest footnote.** The single most valuable thing this project produced was not a result;
it was a *check that failed*. The network's glitch-rejection score had been depressed for three
straight stages by a one-line inconsistency nobody had reason to suspect. It was only caught by
rebuilding a baseline that was already believed to be correct — which is the entire argument for
building things you think you don't need.

---

## Abstract

A 251k-parameter 1D CNN was trained on synthetic binary-black-hole injections
(IMRPhenomD, 10–50 M☉, SNR 4–20), first in simulated Gaussian noise at design sensitivity
(Stage 1: test AUC **0.9877**), then in 29.7 h of real O3a H1 strain (Stage 2), then with 1,851
labelled Gravity Spy glitches as hard negatives (Stage 3). Stage 4 built a PyCBC-style
matched-filter baseline — a 62-template stochastic bank at minimal match 0.97, with and without
the χ² signal-consistency veto — and compared all arms at equal false-alarm probability.

**Findings.** (1) Transferring the Gaussian-trained CNN to real noise barely moves its AUC
(−0.017) but multiplies false alarms **52×** at FAP 10⁻³ — the damage lives entirely in the
non-Gaussian tail, which was measured directly: P(|x| > 5σ) is **70× Gaussian** even in
glitch-free stretches, plus ~28 glitches/hour. (2) Retraining the identical architecture on real
noise re-opens the deep-threshold regime (efficiency 0.008 → 0.957 at SNR 8–10 @ FAP 10⁻³) — the
Gaussian/real gap is **learnable structure**, not irreducible randomness. (3) Where the optimality
theorem applies (quasi-Gaussian bulk, FAP 10⁻²), it held to the decimal: no CNN beat the
true-parameter oracle anywhere (measured gap +0.000). (4) *Naked* max-SNR matched filtering
collapsed on real data — functionally blind at FAP 10⁻³, firing on 58% of glitches — and was
rescued only by the χ² veto, an engineered response to the same assumption failure the CNN
learned from data: on glitch rejection the veto-armed MF edges the glitch-trained CNN **0.100 vs
0.166** @ FAP 10⁻² and they effectively tie at 10⁻³ (**0.047 vs 0.050**). (5) The CNN's decisive
win is cost: **~20× lower latency and ~3,700× higher throughput** than the bank, on respectable
hardware both sides.

**Conclusion.** The theorem holds at home; on real noise the engineered veto — not raw optimality
— is what keeps matched filtering alive, and a small CNN can learn an equivalent veto from ~1,000
labelled specimens while running three orders of magnitude faster. That combination — near-parity
robustness at a fraction of the compute — is precisely why the field cares about learned
detectors for low-latency alerts, with matched filtering remaining the offline gold standard.

---

## Results at a glance

| Stage | What it does | Headline result |
|---|---|---|
| **0** ✅ | Pull GW150914, whiten, bandpass, look at it | Saw the chirp. H1/L1 align at **+6.9 ms** and inverted. Pipeline verified on a known answer. |
| **1** ✅ | 100k injections into **simulated** design noise → 1D CNN | **Test AUC 0.9877** (251k params, ~1 min to train). Fires on real GW150914 above all 112 off-source background segments. |
| **2** ✅ | Same, but into **29.7 h of real O3a noise** | AUC −0.017 — but **false alarms/hour ×52** at FAP 10⁻³. Retraining re-opens the regime (0.008 → 0.957 at SNR 8–10). |
| **3** ✅ | Add **Gravity Spy glitches** as labelled hard negatives | Glitch false alarms **halved** @10⁻² (0.331 → 0.166), **cut 3×** @10⁻³ (0.159 → 0.050) — and detection *improved* (AUC 0.9791 → 0.9839). |
| **4** ✅ | **Matched-filter baseline**, compared at equal FAP | Theorem holds on the bulk (no CNN beats the oracle, gap +0.000). Naked MF blind at 10⁻³; the **χ² veto** rescues it and narrowly wins glitch rejection. CNN wins speed: **~20× latency, ~3,700× throughput**. |

---

## 1. Introduction

LIGO detects gravitational waves by correlating detector strain against a bank of relativity-derived
waveform templates — matched filtering. For a known signal in Gaussian noise this is optimal in the
Neyman–Pearson sense; nothing scores better at fixed false-alarm rate. But the optimality proof has
two load-bearing assumptions, and real detectors violate both:

1. **Gaussianity.** Real strain has fat tails and transient instrumental artifacts ("glitches") at
   tens per hour. The tail, not the bulk, sets any deep detection threshold.
2. **Template coverage.** MF can only find what's in the bank.

This project drives at the first crack. The question, stated so it can be lost: *on real,
glitch-ridden noise, at equal false-alarm probability, does a small learned model close any of the
gap that the Gaussian assumption opens in matched filtering — and at what computational price?*

The design discipline throughout: **one variable per stage.** Same waveform family, same SNR range,
same architecture, same conditioning contract — only the noise (Stage 2), then the negatives
(Stage 3), then the detector itself (Stage 4) change.

Each stage is written up in full — including dead ends and bugs — in its own note beside its code:
[Stage 0](Stages/Stage%200/Stage%200.md) · [Stage 1](Stages/Stage%201/Stage%201.md) ·
[Stage 2](Stages/Stage%202/Stage%202.md) · [Stage 3](Stages/Stage%203/Stage%203.md) ·
[Stage 4](Stages/Stage%204/Stage%204.md).

## 2. Methods

### 2.1 Data

- **Strain:** public GWOSC data. Stage 0/1 sanity checks use GW150914-era O1; Stages 2–4 use
  **H1 O3a from GPS 1238166018 (2019-04-01)** — 29.7 h of science-mode strain in 512 s blocks,
  resampled 4096 → 2048 Hz, with **±128 s vetoed around every GWOSC catalog event** so no real
  signal can hide in a "negative".
- **Positives:** synthetic **IMRPhenomD** injections, m1 ≥ m2 ~ U[10, 50] M☉, optimal SNR
  ~ U[4, 20] *measured against the local (per-block) PSD over the 30–350 Hz band of the crop
  actually shown to the model*. Merger placed at U[0.70, 0.95) of the 1 s window. Real events are
  never trained on — the catalog is barely a test set (~a few hundred events), so injections carry
  the dataset and GW150914 serves as a held-out sanity check.
- **Glitches (Stage 3):** Gravity Spy ML classifications for H1 O3a (Zenodo 5649212), confidence
  ≥ 0.9, 13 classes, deduplicated (±0.25 s collapse + ≥ 2 s spacing) to **1,851 distinct
  specimens**, strain fetched by greedy per-class file coverage. Classes `No_Glitch`,
  `None_of_the_Above`, and `Chirp` excluded by construction.

### 2.2 Conditioning — and the leak discipline

Every segment is a 1 s, 2048-sample crop cut from a 4 s buffer: whiten → bandpass 30–350 Hz
(zero-phase FIRs) → crop the filter-corrupted edges. The rule that everything else hangs off:

> **`condition()` never looks at the segment's own content.**

Stage 1 whitens with the *known* design PSD. Stage 2+ whitens block *k* with a **causal median-Welch
PSD from block k−1** (block 0 of every stretch is PSD-source only, never data). Injections happen
*after* PSD estimation. One global scale constant, measured on train-split noise only — never
per-segment normalisation, because an injection changes the variance.

This matters because **preprocessing leakage is how these projects die** — it produces 99% AUC and
looks like success. The three mechanisms (PSD from the segment itself, per-segment normalisation,
positives and negatives from different stretches) are each closed by construction and *proven*
closed by a falsifiable check: a signal-blind operator satisfies
`C(n₁+h) − C(n₁) = C(n₂+h) − C(n₂)`, and the pipeline passes at machine precision
(2.4×10⁻¹⁶ on real O3 noise) where a per-segment whitener disagrees with itself by **36%**.

Stage 3 adds the same rule in new clothes: **every glitch block contributes both classes**
(glitch negatives *and* injections into clean crops of the same block), so "unfamiliar stretch →
say no" is never a winning shortcut, and glitch placement (U[0.10, 0.95)) deliberately overlaps
injection placement so position can't become the label. Stage 3 also saturates all crops at
**±20σ at read time** — see §3.4 for the bug that forced it.

### 2.3 Datasets and splits

| | Stage 1 | Stage 2 | Stage 3 (added) |
|---|---|---|---|
| Rows | 100,000 | 100,273 | 7,312 |
| Balance | 50/50 | 50/50 | 50/50 (1,828 glitch + 1,828 plain neg vs 3,656 inj) |
| Split | 80/10/10 | **time-ordered** 80/10/10 at block level | class-stratified 60/10/30 at block level |
| Held-out test noise | 1.38 h | **1.48 h** (strictly later than train) | 622 glitch specimens on test |

Stage 2's split is time-ordered because real pipelines train on the past and run on the future;
Stage 3's is class-stratified because a per-class census needs every class measurable on test
(the first, time-ordered attempt gave Scattered_Light 0 train / 84 test). Stage 2's frozen split
remains the deployment-honesty guard throughout. Waveform parameters are stored float64 and every
dataset passes a **bit-exact rebuild** check (max|Δ| = 0.0 exactly).

### 2.4 The model

`Stage1CNN`, unchanged from Stage 1 through Stage 4 — architecture search was never the question:

```
Conv1d(k=64) → BN → ReLU → MaxPool(4)   ×3, widening 16 → 32 → 64
→ Flatten → Dense(128) → Dropout(0.5) → Dense(1)          # raw logit
```

251,361 parameters; BCE-with-logits, Adam; model selection on **val AUC, never accuracy**; conv
layers bias-free (BatchNorm absorbs it); no per-segment input normalisation (that's the variance
leak). The 64-sample (31 ms) first kernel is the one non-generic choice: long enough that layer 1
can be read as a **learned template bank** — a 1D convolution *is* a matched filter.

### 2.5 The matched-filter baseline (Stage 4)

- **Bank:** stochastic placement at minimal match 0.97 over **crop-truncated** templates pushed
  through the identical conditioning pipeline as the injections → **62 templates** (from 3,443
  draws). Effectualness on held-out draws: median FF 0.9923, min 0.9696, 100% ≥ 0.965.
- **Statistics:** ρ = max over templates × lags × phase of the normalized correlation, per-block
  causally re-whitened (conditioned noise is unit-variance but *not* white — see §3.5's bug box);
  χ² = 8-bin power chi-squared at the peak; **ρ̃ = newSNR(ρ, χ²ᵣ)** — the reweighted statistic
  real searches actually threshold on.
- **Equal footing:** MF sees the *same conditioned crops* as the CNNs and gets the same per-block
  PSD knowledge the conditioning pipeline has — no more, no less. An **oracle** arm (true-parameter
  template per injection) provides MF's ceiling and the experiment's fraud alarm.
- **Arms:** **C** = Stage-2 CNN (glitch-naive) · **D** = Stage-3 CNN (glitch-trained) ·
  **E** = MF max-ρ · **F** = MF ρ̃ (χ²-vetoed) · **O** = oracle.

### 2.6 Metrics — and what is *not* claimed

All comparisons are made **at equal false-alarm probability** (never equal accuracy): thresholds
are the FAP 10⁻² / 10⁻³ quantiles of each arm's own scores on val negatives, then measured on
test. False-alarm rates are reported **per hour of held-out noise** — the conventional per-year
figure requires a background set orders of magnitude larger than 1.48 h, so it is explicitly not
claimed. The 10⁻³ rows ride on ~10 counts and carry that uncertainty.

Every pipeline step ships a `*_check.py` that compares against something independently known and
**can fail**. This was not ceremony: the checks caught ten-plus real bugs across the project, every
one of which produced plausible numbers rather than a crash (a selection in §5).

## 3. Results

### 3.1 Stage 0 — the pipeline sees the chirp

Whiten → bandpass → crop applied to GW150914 shows the textbook 35 → 250 Hz sweep in both
detectors; the H1/L1 overlay aligns after a **+6.9 ms** shift and an inversion — the light-travel
time between the sites, falling out of public data on its own. The Q-transform above is the
figure. Two real findings for later stages: the correct merger GPS is 1126259462.423 (the commonly
quoted .4 misplaces the chirp by 23 ms), and gwpy's `.whiten()` estimates the PSD from the segment
it whitens — fine for one look at one event, **catastrophic** if carried into a training set (§2.2).

### 3.2 Stage 1 — clean-room baseline in Gaussian noise

**Test AUC 0.9877** on 100k segments; training takes one minute on an RTX 3070. Detection
efficiency at fixed FAP degrades at low SNR exactly as it must, and sits below the Neyman–Pearson
known-signal ceiling everywhere — the tell that would have exposed leakage stayed silent:

| SNR bin | FAP 10⁻¹ | FAP 10⁻² | FAP 10⁻³ |
|---|---|---|---|
| 4–6 | 0.775 | 0.499 | 0.308 |
| 6–8 | 0.973 | 0.907 | 0.794 |
| 8–10 | 0.997 | 0.992 | 0.969 |
| 10–12 | 1.000 | 1.000 | 0.998 |
| 12–20 | 1.000 | 1.000 | 1.000 |

The model, trained purely on synthetic injections in synthetic noise, **fires on real GW150914**
(logit +33.7, above all 112 off-source background segments, whitened with an off-source Welch
PSD). And the payoff figure: **15 of 16 first-layer kernels have their spectral peak inside the
30–350 Hz analysis band** (random init: ~6/16) — the network moved its template bank into the
band the signals live in, from labels alone.

<p align="center">
  <img src="Stages/Stage%201/outputs/7_kernels.png" width="80%" alt="First-layer kernels">
  <br><em>The learned template bank. Band-limited chirp snippets — a 31 ms window can't hold a whole sweep, so the bank covers the band collectively.</em>
</p>

One number in Stage 1 was really a Stage 2 measurement made early: on real O1 noise the model's
median logit is **+17.5**, against **−3** on simulated negatives. Real noise looks more
signal-like than anything the model trained on — the false-alarm floor rises before any signal
gets louder.

### 3.3 Stage 2 — real noise: AUC lies, the tail doesn't

Same model, same signals, only the noise changed. The transfer measurement (Stage 1 checkpoint,
untouched, on real O3a test data):

| target FAP | sim FA/h | real FA/h | ratio |
|---|---|---|---|
| 10⁻¹ | 388.5 | 782.3 | **2.0×** |
| 10⁻² | 34.1 | 209.1 | **6.1×** |
| 10⁻³ | 2.2 | 113.0 | **52×** |

AUC drops only 0.9877 → 0.9703 — **that number is a liar.** The false-alarm inflation grows
*multiplicatively with threshold depth*, the signature of fat tails, and the tails were measured
directly: conditioned real O3 noise has P(|x| > 5σ) = 4.0×10⁻⁵ — **seventy times the Gaussian
rate** — *excluding* glitches, which arrive at ~28 crops/h (worst single crop: std 263, peak
~6,000σ). At FAP 10⁻³ the Gaussian-trained model is functionally blind: 3% efficiency even at
SNR 18–20, because the loudest 0.1% of real noise outscores essentially every signal it knows.

Retraining the *identical* architecture on real noise (arm C) recovers half the AUC (0.9791) and
nearly all of the operating range:

| SNR bin | S1-model @10⁻³ | retrained @10⁻³ |
|---|---|---|
| 4–6 | 0.000 | **0.224** |
| 6–8 | 0.001 | **0.720** |
| 8–10 | 0.008 | **0.957** |
| 18–20 | 0.029 | **0.999** |

<p align="center">
  <img src="Stages/Stage%202/outputs/6_eval.png" width="90%" alt="Stage 2 three-arm evaluation">
</p>

**Retraining doesn't shave the false-alarm rate — it re-opens a regime the glitches had closed.**
The Gaussian/real gap is learnable structure, which is exactly the crack in MF's optimality
assumptions this project set out to drive into. (One honest prediction failed: Stage 1's O1 clue
suggested a wholesale +17.5-logit floor shift; with proper per-block causal PSDs the bulk barely
moved. Right mechanism — false alarms — wrong shape: tail, not shift.)

### 3.4 Stage 3 — glitches by name

1,851 labelled Gravity Spy glitches (13 classes) enter as hard negatives, under the anti-shortcut
rule of §2.2. The retrain (arm D) **improved detection while halving glitch false alarms** —
stage2-test AUC 0.9791 → 0.9839, efficiency flat-or-better in every SNR bin above 6, and on the
622-specimen glitch test set:

| | C (glitch-naive) | **D (glitch-trained)** |
|---|---|---|
| ALL GLITCHES @ FAP 10⁻² | 0.331 | **0.166** |
| ALL GLITCHES @ FAP 10⁻³ | 0.159 | **0.050** |

Per class, three stories: **tamed** (Tomte 0.35 → 0.00, Scattered_Light 0.20 → 0.04, Blip
0.38 → 0.18 — seeing specimens by name works); **halved but standing** (Extremely_Loud 0.80 →
0.455, Koi_Fish 0.82 → 0.35, Repeating_Blips, Scratchy — the loud, chirp-adjacent morphologies
remain the blind spot); **never fooled anyone** (Low_Frequency_Burst 0.000 — peaks below band).

The stage's headline bug is worth the report space: whitened glitch crops peak at up to
**6,094σ** (Stage 2's loudest-ever crop: 18σ). A near-linear CNN scales its logit with input
amplitude, so under BCE the only way to afford confidently-wrong 6,000σ negatives is to shrink
the network's *overall gain* until every logit fits in ±7 — glitch rejection learned, weak-signal
sensitivity destroyed (SNR 6–8 efficiency 0.89 → 0.37, while val AUC still "passed"). The fix —
**saturate crops at ±20σ at read time** (the physical analogue is sensor saturation; a rail at
20σ is still unmistakably a glitch) — is a no-op for every Stage 2 crop and every injection, and
flipped the eval to green.

### 3.5 Stage 4 — the benchmark

<p align="center">
  <img src="Stages/Stage%204/outputs/3_eval.png" width="95%" alt="Stage 4 verdict: efficiency at equal FAP, per-class glitch FA, speed">
</p>

**Front 1 — the theorem's home turf** (efficiency on real-noise injections, FAP 10⁻²): matched
filtering is optimal like it's supposed to be. Arm E and the oracle hit **1.000 in every bin at
SNR ≥ 8**; the CNNs track at 0.986–1.000, always ≤ oracle — the pre-registered fraud alarm ("if
any CNN beats the oracle at SNR ≥ 8, it's a bug, not a triumph") read **+0.000**. At the ragged
SNR 4–6 edge the CNN noses ahead of the *single-template* oracle (0.465 vs 0.376) — trials-factor
physics at the noise floor, not a theorem crack.

**Front 1b — and then FAP 10⁻³ happens.** Naked max-ρ MF is **functionally blind**
(0.000–0.006 efficiency everywhere): its deep threshold is owned by the unlabelled-glitch tail of
real noise — Stage 2's discovery, now reproduced for matched filtering itself. The χ² veto gives
the regime back: arm F sits at 0.85–0.999 and **beats both CNNs at every SNR at 10⁻³**. The
theorem's fine print was always "Gaussian noise"; on real data **the veto is load-bearing.**

**Front 2 — the theorem's graveyard.** All-glitch false-alarm fraction, the project's money table:

| arm | @10⁻² | @10⁻³ |
|---|---|---|
| E — MF max-ρ | 0.582 | 0.116 |
| C — CNN glitch-naive | 0.331 | 0.159 |
| D — CNN glitch-trained | 0.166 | 0.050 |
| **F — MF + χ² veto** | **0.100** | **0.047** |

Naked MF is the strawman the plan warned against — it fires on **100% of Koi_Fish and Tomte** at
10⁻². The honest fight is D vs F: **F wins narrowly** overall (a dead heat at 10⁻³), with real
per-class complementarity — the χ² annihilates Koi_Fish (0.000 @10⁻³ vs D's 0.102: loud-and-
mismatched is its home case) while D handles Extremely_Loud better (0.109 vs 0.145) and they tie
on Whistle. Read it plainly: **from 1,020 labelled specimens the CNN learned, by gradient descent,
approximately what the χ² veto encodes analytically** — it beat naked MF everywhere but did not
surpass the engineered statistic.

**Front 3 — speed.** Same crops, respectable hardware on both sides (3700X × 16 threads for MF,
RTX 3070 for the CNN, single-core CPU numbers reported for honesty):

| | MF (62-template bank) | CNN |
|---|---|---|
| Latency (batch-1) | ~11 ms/crop | **0.47 ms** (1 CPU core!) / 0.81 ms (GPU) |
| Throughput | 42 crops/s (16 threads) | **158,000 crops/s** (GPU, batch 256) |

~20× latency, ~3,700× throughput — and the CNN needs no template bank, no χ² binning, and no
per-block re-conditioning at inference.

## 4. Discussion

The result set reads as a referee's card for matched filtering's two assumptions:

1. **Gaussianity.** Where it holds (the bulk, FAP 10⁻²), the theorem held *to the decimal* —
   nothing beat the oracle, anywhere. Where it fails (deep thresholds, glitch-rich data), naked
   matched filtering didn't degrade gracefully — it **collapsed** — and was rescued by the χ²
   veto, which is itself an engineered patch for the assumption failing. The CNN learned an
   equivalent patch from data alone and finished within noise of the engineered one on glitch
   rejection. That is the project's thesis, demonstrated and bounded: *the gap MF's theorem leaves
   on real noise is learnable structure — but a well-engineered classical statistic already
   captures most of it.*
2. **Template coverage.** Deliberately untested: every injection came from the family the bank
   covers (FF ≥ 0.97). The CNN's generalization edge, if any, lives outside this benchmark.
3. **The practical margin is compute**, and it is enormous. A 251k-parameter model at 0.5 ms/crop
   with near-veto-grade glitch rejection is exactly the trade that makes learned detectors
   attractive for low-latency alerts — with the template bank remaining the offline gold standard.

Against the reference literature: Stage 1 lands in the Gabbard et al. (2018) ballpark (efficiency
collapse in the same SNR ≲ 8 regime, below the known-signal ceiling everywhere), and the project
extends that comparison to where the 2018 papers didn't go — real noise, labelled glitches, a
χ²-armed baseline, and equal-FAP scoring throughout.

## 5. Threats to validity, and what went wrong

Everything below is documented at full length in the stage notes; this is the honest-summary table.

**Scope limits (by design):**
- Single detector (H1). Coincidence — most of real detection's discriminating power — is unused;
  Stage 0's overlay is the argument for a 2-channel follow-on.
- 1.48 h of held-out test noise supports FA/h claims down to ~0.7/h; 10⁻³ rows ride on ~10
  counts. No per-year FAR is claimed.
- Nonspinning IMRPhenomD only, 10–50 M☉; no bank-mismatch test; no 2D/spectrogram arm (deferred).
- SNR labels in non-stationary noise are optimistic in the tail (ρ/target 5th pct 0.45) — a
  property of real noise that afflicts real pipelines' templates identically.

**Bugs that produced plausible numbers instead of crashes** (each caught by a check that could
fail, none by inspection — the project's stated failure mode is a silent wrong answer):

| Bug | Would have corrupted | Caught by |
|---|---|---|
| `resize()` kept the inspiral, discarded the merger | every injected SNR | SNR-recovery check (bias worsening with SNR) |
| Metadata stored float32, used float64 | row/label integrity | bit-exact rebuild refusing to pass at 10⁻⁵ |
| Per-segment whitening (Stage 0 habit) | everything — the 99%-AUC fraud | signal-blindness identity (36% vs 10⁻¹⁶) |
| Gravity Spy CSV duplicate rows | 74% of glitch specimens silently lost | offset-collision audit |
| 6,000σ glitches crushing the CNN's gain under BCE | weak-signal sensitivity | efficiency regression on the frozen Stage 2 split |
| Flat-metric correlation on bandlimited noise | every MF threshold (1.7× miscalibration) | noise-quadrature std check |
| Pooled variance calibration vs one loud glitch | 5% of MF efficiency, uniformly | the oracle tell (CNN "beating" the oracle by +0.054) |

The last one deserves a sentence: the eval's pre-registered theorem-tell — *a CNN beating the
oracle means a bug* — is what caught the final calibration error after the dedicated check had
passed. **The strongest checks are the ones physics writes for you.**

## 6. Reproducibility

This repo is an **Obsidian vault**: every stage folder holds its note (the lab notebook, including
the parts that went wrong), its scripts, its `requirements-stageN.txt`, and its committed output
figures. Code is in git; **data is not** (several GB of strain, HDF5 datasets, and checkpoints
live on the PC at `~/ligo-data/`) — every stage re-fetches what it needs from GWOSC, and every
dataset is deterministic from its seed (bit-exact rebuild enforced by check).

```
Stages/
  GW Signal Classifier - Brainstorm.md    ← the plan, and three corrections to it
  Stage 0/  … Stage 4/                    ← note + code + requirements + outputs, per stage
Setup/                                    ← WSL2 + CUDA, and where the docs lie
```

(`Explained/` — my own physics notes, worked out while building this — is deliberately not in the
repo. The report stands on its own; those are scaffolding.)

Stages 1–4 run **inside WSL2** on a desktop PC (Ryzen 7 3700X / RTX 3070) — not by preference:
`lalsuite` (which PyCBC needs) ships Linux wheels only, and WSL2 is the only Linux VM with CUDA
passthrough. Stage 0 is pure `gwpy`/`scipy` and runs on native Windows (pin
`igwn-segments==2.0.0` — the newer release is source-only and dies without MSVC).

```bash
# Ubuntu 26.04 ships Python 3.14; pycbc has no cp314 wheel yet. The 3.13 pin is load-bearing.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 ~/venvs/ligo && source ~/venvs/ligo/bin/activate
uv pip install -r "Stages/Stage 1/requirements-stage1.txt"

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX 3070
```

> ⚠️ **Never install a Linux NVIDIA driver inside WSL2.** It overwrites the Windows driver stubs
> and breaks the passthrough chain. Driver on **Windows only**; no CUDA toolkit needed (torch's
> wheels bundle their own runtime). [The full setup story](Setup/Setup%20-%20Desktop%20PC.md),
> including the two places the official docs turned out to be wrong.

Run order per stage: the numbered scripts and their paired `*_check.py`, in filename order — each
stage note lists them. **The check scripts are not optional** (see §5).

## References

- **Gabbard et al. 2018** — [*Matching Matched Filtering with Deep Learning*](https://arxiv.org/abs/1712.06041) — the Stage 1 target.
- **George & Huerta 2018** — [*Deep Learning for Real-time Gravitational Wave Detection*](https://arxiv.org/abs/1701.00008)
- **Glanzer et al.** — Gravity Spy ML classifications, [Zenodo 5649212](https://zenodo.org/records/5649212)
- [GWOSC](https://gwosc.org) · [gwpy](https://gwpy.github.io) · [PyCBC](https://pycbc.org) · [Gravity Spy](https://gravityspy.org)
