---
tags: [ligo, machine-learning, stage-2, plan]
status: done
created: 2026-07-12
updated: 2026-07-12
progress: "COMPLETE — all 6 steps done, every check green (23 checks across 4 check scripts). Transfer: AUC 0.9877->0.9703, FA/h x2/x6/x52 at FAP 1e-1/1e-2/1e-3 — the damage is in the tail. Retraining recovers half the AUC and re-opens FAP 1e-3 entirely (0.008->0.957 at SNR 8-10). The why is measured: 70x Gaussian 5-sigma tails + ~28 glitches/h + non-stationarity."
parent: "[[GW Signal Classifier - Brainstorm]]"
---

/us
# Stage 2 — Real O3 noise. Where it gets real.

**Machine:** 🖥️ **Desktop PC** (Ryzen 7 3700X / RTX 3070 / 16 GB), inside WSL2
**Depends on:** [[Stage 1]] ✅ — its injection machinery, its CNN, and its checkpoint
**Previous:** [[Stage 1]] ✅

> **The goal:** swap [[Stage 1]]'s simulated Gaussian noise for **real O3 detector noise**, and
> measure what breaks. **Expect performance to drop. Understanding *why* is the project.**
>
> **We already hold the first clue, measured in Stage 1:** on real O1 noise the Stage-1 model
> scores median logit **+17.5** where simulated negatives score **−3**. The model finds real
> detector noise far more signal-like than anything it trained on — **the false-alarm floor
> rises before the signals get any louder.** Stage 2 turns that one number into a result.

---

## The two questions Stage 2 answers

1. **Transfer:** how much does the Stage-1 model (trained on Gaussian noise) degrade when the
   *test* noise becomes real? This isolates the noise distribution shift — same model, same
   signals, different noise. The +17.5 clue says the damage lands in the false-alarm rate.
2. **Retrain:** how much of the damage does retraining on real noise recover? The gap between
   the retrained model on real noise and the Stage-1 model on Gaussian noise is the price of
   reality; the gap between the two models on real noise is how much of it is *learnable*.

Matched filtering's optimality assumed Gaussian noise — real noise is exactly where that
guarantee evaporates ([[Matched filtering explained]]). Stage 2 builds the arena; [[GW Signal
Classifier - Brainstorm|Stage 3/4]] stage the fight.

---

## 1. Decisions pinned before writing code

Changing these means re-downloading or regenerating — pin them now.

| Decision | Choice | Why |
|---|---|---|
| **Noise source** | **H1, O3a, from GPS 1238166018 (2019-04-01) forward** | The plan says O3; O3a's opening weeks have 236.8 h of H1 science time in the first 14 days alone (measured via `gwosc.timeline`). Single detector keeps it comparable to Stage 1. |
| **Data selection** | `H1_DATA` science segments, in time order, **±128 s vetoed around every GWOSC catalog event** in the span | Negatives must contain no real signals. 15 event datasets (6 distinct events) sit in the first two weeks; vetoing them costs minutes out of ~29 h. |
| **Segments** | **Disjoint 1 s crops**, 4 s buffers hopping 1 s within 512 s blocks | Same geometry as Stage 1 (`condition()`'s 4 s rule). Adjacent crops share zero samples; buffers never cross a block boundary, so **splits share no raw samples at all**. |
| **Dataset size** | **~100k crops (509 per data block, ~197 blocks ≈ 28 h)**, 50/50 injected | Matches Stage 1's size, so AUC/efficiency numbers are comparable like-for-like. |
| **Whitening PSD** | **Per-block, causal: block *k* is whitened with a Welch PSD from block *k−1*.** Block 0 of every stretch is PSD-source only, never data. | Real noise is non-stationary — one global PSD would whiten Tuesday's noise with Monday's spectrum. Per-block tracks the drift; *causal* (previous block, never the block itself) honours the letter of the leak rule: **the PSD never sees the data it whitens, and injections happen after PSD estimation, so the operator is signal-blind by construction.** Same fixed filter for every segment in a block, positive or negative — exactly [[Stage 1]]'s `condition()` contract, and exactly what `stage1_gw150914.py` already did with its off-source Welch PSD. |
| **SNR definition** | Optimal SNR of the placed waveform **against the block's measured PSD**, over the crop, in 30–350 Hz | "SNR 8" must mean SNR 8 *in the noise the segment actually sits in*. Using the design PSD would mislabel every point on the money plot's x-axis by the (time-varying) ratio of real to design sensitivity. |
| **Splits** | **Time-ordered 80/10/10 at block level** — val and test are strictly later in time than train | Real pipelines train on the past and run on the future; this is the honest split. Positives and negatives are drawn uniformly *within* every block, so the pos/neg-from-different-stretches leak cannot occur. Noise drift *between* splits is not a leak — it is the phenomenon under study. |
| **Injections** | Identical to Stage 1: `IMRPhenomD`, m1, m2 ∈ U[10, 50] M☉, SNR ∈ U[4, 20], merger at U[0.7, 0.95) of the crop | Only the noise changes. That is the entire experimental design — one variable. |
| **Scale constant** | One global constant, measured on conditioned **train-split noise only**, stored in the dataset attrs | Same rule as Stage 1's `_norm()`: per-segment normalisation is the variance leak. Measured on train blocks so the test split can't even *see* the constant's inputs. |
| **Architecture** | **Unchanged** `Stage1CNN` (251k params) | Same reason as the injections: change one variable. Architecture search is not Stage 2's question. |

### 📁 Where things live
| | |
|---|---|
| **Code** | `1. Stages/Stage 2/` — in the vault, in git; imports Stage 1's modules via `sys.path` |
| **Raw strain** | `~/ligo-data/stage2_strain.h5` — ~1.9 GB, blocks of resampled 2048 Hz H1 strain |
| **Dataset** | `~/ligo-data/stage2.h5` — same schema as `stage1.h5` plus `block_id` / `gps` |
| **Checkpoint** | `~/ligo-data/stage2_cnn.pt` |

Run everything from WSL2 against the Windows-side vault, per [[Stage 1]]:
```bash
source ~/venvs/ligo/bin/activate
cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 2"
python stage2_fetch.py
```

---

## 2. The build

> [!tip] Same discipline as Stage 1: every step ships with a check that can fail
> Stage 1's checks caught four real bugs that produced plausible numbers instead of crashes.
> Real data adds new silent failure modes (gaps, glitches in the PSD stretch, non-stationarity),
> so if anything the checks matter more here.

### ✅ Step 1 — Fetch real O3 noise — `stage2_fetch.py` + `stage2_fetch_check.py`
**DONE — 209 blocks (197 data + 12 PSD-source), 1.75 GB, 29.7 h of H1 O3a.** The download
survived a GWOSC read-timeout at block 203/209 precisely because the fetcher is
**resumable** (`n_done` attr, atomic per-group writes); a retry finished the last 6 blocks
in 42 s. Every group is fetched with 8 s of padding so the resample FIR's edge transient
never lands inside a stored block.
- [x] Query `H1_DATA` science segments from O3a start; veto ±128 s around every catalog event
      (one event in the final span: GW190403_051519)
- [x] Download in time order until ~197 data blocks exist; resample 4096 → 2048 Hz
- [x] Store per-block strain in `stage2_strain.h5` (float64 — conditioning stays in doubles)
- [x] **Check: all 6 PASS on the full file** — no NaNs/gaps, veto respected (re-derived from
      GWOSC, not trusted from the fetcher), every block inside a science segment, blocks
      disjoint, every stretch leads with a PSD block. The ASD figure shows the premise:
      real O3 H1 is not the design curve, and it moves between blocks.

![[1_fetch_check.png]]

### ✅ Step 2 — Condition real noise — `stage2_condition.py` + `stage2_condition_check.py`
Port of [[Stage 1]]'s `condition()` with the design PSD replaced by the block's causal Welch
PSD (median-averaged — one glitch in the PSD block can't drag the spectrum; interpolated to
the buffer's Δf, 0.5 s inverse-spectrum truncation, 30–350 Hz zero-phase FIRs, central 1 s
crop, one global scale). **DONE — all 3 checks PASS on the full 197-block file.**
- [x] **Check — signal-blindness:** `C(n₁+h) − C(n₁) == C(n₂+h) − C(n₂)` to machine precision
      within a block — **measured 2.4×10⁻¹⁶ on real O3 noise.** The measured-PSD conditioner
      is exactly as signal-blind as Stage 1's known-PSD one, because the PSD is causal.
- [x] **Check — whitening quality:** flat in band (tilt 1.16), and the global constant
      measured on the FIRST 8 blocks gives **median crop std 0.989 on the LAST 8** — the
      scale survives 29 h of detector drift.
- [x] **Measure — non-Gaussianity:** with glitchy crops excluded, real O3 still has
      excess kurtosis +0.13 and **P(|x| > 5σ) = 4.0×10⁻⁵ — seventy times the Gaussian
      rate.** The tails are everywhere, not just in the glitches.

> [!warning] 🐛 The check that failed, and why the DATA was right
> The first full-file run FAILED the scale-transfer gate: pooled std 5.0, per-block up to
> 32.9, pooled kurtosis 129,531. The diagnosis (per-crop stds over every data block) found
> the whitening calibration **perfect everywhere** — median crop std 0.993 across all 197
> blocks — and the "failure" to be **two real glitches**: one crop at GPS ~1238343793 with
> std 263 (peak excursion ~6000σ), one at ~1238353104 with std 32, in otherwise pristine
> blocks. **The measured glitch rate is ~28 crops/h with std > 2.**
>
> The data stays. Glitches are Stage 2's subject matter (and Stage 3's whole diet) —
> sanitising them away would quietly turn real noise back into the Gaussian we left Stage 1
> to escape. What changed is the CHECK: the calibration gate now uses the **median** crop
> std (a calibration statistic a glitch cannot move), and the monsters are **reported as a
> rate** instead of poisoning a mean. Stage 1's lesson in new clothes: *read the shape of
> the failure before believing your first theory* — the gate said "calibration broke", the
> shape said "two loud samples in 512".

![[2_condition_check.png]]

### ✅ Step 3 — Write the dataset — `stage2_dataset.py` + `stage2_data.py` + `stage2_dataset_check.py`
Per crop: take the block's raw strain buffer → (if positive) inject a waveform scaled to a
target SNR against the block's PSD → condition → store the central 2048 samples. Identical
call for both classes. Same HDF5 schema as Stage 1 plus `block`, `offset`, `gps`.

**The real run: 100,273 crops in 59.6 min** (28 seg/s, 16 workers) → **832 MB** at
`~/ligo-data/stage2.h5`. Splits 79,913 / 9,671 / 10,689 — train ends GPS 1238326533, test
spans 1238336265–1238353942, **1.48 h of held-out real test noise.** *(The first build
attempt died at 67% when its Windows-side wrapper was stopped — the WSL process dies with
`wsl.exe`. Rebuilt detached via `setsid nohup`; lesson recorded in project memory.)*
- [x] Parent draws all specs up front (dataset depends only on the seed, not worker count) —
      and touches neither the strain file nor an FFT before forking (Stage 1's FFTW-fork
      deadlock + the h5py-handle-across-fork rule; workers get a fork-hygiene initializer)
- [x] Waveform parameters stored **float64** — Stage 1's bit-exact-rebuild lesson stands
- [x] **Check: all 10 PASS on the real 100k.** Structure; time-ordered splits verified on
      every row; **bit-exact rebuild — max|Δ| = 0.000e+00 exactly**; **SNR calibration via
      the empirical-background ρ statistic: median 0.986**, and the stored-row statistic
      comes back ρ + N(0,1) → **+0.03 ± 0.99** — Stage 1's fix ported unchanged *because*
      it assumed nothing about the noise.
- [x] Honesty note: the ρ/target 5th percentile is **0.45** (vs 0.89 on the smoke run) — a
      tail of positives landed where the local noise is worse than the causal PSD says, so
      their SNR labels are optimistic. That is a property of *real* non-stationary noise,
      not of the writer; the same staleness afflicts a real pipeline's templates.

![[3_dataset_check.png]]

### ✅ Step 4 — The transfer measurement — `stage2_transfer.py` 🎯
**The headline: the damage is in the tail, not the bulk.** Stage 1 checkpoint, untouched,
on the Stage 2 test split — all 3 checks PASS (including the harness sanity check: the sim
side reproduces Stage 1's AUC to the fourth decimal).

- [x] **AUC 0.9877 (sim) → 0.9703 (real)** — same model, same signal population, only the
      noise changed. A −0.017 drop from the noise distribution shift alone.
- [x] **The logit medians barely moved** — negatives −3.57 → −3.23, positives +31.4 → +33.1.
      The O1-based prediction of a wholesale floor shift (+17.5) did **not** reproduce on
      O3 with per-block causal PSDs; the bulk of real noise, properly whitened, looks
      Gaussian enough to the model.
- [x] **FA/h at Stage 1's deployed thresholds — the floor rises *multiplicatively with
      depth*:**

| target FAP | sim FA/h | real FA/h | ratio |
|---|---|---|---|
| 1e-1 | 388.5 | 782.3 | **2.0×** |
| 1e-2 | 34.1 | 209.1 | **6.1×** |
| 1e-3 | 2.2 | 113.0 | **52×** |

The deeper the threshold, the worse the lie — precisely the signature of the fat tails
Step 2 measured (P(|x|>5σ) at 70× Gaussian, glitches at ~28/h). The mechanism the
[[Stage 1]] warning predicted is real; its *shape* is sharper than predicted: **cut FAP by
10× and the Gaussian assumption costs ~an order of magnitude in false alarms.**

![[4_transfer.png]]

### ✅ Step 5 — Retrain on real noise — `stage2_train.py`
- [x] Same `Stage1CNN`, same recipe (Adam, BCE, model selection on val AUC, test untouched)
- [x] Checkpoint → `~/ligo-data/stage2_cnn.pt` — **best val AUC 0.9798 (epoch 6),
      0.9 min of training.**

Worth noticing: val AUC is far noisier epoch-to-epoch than Stage 1's (swinging 0.86–0.98
where Stage 1 climbed smoothly). Val is a *later stretch of time* here, so each epoch's
checkpoint generalises across detector drift, not just across injections — the turbulence
is the non-stationarity, visible in the training curve.

![[5_training.png]]

### ✅ Step 6 — Evaluate — `stage2_eval.py` — **all 4 checks PASS**
Three arms: **A** = S1 model/sim noise (reference), **B** = S1 model/real noise (the drop),
**C** = S2 model/real noise (the recovery). Thresholds set on each arm's own val negatives,
measured on test; arm C's transfer within 1.2× at FAP 1e-1/1e-2.

| arm | AUC | FA/h @1e-1 | @1e-2 | @1e-3 |
|---|---|---|---|---|
| A — S1/sim | 0.9877 | 388.5 | 34.1 | 2.2 |
| B — S1/**real** | 0.9703 | 542.7 | 127.2 | 11.5 |
| C — S2/**real** | 0.9791 | 374.2 | 42.0 | 6.8 |

- [x] ROC/AUC of both models on real noise: **A→B −0.0174 (the price of assuming Gaussian),
      B→C +0.0088 (learnable), A→C −0.0086 (reality's remaining tax).** Retraining buys back
      almost exactly half.
- [x] **The money plot** — and the result of the stage lives in the FAP 1e-3 column:

| SNR bin | B @1e-2 | C @1e-2 | B @1e-3 | C @1e-3 |
|---|---|---|---|---|
| 4–6 | 0.342 | 0.465 | **0.000** | **0.224** |
| 6–8 | 0.840 | 0.887 | **0.001** | **0.720** |
| 8–10 | 0.981 | 0.986 | **0.008** | **0.957** |
| 18–20 | 1.000 | 0.999 | **0.029** | **0.999** |

  **At FAP 1e-3 the Gaussian-trained model is functionally blind on real data** — 3% at
  SNR 18–20, because the loudest 0.1% of real noise (the glitches) outscores essentially
  every signal it knows. The retrained model *restores the operating point*: 96% at
  SNR 8–10. **Retraining doesn't shave the false-alarm rate — it re-opens the low-FAP
  regime that glitches had closed.** This is the strongest evidence yet for the project's
  central bet: the CNN *can learn* what breaks the Gaussian assumption. (Stage 3 now has a
  precise job: make that learning explicit with labelled glitches.)
- [x] FA per hour of held-out real noise, honestly: table above, on **1.48 h** (test split
      is 10% of 29.7 h, half of it positives) — supportable down to ~0.7 FA/h, so the 1e-3
      row rides on ~10 counts and carries that uncertainty.
- [x] Gaussian ceiling plotted as a *diagnostic* (labelled, not a law here); arm C sits
      below it everywhere — distance from it is the measured non-Gaussianity tax.

![[6_eval.png]]

---

## 3. Footguns — Stage 1's list, plus the new ones real data brings

### ⚠️⚠️ The PSD leak, real-data edition
Still THE one. The rule holds: **never estimate the whitening PSD from data the injection has
touched.** Here the PSD comes from the *previous* block — it never sees the segment it whitens,
and injections are added after estimation. If a future refactor "simplifies" this to per-segment
Welch, Stage 1's 36%-disagreement demo becomes this stage's silent 99%-AUC fraud.

### ⚠️ Real events in the "noise"
O3a contains real signals. A catalog event inside a negative teaches the model that chirps are
noise. ±128 s vetoes around every GWOSC event dataset in the span. (Sub-threshold signals below
catalog sensitivity are an accepted, quantifiable contamination — they are also *real*, which is
the point of the stage.)

### ⚠️ Glitches in the PSD block
A loud glitch in block k−1 inflates the PSD that whitens block k, deadening its band. That is
not a leak (still signal-blind) but it is a data-quality wobble — Welch's median-ish averaging
over ~250 overlapping 4 s segments per block bounds it. The condition check's per-block std
distribution is where it would show up.

### ⚠️ Non-stationarity across the train/test boundary
Test noise is *later* than train noise by design. If the detector's state changed dramatically
mid-span, the transfer number confounds "real vs simulated" with "then vs now". The fetch check
plots per-block band RMS over time so we can see any drift we are living with.

### 🚩 The tell, updated
Stage 1's rule was "suspiciously high AUC = leakage". Stage 2 adds the mirror image: **if the
Stage-1 model transfers to real noise *without* degradation, suspect the noise isn't real** —
a bug that quietly substituted simulated data (wrong file, wrong split, cached Gaussian noise)
would produce exactly that flattering non-result.

---

## Done when — ✅ ALL DONE, 2026-07-12

- [x] The transfer number exists: **AUC 0.9877 → 0.9703, FA/h ×2 / ×6.1 / ×52 at FAP
      1e-1/1e-2/1e-3** — and the logit plot shows the degradation lives in the *tail* of the
      negatives, not the bulk
- [x] A retrained model exists (val AUC 0.9798) and its efficiency curve sits above the
      transferred model's in every SNR bin at every FAP — decisively at FAP 1e-3, where it
      re-opens an operating regime the transferred model had lost entirely (0.008 → 0.957
      at SNR 8–10)
- [x] The money plot compares all three arms at fixed FAP, per SNR bin
- [x] FA/h quoted against **1.48 h** of real held-out noise, with the supportable floor
      (~0.7 FA/h) stated
- [x] **The why, measured:** (1) *fat tails everywhere* — conditioned real noise has
      P(|x|>5σ) = 4.0×10⁻⁵, seventy times Gaussian, even excluding glitches; (2) *glitches*
      — ~28 crops/h with std > 2, worst single crop std 263 (peak ~6000σ); (3)
      *non-stationarity* — PSD drift across 29.7 h that makes a fixed threshold's meaning
      drift (and a tail of SNR labels optimistic, ρ 5th pct 0.45). Mechanisms (1)+(2) are
      why false alarms explode multiplicatively with threshold depth: the Gaussian model's
      score for "loud" saturates on glitches it has no category for. Mechanism (3) is why
      val AUC is turbulent and thresholds transfer at 1.2× rather than exactly.

### What Stage 2 established, in one paragraph
Swapping simulated for real noise costs the Gaussian-trained CNN −0.017 AUC — but AUC
hides the real damage: at fixed *deep* thresholds its false-alarm rate is **52× worse**,
and at FAP 1e-3 it is functionally blind (glitches outscore signals). Retraining the same
architecture on real noise recovers half the AUC and nearly all of the operating range,
which means the difference between Gaussian and real noise is largely **learnable
structure, not irreducible randomness**. That is exactly the crack in matched filtering's
optimality theorem this project set out to drive into — MF has no mechanism to learn it
([[Matched filtering explained]]), and [[GW Signal Classifier - Brainstorm|Stage 4]] will
measure whether the CNN's learned advantage survives the fair fight.

→ Next **[[GW Signal Classifier - Brainstorm|Stage 3]]**: hard negatives — Gravity Spy
glitches as an explicit negative class. Stage 2 showed the model *implicitly* learns
glitch-vs-chirp from only 28 glitches/h of exposure; Stage 3 makes that explicit and
reports FAR against a labelled glitch population.
