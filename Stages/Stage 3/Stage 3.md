---
tags: [ligo, machine-learning, stage-3, plan]
status: done
created: 2026-07-12
updated: 2026-07-13
progress: "COMPLETE — all 6 steps done, every check green (7 fetch + 13 dataset + eval 3/3). The money table: overall glitch FA 0.331->0.166 @1e-2, 0.159->0.050 @1e-3, with detection IMPROVED (stage2-test AUC 0.9791->0.9839). One real bug on the way: 6,000-sigma glitch crops made BCE crush the CNN's gain (logits pinned at +-7, SNR 6-8 efficiency 0.89->0.37) — fixed by saturating crops at +-20 sigma at read time. Still fooling it: Extremely_Loud, Repeating_Blips, Koi_Fish, Scratchy."
parent: "[[GW Signal Classifier - Brainstorm]]"
---

# Stage 3 — Hard negatives. Glitches, by name.

**Machine:** 🖥️ **Desktop PC**, inside WSL2
**Depends on:** [[Stage 2]] ✅ — its strain/conditioning machinery, its dataset, its checkpoint
**Previous:** [[Stage 2]] ✅

> **The goal:** add **Gravity Spy glitches as an explicit, labelled negative class** and report
> the false-alarm rate *per glitch class*. [[Stage 2]] showed the CNN learns glitch-vs-chirp
> *implicitly* from ~28 unlabelled glitches/h — enough to re-open FAP 1e-3. Stage 3 feeds it
> thousands of labelled specimens and measures which glitch morphologies still fool it.
>
> This is the stage where the CNN earns (or doesn't) its keep against matched filtering:
> glitches are exactly what break MF's Gaussian assumptions, and [[GW Signal Classifier -
> Brainstorm|Stage 4]] will run that comparison. Stage 3 builds the glitch benchmark.

## The data source (probed 2026-07-12)

**Gravity Spy ML classifications, Zenodo record 5649212** (Glanzer et al.), file `H1_O3a.csv`
(90 MB, cached at `~/ligo-data/gravityspy_H1_O3a.csv`): 80,763 H1 O3a glitches with GPS
`event_time`/`peak_time`, `duration`, `peak_frequency`, `snr`, `ml_label` (22 classes),
`ml_confidence`. At confidence ≥ 0.9: 65,457. The big in-band classes: Extremely_Loud (10k,
94% in-band), Scattered_Light (9.4k), Koi_Fish (7.1k, 99% in-band, median SNR 121), Blip (4k,
95% in-band), Whistle (6.3k), Low_Frequency_Burst (18k, peaks below band but with in-band
power), plus Tomte / Repeating_Blips / Blip_Low_Frequency / Scratchy in the hundreds.

**Only 603 confident glitches fall inside [[Stage 2]]'s 29.7 h strain span** — Stage 3 fetches
its own strain, *targeted*: glitches cluster in time, so we greedily select the GWOSC 4096 s
files that cover the most not-yet-capped glitches and fetch only those.

## 1. Decisions pinned before writing code

| Decision | Choice | Why |
|---|---|---|
| **Glitch selection** | conf ≥ 0.9; classes with ≥ 150 specimens; **cap 1,500/class**; drop `No_Glitch` (ordinary noise — Stage 2 has 50k of it), `None_of_the_Above` (unlabelled), `Chirp` (chirp-shaped by definition — a *label* error waiting to happen) | Confident, balanced, and honest about what "glitch" means. |
| **Event veto** | ±128 s around every GWOSC catalog event in O3a, same as Stage 2 | An Extremely_Loud "glitch" that is actually GW190521 would be a signal labelled negative. |
| **Strain fetch** | Reuse Stage 2's block machinery verbatim: 512 s blocks, causal per-block PSD, block 0 of each stretch PSD-only. Stretches = merged glitch neighbourhoods `[g−524 s, g+8 s]` ∩ science segments, chosen by **greedy file-coverage under a download budget (~150 files ≈ 3 h)** | Same conditioning contract as Stage 2 — `stage2_condition` is pointed at the new file unchanged, so every leak property already proven carries over. |
| **The anti-shortcut rule** | **Every Stage 3 block contributes BOTH classes**: glitch-centred crops as negatives, PLUS injections into clean crops of the *same block* as positives (plain-noise crops fill to ~50/50) | Glitch-only additions would let the model learn "this stretch's texture → negative" — the pos/neg-from-different-stretches leak wearing new clothes. Both classes from every block, identically conditioned, kills it. |
| **Glitch placement** | Glitch `peak_time` at **U[0.10, 0.95) of the crop** — deliberately OVERLAPPING the injection range U[0.70, 0.95) | If glitches only ever appeared where mergers never do, position would become the label. |
| **Injections** | Identical to Stages 1–2: `IMRPhenomD`, U[10,50] M☉, SNR U[4,20] vs the block's measured PSD | Still one variable per stage: the new thing is labelled glitches, nothing else. |
| **Splits** | ~~Time-ordered~~ → **CLASS-STRATIFIED 60/10/30 at block level** (revised 2026-07-12; Stage 2's splits stay frozen and time-ordered); training mixes Stage2-train + Stage3-train | The original time-ordered plan met reality: glitch classes cluster in time (storms), and the first build gave Scattered_Light **0 train / 84 test** and Whistle 139/0/1. A per-class census needs stratification. What time-ordering protected is preserved where it matters — block-level sample disjointness holds, and Stage 2's time-ordered split remains the deployment-honesty guard. Test-heavy (30%) because the money table lives on per-class test statistics. |
| **Architecture** | **Unchanged `Stage1CNN`**, trained from scratch on the mix | The question is what the *data* buys, not the architecture. |
| **2D Q-transform arm** | **Deferred** to a Stage 3b if wanted — the brainstorm's 2D comparison is real but the 1D benchmark must exist first | Scope control; [[1D vs 2D - decision explained]] stands. |

### 📁 Where things live
| | |
|---|---|
| **Code** | `1. Stages/Stage 3/` — imports Stage 1 + Stage 2 modules via `sys.path` |
| **Glitch table** | `~/ligo-data/gravityspy_H1_O3a.csv` (Zenodo 5649212) |
| **Selection** | `~/ligo-data/stage3_selection.h5` — chosen glitches + merged spans |
| **Raw strain** | `~/ligo-data/stage3_strain.h5` — same schema as Stage 2's |
| **Dataset** | `~/ligo-data/stage3.h5` — Stage 2 schema + `glitch_label`, `glitch_snr`, `glitch_conf` |
| **Checkpoint** | `~/ligo-data/stage3_cnn.pt` |

## 2. The build

### ✅ Step 1 — Select glitches + plan the download — `stage3_select.py` + check
**DONE (revised) — 1,851 DISTINCT glitches, 13 classes, 114 disjoint spans, 99.6 h /
~161 GWOSC files. All 8 checks PASS.** Storms capped near 400 (Scattered_Light,
Low_Frequency_Burst), chirp-like classes at 134–184 (Blip / Koi_Fish / Extremely_Loud),
long tail at ~30–50.
- [x] Filter (conf ≥ 0.9, class rules, event veto at ±(128+512) s, segment geometry)
- [x] **Deduplicate** — see the bug box below; "400 per class" must mean 400 *events*
- [x] **Per-class round-robin greedy** under a file budget; spans on a **per-segment block
      grid**; write the selection table
- [x] **Check:** every constraint re-verified from the CSV and GWOSC independently
      (the selector caches GWOSC metadata locally after rate-limiting struck; the check
      deliberately keeps querying live)

> [!warning] 🐛 The Gravity Spy CSV lists one glitch many times
> The first dataset build silently lost **74% of its glitch rows** (2,991 → 780). The
> diagnosis: 1,777 "offset collisions" — glitches wanting the same crop. They wanted the
> same crop because they were **the same glitch**: the CSV carries ~many rows per physical
> trigger at near-identical peak_times, and per-file top-confidence selection loaded up on
> repeats. Fix, two stages: collapse true duplicates (±0.25 s, keep max confidence), then
> enforce ≥ 2 s spacing between kept specimens — chosen over a chain-cluster collapse,
> which had quietly turned a 100 s storm of distinct arches into one specimen. The
> post-dedup census also revealed distinct chirp-like glitches are *sparse* (~3/file), so
> the budgets grew to buy them: 80 span-hours, 130 files.

> [!warning] 🐛 Two selection bugs the check (and its own output) caught before any download
> **1. Greedy by raw coverage chased storms.** The first objective — take the file covering
> the most glitches — spent the whole budget on **three Scattered_Light storm files**
> (1,240 of one class, 10 Blips, 10 Koi_Fish). Balance had to *be* the objective: round-robin
> over under-target classes, neediest first, opportunistic take of whatever else the file
> holds.
> **2. Block-completion rounding cascaded 56 h into 214 h.** Spans were extended to complete
> their last 512 s block; extensions overran the next span; merged spans extended further —
> a feedback loop the disjointness check caught (and a fixed-point remerge only made
> quietly worse). Fix: **anchor blocks to a per-segment grid** (`seg_start + 8 + k·512`) and
> fetch exactly the blocks glitches need (k and its causal k−1). No rounding, no feedback —
> and the accounting came back to 56.18 h on its own.

### ✅ Step 2 — Fetch the strain — `stage3_fetch.py` + check
**DONE 2026-07-13 — 5.5 GB, all 7 checks PASS.** The overnight fetch survived; all
1,851 glitches landed in role-1 blocks with contiguous causal predecessors.
- [x] Reuse Stage 2's fetch machinery over the merged spans → `stage3_strain.h5`
- [x] **Check:** Stage 2's fetch check, pointed at the new file (no NaNs, science-mode,
      event-vetoed, disjoint, PSD-block-led stretches) — plus: every selected glitch's
      buffer AND its causal PSD block actually exist in the file

### ✅ Step 3 — Write the dataset — `stage3_dataset.py` + `stage3_data.py` + check
**DONE 2026-07-13 — 7,312 rows exactly 50/50 (1,828 glitch + 1,828 plain negatives,
3,656 injections; 1,828 of 1,851 selected glitches survived placement), splits
4,080/744/2,488, all 13 checks PASS** (bit-exact rebuild, ρ/target = 1.009,
88% of glitches in the primary position window, 30% overlapping the injection range).
- [x] Rows per block: glitch-centred negatives + clean-crop injections + plain-noise
      negatives, ~50/50 pos/neg, identical `condition_real()` call for all
- [x] **Check:** structure; bit-exact rebuild; SNR calibration (ρ statistic, same-block
      backgrounds); **glitch-presence check** — a glitch-centred crop must actually contain
      excess power (std above the block's clean-crop distribution) at the recorded position
- [x] Splits verified: block-disjoint, every class measurable on test

> [!warning] 🐛 Stratifying by block count starved seven tail classes on test
> The first build's splitter gave each class ~30% of its *blocks* to test — but blocks
> are lumpy (a storm block holds hundreds of one class) and a 40-specimen class needs
> min(20, all-it-has) test ROWS to be measurable, which is ~50% of it, not 30%. It also
> ignored rows a class inherited from blocks already assigned by rarer classes. Fix:
> targets are row counts (test first: `max(round(0.3·total), min(20, total))`), counting
> inherited rows. Measurability outranks the fractions; the fractions were never the
> point — the money table is.

### ✅ Step 4 — Train on the mix — `stage3_train.py`
**DONE 2026-07-13 — best val AUC 0.9843 (epoch 2), after the stage's headline bug:**
- [x] Stage2-train ∪ Stage3-train, same recipe, model selection on the combined val
- [x] Checkpoint → `~/ligo-data/stage3_cnn.pt`

> [!warning] 🐛🐛 6,000-sigma inputs made BCE strangle the network — the saturation bug
> First training "worked" (val AUC 0.9329, PASS) and failed the eval twice over: stage2-test
> AUC 0.9791 → 0.9337, SNR 6-8 efficiency 0.89 → 0.37, AND overall glitch FA *worse*
> (0.363 vs 0.304 @1e-2). The diagnosis, measured: whitened glitch crops peak at up to
> **6,094σ** (Extremely_Loud; Stage 2's loudest crop ever: 18σ). A near-linear CNN scales
> its logit with input amplitude — the glitch-naive arm C emits logit **+361** on a 361σ
> glitch. Under BCE the only way the optimizer can afford confidently-wrong 6,000σ
> negatives is to shrink the network's overall gain until *every* logit fits in ±7
> (measured: arm D's stage2-val |logit| p99 = 6.7 vs C's 31.4) — glitch rejection learned,
> weak-signal sensitivity destroyed. **Fix: crops saturate at ±20σ in
> `Stage3Dataset.__getitem__`** — read-time, so the h5 stays raw and the bit-exact rebuild
> check still holds; identical for training and eval, so both arms see the same inputs; a
> no-op for every Stage 2 crop and every injection (physical analogue: sensor saturation —
> a rail at 20σ is still unmistakably a glitch). Retrained: val AUC 0.9843, and the eval
> flipped to 3/3 PASS.

### ✅ Step 5 — The glitch benchmark — `stage3_eval.py` 🎯
**DONE 2026-07-13 — all 3 checks PASS.**
- [x] Arms: **C** = Stage-2 model (glitch-naive), **D** = Stage-3 model (glitch-trained),
      both on: the Stage 2 test split (efficiency must NOT degrade) and the Stage 3 glitch
      test set
- [x] **The money table: per-class glitch false-alarm fraction at FAP 1e-2 / 1e-3
      thresholds** (set on each arm's own val negatives) — which morphologies fool a
      chirp detector, and which stop fooling it once it has seen them by name
- [x] Overall FAR on glitch-rich data, honestly denominated

**The result.** Detection did not just survive glitch training — it improved:
stage2-test AUC **C 0.9791 → D 0.9839**, efficiency flat-or-better in every SNR bin
above 6 (the 4-6 bin dips 0.465 → 0.428, and still fails as it must). On the glitch
test set (622 specimens, 13 classes):

| | C @1e-2 | **D @1e-2** | C @1e-3 | **D @1e-3** |
|---|---|---|---|---|
| **ALL GLITCHES** | 0.331 | **0.166** | 0.159 | **0.050** |

Per class, three stories:
- **Tamed** — Tomte 0.35 → **0.00**, Scattered_Light 0.20 → **0.04**, Fast_Scattering
  0.20 → **0.05**, Blip_Low_Frequency 0.35 → **0.15**, Blip 0.38 → **0.18** (0.05 @1e-3).
  Seeing specimens by name works.
- **Halved but standing** — Extremely_Loud 0.80 → **0.455**, Koi_Fish 0.82 → **0.35**,
  Scratchy 0.65 → **0.35**, Repeating_Blips 0.70 → **0.45**, Whistle 0.33 → **0.26**.
  The loud chirp-adjacent morphologies (Koi_Fish is a teardrop chirp cousin;
  Repeating_Blips *contains* blips; Extremely_Loud saturates everything) remain the
  detector's blind spot — exactly the classes Stage 4's matched filter will also face.
- **Never fooled anyone** — Low_Frequency_Burst 0.000 both arms (peaks below band),
  Low_Frequency_Lines, Air_Compressor ≤ 0.1 throughout.

No 🚩 tell: no class sits at perfect rejection with full efficiency, low SNR still
fails, and the Stage-2 split stayed frozen.

### ✅ Step 6 — Notes + commit

## 3. Footguns

### ⚠️⚠️ The stretch-texture shortcut (this stage's own leak)
Adding glitches only as negatives from new time spans teaches "unfamiliar stretch → say no".
The anti-shortcut rule above (both classes from every block) is not optional, and the eval
guards it from the other side: Stage-2-test efficiency must not drop.

### ⚠️ A "glitch" that is a signal
`Chirp` class excluded; catalog events vetoed; Extremely_Loud specimens near event times die
with the veto. Sub-threshold astrophysics in the glitch set remains possible and is accepted
(same honesty note as Stage 2's).

### ⚠️ Position as label
Glitch placement overlaps injection placement by construction. Check it stays that way.

### ⚠️ The PSD block may itself be glitchy
Glitches cluster, so the 512 s before a glitch often holds more glitches. Median Welch
bounds the damage (Stage 2's choice, made for exactly this); the dataset check's SNR
calibration measures what remains.

### 🚩 The tell, Stage 3 edition
If arm D rejects every class at ~100% while keeping full efficiency, be suspicious: some
classes (Blip especially) are morphologically close to short chirps, and a perfect score
more likely means the glitch test set leaked into training than that the problem died.

## Done when

- [x] Per-class glitch FA table exists for both models, at FAP 1e-2 and 1e-3
      (`stage3_eval.py` output + `outputs/6_eval.png`)
- [x] The Stage-3 model beats the Stage-2 model on glitch rejection **without losing
      efficiency on the Stage 2 test split** (FA halved @1e-2, cut 3× @1e-3; AUC up)
- [x] The classes that still fool the detector are named, with example figures —
      Extremely_Loud, Repeating_Blips, Koi_Fish, Scratchy (gallery in
      `outputs/3_dataset_check.png`, per-class bars in `outputs/6_eval.png`)
- [x] The why is written down — the saturation bug box (Step 4) and the three-story
      per-class breakdown (Step 5)

→ Then **[[GW Signal Classifier - Brainstorm|Stage 4]]**: the matched-filter baseline, at
equal false-alarm rate, on data that now includes labelled glitches — the fight the whole
project was built to referee.
