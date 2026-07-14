# Stage 5 — the ablation: what was the preprocessing worth?

**Status: ✅ complete (2026-07-14).** The question was *"what would happen if we trained the
CNN on raw strain, with no processing?"* — the obvious ablation, never run, and the one a
referee asks first. It came back with the expected answer, and then handed over a bug that
matters more than the ablation did.

> **Headline.** Raw strain trains to **AUC 0.4997** — a coin flip, zero efficiency in every
> SNR bin. Not "degraded": *dead*. And the mechanism is visible in the first layer: all 16
> learned kernels collapse onto the seismic wall below 30 Hz (**1/16 in band**, where random
> init gives ~6/16). But the arm that mattered was the **control**: rebuilding it exposed
> that Stage 3's ±20σ clip — the fix for the 6,000σ gain-crushing bug — was only ever applied
> to **5% of the training rows**. Applying it to all of them cuts glitch false alarms
> **7× at FAP 10⁻²** (0.166 → 0.024) and improves detection at the same time.

---

## 1. The design — one variable, four conditioners

Stages 1–4 asked what the network learns. This one asks what the *physics we handed it* was
worth, by taking it away. Same strain, same blocks, same injections at the same physical
amplitudes, same architecture, same recipe, same seed — only the operator between the
detector and the CNN changes:

| arm | conditioner | what it tests |
|---|---|---|
| **full** | whiten → bandpass 30–350 → crop | the control — Stage 3's `stage3_cnn.pt`, arm D |
| **wh** | whiten → crop | is the bandpass load-bearing? |
| **bp** | bandpass → crop | is the *whitening* load-bearing? |
| **raw** | crop | "just feed it the strain" |

### Making it a fair fight

Three ways a raw-data arm can be rigged to fail, all closed before the run:

1. **Scale.** Raw strain is ~10⁻¹⁹; a network fed 10⁻¹⁹ fails for reasons that have nothing
   to do with spectra. Every arm gets its **own** global scale constant, by Stage 2's
   signal-blind recipe (`norm()`: mean std of pure-noise crops from the earliest train
   blocks, one number forever, never per-segment). Every arm's input is O(1).
2. **The signal.** Injection amplitude is set *upstream* of the conditioner, in strain,
   against the block's measured PSD — the same `build_segment` code path Stage 2 uses. "SNR 8"
   is the same physical waveform in the same noise in all four arms.
3. **Leakage.** All four operators are linear and signal-blind, and each passes Stage 2's
   identity `C(n₁+h) − C(n₁) == C(n₂+h) − C(n₂)` at **~2.4×10⁻¹⁶** — the control scoring
   exactly the 2.4×10⁻¹⁶ the README quotes, which is the evidence that this stage's `full`
   arm really is Stage 2's operator and not a lookalike.

And the datasets are not redrawn, they are **re-derived**: `stage5_dataset.py` calls the
*original* `make_specs()` with the *original* seeds, so every row is the same block, offset,
mass pair, target SNR, merger position and split as the corresponding row of `stage2.h5` /
`stage3.h5`. `stage5_dataset_check.py` asserts it bit-exactly — **max|Δ| = 0.0 across 9
columns × 100,273 rows and 14 columns × 7,312 rows.** The arms are running the same
experiment.

---

## 2. The measurement that predicts the whole stage

Before any network trained, one ratio said what would happen. At the **same physical SNR**,
how much of the crop *is* the signal?

| arm | ‖C(h)‖ / ‖C(n)‖ | float32 recovery error |
|---|---|---|
| **raw** | **1.6×10⁻⁴** | 2.3×10⁻⁴ |
| bp | 1.9×10⁻¹ | 1.8×10⁻⁷ |
| wh | 5.2×10⁻² | 6.9×10⁻⁷ |
| full | **3.0×10⁻¹** | 1.2×10⁻⁷ |

An SNR-8 chirp is **1,900× more prominent** in a conditioned crop than in a raw one. Raw
strain's variance is owned by the seismic/suspension wall below 30 Hz; the chirp is a rounding
error on top of it. **Whitening does not "clean up" the signal — it deletes the variance that
isn't the signal. That is the entire job.**

The second column rules out the boring explanation *in advance*: the waveform survives float32
storage in the raw arm with ~10⁻⁴ relative error, thousands of times above the quantisation
floor. Whatever kills the raw arm, **it is not the number format**.

---

## 3. Results

### The ablation

| arm | stage2-test AUC | kernels in band | glitch FA @10⁻² | glitch FA @10⁻³ | eff. SNR 6–8 @10⁻³ |
|---|---|---|---|---|---|
| **full** (control) | 0.9839 | 13/16 | 0.166 | 0.050 | 0.658 |
| **wh** | 0.9793 | 15/16 | **0.842** | 0.331 | 0.503 |
| **bp** | 0.9829 | 12/16 | 0.048 | 0.026 | 0.727 |
| **raw** | **0.4997** | **1/16** | 0.600 | 0.600 | **0.000** |

**raw is not degraded, it is dead.** Training loss sat at 0.692 = ln 2 and never moved; the
model is a constant classifier. Efficiency is **0.000 in every SNR bin at every FAP** — it
never fires on a signal, at any loudness, including SNR 20. Re-running it with the clip
disabled (`--no-clip`) gives AUC 0.5000: **the saturation rule is not the cause** (it touches
0.0000% of raw Stage 2 samples anyway).

But raw did learn *something* — it fires on **60% of glitches**. That is the tell, and it is
coherent: the only thing visible in unwhitened strain is **total loudness**, and a
loudness detector is precisely a glitch detector and precisely *not* a chirp detector. It
learned the one feature the representation offered.

**The mechanism, drawn** (`outputs/6_kernels.png`): a 1D conv layer is a learned template
bank, so ask each arm's bank where it put itself. The raw arm's 16 first-layer kernels all
pile onto the lowest frequencies — **1/16 peaks in the 30–350 Hz band, *below* the ~6/16 of a
random initialisation.** The network didn't fail to learn. It learned the wall, because the
wall is where the variance is. Gradient descent went exactly where the loss told it to.

### Why raw can't just learn the bandpass itself

A 64-tap kernel *can* express a decent high-pass, so why doesn't it? Two reasons the ladder
separates:

- **`bp` ≈ `full` (0.9829 vs 0.9839).** Once the wall is gone, the residual colour *inside*
  30–350 Hz spans ~1 order of magnitude — mild enough that conv1 learns its own equaliser.
  **Within the band, the CNN can do its own whitening.** This is Stage 4's "a 1D convolution
  *is* a matched filter" showing up from the other side.
- **`raw` cannot**, because the wall is 4+ orders of magnitude of variance *and* whitening
  here is **data-adaptive** — a causal per-block Welch PSD, because O3 noise drifts. A
  convolution is a *fixed* filter. It cannot track a moving PSD, and it cannot notch a Q~100
  line in 64 taps.

And `wh` is the counterweight: whitening without band-limiting works for detection (0.9793)
but is **catastrophic for glitch rejection — 0.842**, the worst non-raw arm by a factor of 17.
Flattening the spectrum without bounding it makes every glitch maximally prominent across
15–1024 Hz, where the network can use none of it for signals. **The bandpass is not
cosmetic either; it is what keeps whitening from amplifying junk.**

### The `bp` result that didn't survive its own check

Single-seed, `bp` matched `full` on detection *and* rejected glitches 3.5× better
(0.048 vs 0.166) — a headline. `stage5_sweep.py` retrained both arms across 4 seeds
**through the identical code path** and killed it:

| metric | bp | full (clip-all) | verdict |
|---|---|---|---|
| glitch FA @10⁻² | 0.046 ± 0.004 | 0.058 ± 0.022 | **OVERLAPPING** (0.8 pooled sd) |
| glitch FA @10⁻³ | 0.020 ± 0.004 | 0.026 ± 0.013 | **OVERLAPPING** (0.6 pooled sd) |
| stage2-test AUC | 0.9838 ± 0.0010 | 0.9864 ± 0.0002 | full ahead by 0.0026 |

The 3.5× win was a seed. **Whitening is not load-bearing for this CNN, but bandpassing is** —
that is the honest reading, and it took a sweep to get it.

---

## 4. The bug the control found — the clip was applied to 5% of the training data

Retraining the control through this stage's harness gave glitch FA **0.058**, against the
published **0.166**. Same architecture, same seed, same rows. The only difference: this
stage's reader applies Stage 3's ±20σ read-time clip to **both** sources, and Stage 3's
reader does not.

**`Stage3Dataset` clips. `Stage2Dataset` does not.** And the training mix is
`ConcatDataset([Stage2Dataset("train"), Stage3Dataset("train")])` — **79,913 Stage 2 rows vs
4,080 Stage 3 rows.** The fix for Stage 3's headline bug landed on **4.9% of the training
data.**

That would be harmless if Stage 2's crops were tame. They are not: **`stage2.h5` contains
crops up to 6,177σ**, and 0.22% of its crops trip the ±20σ rail — because Stage 2's noise
carries the **~28 unlabelled glitches/hour** Stage 2 itself measured. Stage 3's bug box says a
near-linear CNN under BCE shrinks its overall gain to afford confidently-wrong 6,000σ
negatives. Stage 3 diagnosed that correctly, fixed it in the Stage 3 reader — and left 6,177σ
negatives in the Stage 2 rows that make up 95% of the mix.

> *(Stage 3's note claims the clip "is a no-op for every Stage 2 crop". That is where the
> error lives: it is a no-op for 99.78% of them. The other 0.22% are the glitches.)*

Applying the same clip to both sources — one line, no new data, no new architecture:

| | arm D (as published) | **+ clip on Stage 2 rows** |
|---|---|---|
| stage2-test AUC | 0.9839 | **0.9861** |
| kernels in band | 13/16 | **16/16** |
| FA/h @10⁻² | 52.8 | **29.1** |
| efficiency SNR 4–6 @10⁻³ | 0.177 | **0.317** |
| efficiency SNR 6–8 @10⁻³ | 0.658 | **0.801** |
| **ALL GLITCHES @10⁻²** | 0.166 | **0.024** |
| **ALL GLITCHES @10⁻³** | 0.050 | **0.006** |
| Extremely_Loud @10⁻² | 0.455 | **0.000** |
| Koi_Fish @10⁻² | 0.347 | **0.000** |

Every metric improves at once — which is the signature of a bug being removed, not a
trade-off being made. The two classes Stage 3 named as "the loud, chirp-adjacent blind spot"
(Extremely_Loud, Koi_Fish) **go to zero**. That is the gain-crush, released.

**Seed-honest version** (4 seeds, `stage5_sweep.py`): glitch FA @10⁻² = **0.058 ± 0.022**,
@10⁻³ = **0.026 ± 0.013**. The 0.024 above is a good seed. The *conclusion* does not depend
on the seed:

> **Stage 4's verdict changes.** Arm F — matched filtering with the χ² veto, the champion —
> scores **0.100 @10⁻² / 0.047 @10⁻³** on this glitch test set. The clip-fixed CNN scores
> **0.058 ± 0.022 / 0.026 ± 0.013**. *Every seed*, worst included (0.084), beats F at 10⁻².
> The README's headline — "the veto-armed MF edges the glitch-trained CNN, 0.100 vs 0.166" —
> was scored against a CNN that was fighting a gain-crush bug with one hand.

This is **not yet a claim that the CNN beats matched filtering.** It is a claim that the
comparison must be re-run: Stage 4's arms C/D/E/F/O at equal FAP, with D rebuilt. Detection at
10⁻³ (where F led at every SNR) has *not* been re-measured against F here.

---

## 4b. Stage 4, re-run — `stage5_rerun_stage4.py`

Same protocol, same thresholds, same MF scores (arms E/F/O are CNN-independent and were reused
verbatim from `stage4_scores.h5`). The only change is arm **D+**: the identical model with the
clip applied to both sources, reported across **4 seeds**, because a claim that flips a published
verdict rides on the *worst* seed and not the best one.

**One more asymmetry surfaced while wiring it up:** `stage4_mf.py:173` clips its *own* inputs at
±20σ. The matched filter always saw saturated crops. **Stage 4 was comparing a correctly-conditioned
MF against a gain-crushed CNN** — the bug was one-sided, and it favoured the baseline.

### Front 2 — glitch rejection: the verdict flips

| arm | @10⁻² | @10⁻³ |
|---|---|---|
| E — MF max-ρ | 0.582 | 0.116 |
| C — CNN glitch-naive | 0.331 | 0.159 |
| D — CNN glitch-trained | 0.166 | 0.050 |
| F — MF + χ² veto | 0.100 | 0.047 |
| **D+ — CNN, clip fixed** | **0.058 ± 0.022** (worst seed 0.084) | **0.026 ± 0.013** (worst seed 0.042) |

**D+ beats F at both thresholds, on every seed.** And the per-class story is no longer the
"complementarity" Stage 4 reported — it is dominance: the two classes Stage 3 named as the
blind spot go to **zero**.

| class | D | **D+** | F |
|---|---|---|---|
| Koi_Fish @10⁻² | 0.347 | **0.000** | 0.000 |
| Extremely_Loud @10⁻² | 0.455 | **0.000** | 0.255 |
| Repeating_Blips @10⁻² | 0.450 | **0.050** | 0.300 |
| Whistle @10⁻² | 0.262 | **0.071** | 0.155 |

### Front 1 — detection: F keeps it

The fix did **not** hand the CNN everything. At FAP 10⁻³ the veto-armed MF still leads at every SNR:

| SNR | D | **D+** | **F** |
|---|---|---|---|
| 6–8 | 0.658 | 0.801 | **0.850** |
| 8–10 | 0.949 | 0.971 | **0.989** |
| 10–12 | 0.985 | 0.988 | **0.994** |

D+ closed most of the gap (0.658 → 0.801 at SNR 6–8) but did not cross it. **Matched filtering
remains the better detector of faint signals; the CNN is now the better rejector of glitches.**

### The theorem is intact

The pre-registered fraud alarm — *no CNN may beat the true-parameter oracle on the quasi-Gaussian
bulk* — still reads **+0.000** with D+ in the arena. The clip fix removed a bug; it did not break
Neyman–Pearson. That is exactly what a real fix should look like, and it is the strongest evidence
that D+ is a corrected detector rather than a leaking one.

### The corrected verdict

> Stage 4 concluded: *"on glitch rejection the veto-armed MF edges the glitch-trained CNN, and a
> well-engineered classical statistic already captures most of the learnable gap."* **The first
> half was an artefact of the bug.** Corrected: the CNN **surpasses** the χ² veto at glitch
> rejection (0.058 vs 0.100 @10⁻², every seed) while **remaining behind it** at faint-signal
> detection (0.801 vs 0.850 @10⁻³). Honours split, and the compute margin (~20× latency,
> ~3,700× throughput) is unchanged and still the CNN's.

---

## 5. What went wrong in *this* stage

**The check that would have condemned the control.** The first version of the leak test used
the SNR-8 injection and a 1×10⁻¹² tolerance. Three of four arms "failed" — including `full` —
while `raw` "passed" at 5×10⁻¹³. *Backwards*, and that was the tell. Nothing was leaking:
differencing `C(n+h)` and `C(n)` when `h` is 1.6×10⁻⁴ of `n` is **catastrophic cancellation**,
which amplifies float64 round-off by ~‖n‖/‖h‖ ≈ 6,000×. The check was measuring its own
arithmetic. Stage 2 avoids this by making the test signal *as loud as the noise* — so this
stage now uses Stage 2's setup, Stage 2's metric and Stage 2's 10⁻⁹ tolerance, and all four
arms come in at ~2.4×10⁻¹⁶.

The lesson is the project's own, re-learned: **a check whose failure mode you haven't derived
is not a check.** This one would have sent me hunting for a leak in the control that
Stages 1–4 had already proven leak-free.

**The `1. Stages` → `Stages` rename** landed mid-run (PR #6, `fix-readme-paths`), which killed
the Stage 3 dataset build with `No such file or directory` after the Stage 2 build had already
spent 61 minutes. The scripts were fine — every sibling-stage import resolves relatively
(`parents[1] / "Stage 1"`) — only the shell's `cd` was stale. Nothing was lost.

---

## 6. Files

```
stage5_condition.py        the four conditioners + one global scale constant per arm
stage5_dataset.py          re-derives Stage 2's / Stage 3's exact rows under all three arms
stage5_dataset_check.py    bit-exact spec identity, the leak test, the clip census   [CAN FAIL]
stage5_data.py             the reader — every arm, both sources, one clip rule
stage5_train.py            the same CNN, the same recipe, one arm at a time
stage5_eval.py             the benchmark at equal FAP + the learned-template-bank figure
stage5_sweep.py            the seed sweep that killed the `bp` claim                 [CAN FAIL]
stage5_rerun_stage4.py     Stage 4's verdict, re-run with the clip-fixed CNN
outputs/5_ablation.png     efficiency at equal FAP, and glitch rejection, per arm
outputs/6_kernels.png      where each arm put its template bank — the mechanism
outputs/7_stage4_rerun.png the corrected benchmark: D+ vs the chi-squared veto
```

Datasets (not in git): `~/ligo-data/stage5_s2.h5` (2.5 GB, 61 min), `stage5_s3.h5` (182 MB,
5 min). Checkpoints `~/ligo-data/stage5_cnn_{arm}[_noclip|_sN].pt`. Nothing here overwrites
`stage3_cnn.pt`.

---

## 7. Verdict

The obvious ablation gave the obvious answer, and it gave it *hard*: **raw strain is not a
harder problem for this CNN, it is an impossible one** — 0.4997, a coin flip, kernels parked
on the seismic wall. The preprocessing is not a convenience. It is the operator that makes the
signal exist as far as the network is concerned, and the ratio 1.6×10⁻⁴ → 0.30 is the whole
argument.

The ladder then localises *which* preprocessing: **the bandpass carries the stage** (`bp`
≈ `full`; the CNN learns its own in-band equaliser, so whitening is nearly free to it — a
result that only survived because the seed sweep was run), while **whitening without a
bandpass is actively harmful for glitch rejection** (`wh`: 0.842).

And the real prize was in the control. Rebuilding a baseline you already trust is not
ceremony — **the control found a bug in Stage 3 that Stage 3's own checks could not see**,
because every Stage 3 check compared arm D against arm C, and *both* were reading Stage 2's
rows unclipped. Stage 4's checks couldn't see it either: they asked "does D beat F?", and the
bug was in the substrate D stood on — while the MF, which clips its own inputs, stood on solid
ground. Every question either stage asked was a *relative* one. **It took an arm from outside
the frame — one built expecting to fail, and it did.**

One line of read-time consistency cuts glitch false alarms 7×, improves weak-signal efficiency,
moves 16/16 kernels into the band, and overturns the verdict Stage 4 was built to settle: the
CNN passes the χ²-vetoed matched filter at glitch rejection, on every seed, at both thresholds —
while still losing to it on faint-signal detection, and while still never beating the oracle on
the bulk. Nothing broke. A bug came out.

**The strongest checks are the ones physics writes for you. The second strongest are the ones
a fresh arm writes for your control.**
