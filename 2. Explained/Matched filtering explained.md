---
tags: [ligo, signal-processing, reference, explainer]
status: reference
created: 2026-07-11
parent: "[[GW Signal Classifier - Brainstorm]]"
---

# Matched Filtering, Explained

*The incumbent. The whole project is a benchmark against this method, so it's worth understanding
properly rather than treating it as a black box labelled "the thing physicists use." In particular:
**understanding why it's optimal tells you exactly where it isn't** — and that gap is where
[[GW Signal Classifier - Brainstorm|Stages 2–4]] live.*

---

## The core idea

**We know what we're looking for.** General relativity predicts the *exact* waveform two black
holes of given masses produce as they spiral in. The shape is not the mystery.

The mystery is that the signal sits ~1000× **below** the noise floor. Plot raw LIGO strain and you
see a wall of seismic noise and nothing else — see [[Stage 0]].

So don't try to *see* the signal. **Correlate.**

1. Take the predicted waveform — the **template**.
2. Slide it along the data.
3. At every time offset, compute the **dot product** of template against data.

Where the template lines up with a real signal, the products reinforce and the sum spikes. Over
noise, the products have random signs and largely cancel. The output is one number per offset — the
**SNR time series** — and detection is just *"did it cross threshold anywhere."*

### Why it works: coherent vs. incoherent accumulation
The noise doesn't cancel *perfectly*. It cancels **incoherently**, while the signal adds
**coherently**:

| | Grows like |
|---|---|
| Signal (adds in phase) | **N** |
| Noise (random signs) | **√N** |

So effective SNR grows like **√N** over N correlated samples. **You dig the signal out from under
the noise not by removing noise, but by accumulating agreement over thousands of samples.** That
asymmetry is the entire game.

---

## Why "matched" — and why whitening is half the filter

The filter is *matched* to the waveform you expect. But LIGO's noise is **not equally bad at all
frequencies**: seismic rumble dominates low, shot noise dominates high, with a quiet band between.
A plain dot product would let the loud, useless bins swamp the informative ones.

So the real matched filter **weights each frequency by the inverse of the noise power there** —
down-weight the loud frequencies, trust the quiet ones.

> **That is exactly what whitening does.**

Which means [[Stage 0]]'s pipeline (whiten → bandpass) is not cosmetic preprocessing. **It is the
first half of a matched filter.** In practice: whiten the data *and* the template, then correlate.

---

## Phase is where the power lives ⚠️

**This is the part that drives the [[1D vs 2D - decision explained|architecture decision]], and
it's the easiest thing here to skim past.**

The dot product is **signed**. It demands that the template's peaks land on the data's peaks and its
troughs on the data's troughs — not merely that there's *"energy around 100 Hz at roughly the right
time."* That phase alignment is **what makes the noise cancel**.

This is called **coherent integration**, and it is the entire source of matched filtering's
sensitivity — most acutely at **low SNR**, where you need every scrap of cancellation available.

> **Consequence:** a [[Q-transform explained|Q-transform]] *magnitude* image has **thrown the phase
> away.** Benchmark a phase-blind CNN against a phase-exploiting matched filter and you have not
> learned that the CNN is worse — you have learned that **you deleted the information the baseline
> wins with, then acted surprised.** The confound eats the experiment.

**1D keeps both methods looking at the same information.** That's why the comparison means anything
at all. Full argument: [[1D vs 2D - decision explained]].

---

## Why we cannot beat it — and why that's fine

For a **known waveform in Gaussian noise**, matched filtering is **provably optimal**: it *is* the
Neyman–Pearson likelihood-ratio test for that problem. No detector of any kind — no CNN, no
transformer, nothing — does better at a given false-alarm rate.

**That is a theorem, not a strong empirical result.** There is no room above it.

> 🚩 **So if the [[Stage 1]] CNN beats matched filtering, that is a bug** — almost certainly
> [[Stage 1#3. Footguns|leakage]]. The correct Stage 1 outcome is landing **near** it.

Which sounds like a deflating goal — until you notice **what the theorem quietly assumes.**

---

## The two cracks — and the project drives into both

The guarantee is **conditional**. Both conditions are attackable.

### Crack 1 — *"known waveform"* is doing a lot of work
We don't know the masses in advance. So one template won't do: you need a **template bank** spanning
the parameter space, and you correlate against **every single one**. Real searches use hundreds of
thousands of templates.

Three consequences:
- **Cost.** It's CPU-bound and embarrassingly parallel — this is what the 3700X's 16 threads are
  for in [[GW Signal Classifier - Brainstorm|Stage 4]].
- **Latency.** A CNN approximates the *whole bank* in one forward pass. **This is why LIGO people
  actually care about ML.**
- **Blindness.** MF can only find what is *in* the bank.

### Crack 2 — *"Gaussian noise"* is straightforwardly false ⭐
Real LIGO data is full of **glitches** — non-Gaussian instrumental transients.

**The moment the noise stops being Gaussian, the optimality theorem simply does not apply** — and MF
has no principled defence. A glitch that happens to correlate with a template rings the SNR bell
exactly like a real signal would.

> **This crack is where this project lives.**

| Stage | Noise | What MF is doing there |
|---|---|---|
| **[[Stage 1]]** | Simulated Gaussian | 🧪 The **clean room** — MF is unbeatable *by theorem*. Matching it **proves our pipeline is correct.** |
| **Stage 2** | Real O3 noise | The assumption starts to crack. Expect our performance to drop. |
| **Stage 3** | + Gravity Spy glitches | ⭐ Assumption **broken on purpose**. The CNN's genuine, non-theorem-violating shot at winning. |
| **Stage 4** | — | The head-to-head, compared **at equal false-alarm rate**. |

**Stage 1 uses Gaussian noise *precisely because* MF is unbeatable there.** That's not a limitation
of the stage — it's the point of it.

---

## The bit that ties it all together 🎁

> ### A 1D convolution **is** a matched filter.

Not an analogy. A conv kernel slides along the signal computing a dot product at each lag — which is
**precisely** what correlating against a template does.

So **the first layer of the trained 1D CNN is, functionally, a learned template bank.** After
training, plot those kernels: you should see **chirp-shaped filters the network invented on its
own**, never having been told what a black hole is.

That is the single best figure this project will produce, and it is the real reason we went 1D.
→ [[Stage 1|Stage 1, Step 7]].

---

## In one paragraph

Matched filtering correlates the data against a **bank of known templates**, weighting frequencies
by inverse noise power (**= whitening**) and integrating **coherently in phase**, so signal grows
like N while noise grows like √N. For a **known waveform in Gaussian noise** this is
**Neyman–Pearson optimal** — beating it means you have a bug. But it is **expensive** (whole bank,
every time), **blind** to anything outside the bank, and **undefended** against non-Gaussian
glitches, which is exactly where a CNN can win **in practice** rather than in theory.

---

## Related
- [[1D vs 2D - decision explained]] — why phase makes 1D the only fair benchmark
- [[Q-transform explained]] — the representation that discards phase
- [[Stage 0]] — where whitening (half of MF) already got built
- [[Stage 1]] — the clean-room comparison
- **Gabbard et al. 2018 — "Matching Matched Filtering with Deep Learning"** — https://arxiv.org/abs/1712.06041
- `pycbc.filter.matched_filter` — https://pycbc.org
