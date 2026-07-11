---
tags: [ligo, machine-learning, decision, explainer]
status: decided
created: 2026-07-11
decision: "1D primary, 2D as a Stage 3 comparison arm"
parent: "[[GW Signal Classifier - Brainstorm]]"
---

# 1D vs 2D — Architecture Decision

> **DECIDED: Build 1D. Keep the [[Q-transform explained|Q-transform]] code as a Stage 3 comparison arm, not a dead end.**

*Written down because in three weeks you'll be staring at a mediocre 1D result wondering if you should have gone 2D. The answer is **no** — and this is why.*

---

## The two options

| | What the network eats |
|---|---|
| **1D** | The whitened, bandpassed **time series**. A 1D CNN slides kernels along it. |
| **2D** | The **Q-transform image** ([[Q-transform explained]]). A 2D CNN treats it as a picture. |

---

## The decisive argument: *the representation must not be the confound*

The spine of this project is **benchmarking against matched filtering**. That's the thing that makes it more than a toy.

**Matched filtering operates on the time series, *with phase*.** Its entire power comes from **coherent integration** — lining up the template's phase with the data's phase and summing. That coherence *is* what makes it optimal (see [[GW Signal Classifier - Brainstorm|the Neyman–Pearson point]], and [[Matched filtering explained]] for why).

**A Q-transform magnitude image throws phase away.**

So if we build a 2D CNN and it loses to matched filtering, **we have learned nothing.** We cannot distinguish:

- ❌ *"The CNN is a worse detector than MF"* ← the interesting result
- ❌ *"I deleted the exact information MF wins with, then acted surprised"* ← a self-inflicted wound

> **The confound eats the experiment.**

With **1D**, both methods see the same input and the same information. **The comparison actually means something.**

---

## The tradeoffs, honestly

| | **1D — time series** | **2D — Q-transform image** |
|---|---|---|
| **vs. matched filter** | ✅ Fair — same input, same information | ❌ **Confounded** — phase discarded |
| **Literature** | ✅ *Detection* lit (Gabbard; George & Huerta) → **reference results to reproduce** | *Glitch-classification* lit (Gravity Spy) |
| **Learning difficulty** | ⚠️ Harder — must learn time-frequency analysis itself | ✅ Easy — the chirp is *visually obvious* |
| **Transfer learning** | ❌ None. Train from scratch. | ✅ Pretrained ResNet/EfficientNet, fast convergence |
| **Low SNR** *(the interesting regime)* | ✅ Phase coherence available | ❌ Magnitude-only can't integrate coherently |
| **Glitch discrimination** | ⚠️ Harder | ✅ Gravity Spy classes are *literally defined* by 2D shape |
| **Interpretability** | ⚠️ Wiggles. Hard to eyeball. | ✅ See *why* a false positive fired. Grad-CAM works. |
| **Cost** | ✅ ~2048 floats/segment | ⚠️ ~8× bigger + a Q-transform per sample — **bites our 16 GB** |

---

## The thing that makes 1D genuinely exciting

> ### A 1D convolution **is** a matched filter.

That is not an analogy. A conv kernel slides along the signal computing a dot product at each lag — which is **precisely** what correlating against a template does.

So **the first layer of the trained 1D CNN is, functionally, a learned template bank.**

After training, plot those kernels. You should see **chirp-shaped filters that the network invented on its own**, without ever being told what a black hole is.

- That is the **single best figure** this project will produce.
- It's a real insight into *why* a CNN can approach MF performance at all.
- **The 2D route severs this connection entirely.**

- [ ] **TODO (Stage 1):** after training, plot first-layer conv kernels. Look for chirps.

---

## Where 2D genuinely wins

Not pretending it has no case. **Glitch rejection is inherently a shape problem in the time-frequency plane** — a *Blip* and a chirp are far more separable as **images** than as **wiggles**.

And per the brainstorm, **glitch robustness is our most interesting result** — it's the one place a CNN can beat MF *in practice*, because glitches break MF's Gaussian-noise assumption.

### So the two goals genuinely pull apart:
| Goal | Wants |
|---|---|
| **Benchmark fairness** | → **1D** |
| **Glitch rejection** | → **2D** |

**Notice this rather than papering over it.**

---

## Resolution: don't choose — *measure*

**1D is primary. Add 2D as a comparison arm at [[GW Signal Classifier - Brainstorm|Stage 3]]**, when the Gravity Spy hard negatives arrive.

By then the dataset and eval harness already exist — we're only swapping the front end, so this costs **almost nothing**.

Then report both:

> *"1D matches matched filtering on sensitivity but is fooled by Blips; 2D sacrifices low-SNR sensitivity but rejects glitches cleanly."*

That's a **real finding**. It's honest, and it's a far better conclusion than either arm alone.

---

## Caveat: the hybrid that dissolves the main objection

Feed the CNN the **complex** spectrogram — real + imaginary as **two channels** — which **preserves phase**. You'd get 2D structure *and* a fair benchmark.

**Don't start here.** It's an unusual setup with **no reference results**, so if it underperforms you won't know whether it's the idea or the implementation. That's the same trap as starting in 2D.

**Do reach for it if** Stage 3 shows 1D losing badly to glitches and we don't want to pay the phase cost to fix it.

*(Related: a learned front-end, SincNet-style. Scope creep. Noted, not planned.)*

---

## Summary

1. **1D**, because it's the only **fair** benchmark against matched filtering — and that benchmark is the project.
2. **Reference results exist** (Gabbard), so we can tell whether we're *correct*, not just whether we're *done*.
3. **Learned conv kernels ≈ a template bank** — the best insight and the best figure available here.
4. **Cheaper**, which matters on a 16 GB box.
5. **Low SNR is where the comparison gets interesting**, and 2D magnitude throws away the phase coherence needed there.
6. **2D is not discarded** — it's the Stage 3 comparison arm. The [[Stage 0]] Q-transform code is already written.
