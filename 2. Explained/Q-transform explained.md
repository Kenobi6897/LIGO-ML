---
tags: [ligo, signal-processing, reference, explainer]
status: reference
created: 2026-07-11
parent: "[[Stage 0]]"
---

# The Q-Transform, Explained

*The time-frequency view from [[Stage 0]] (`3_qtransform.png`). Worth understanding properly, because **this is the representation Gravity Spy works in** — our Stage 3 hard negatives are defined by their shape in this plane.*

---

## The problem it solves

Three ways to look at a signal. The first two are both useless for a chirp:

| View | Shows | Why it fails here |
|---|---|---|
| **Time series** (amplitude vs. time) | *When* things happen | No frequency info. A wiggle is a wiggle. |
| **FFT** (amplitude vs. frequency) | *What frequencies* are present | Time is thrown away entirely. "Power at 35–250 Hz" describes both a chirp **and** a random glitch. |
| **Time-frequency** (Q-transform) | Both at once | ✅ This is the one. |

**A chirp is *defined* by frequency changing over time.** Neither of the first two can express that. You need both axes simultaneously.

---

## The catch: you can't have both perfectly

A hard limit, not an engineering shortfall — the signal-processing cousin of Heisenberg (the Gabor limit).

To measure a frequency you must **watch for a while**. The longer you watch, the sharper your frequency estimate — but the vaguer you are about *when* it happened.

- **Short window** → sharp in time, blurry in frequency
- **Long window** → sharp in frequency, blurry in time

### Why a plain spectrogram (STFT) fails
An ordinary spectrogram picks **one fixed window length** and applies it at every frequency. For a chirp that choice is wrong *everywhere at once*:

- At **35 Hz**, one cycle takes **28 ms**. You need several cycles to measure the frequency at all → you need a **long** window.
- At **250 Hz**, the whole thing is over in a few ms. A long window smears it into mush → you need a **short** window.

One fixed window cannot serve both ends of the sweep. A plain spectrogram of GW150914 looks bad no matter what you set it to.

---

## What the Q-transform does differently

> **It scales the window with frequency.** Each frequency gets a window that is a fixed number of ***cycles*** long, not a fixed number of ***seconds***.

That's what **Q** means:

$$Q \approx \frac{f}{\Delta f} \approx \text{number of cycles in the analysis window}$$

| | Window | Good for |
|---|---|---|
| **High Q** | many cycles → narrow in freq, long in time | sustained tones (instrumental lines) |
| **Low Q** | few cycles → broad in freq, short in time | clicks, bursts, blips |

Holding Q **constant** means low frequencies automatically get long windows and high frequencies get short ones — **exactly the adaptive behaviour a chirp needs.** It's a constant-Q (wavelet-like) transform rather than a fixed-window one.

---

## What we actually called

```python
q = raw.q_transform(
    outseg=(GPS_MERGER - 0.3, GPS_MERGER + 0.15),  # time window to display
    qrange=(20, 60),                                # SEARCH Q in this range
    frange=(20, 400),                               # frequencies to cover
)
```

Two non-obvious things about this call:

### 1. It's a Q-*scan*, not a single transform
`qrange=(20, 60)` does **not** set Q. It says *"try a range of Q values and return the plane from whichever gave the loudest response."* The transform tiles the **(time, frequency, Q)** space and returns the best-fitting slice.

**This is why it finds the chirp without us telling it the shape in advance.**

### 2. We passed the RAW strain, not the whitened data
`q_transform` **whitens internally** (`whiten=True` by default). Feeding it our already-whitened series would whiten *twice* and distort the result.

⚠️ This is why `main()` hands `q_transform` the `raw` series while plots 1 and 2 get the conditioned one. Easy to "fix" by accident and break.

---

## Reading the picture

### The bright arc — the signal
Sweeps from **~35 Hz up to ~250 Hz**, then **cuts off abruptly at t = 0**.

- The black holes orbit **faster** as they spiral inward, and GW frequency = **2× orbital frequency**.
- It curves upward ever more steeply because the inspiral *accelerates*: $f(t) \propto (t_{\text{merge}} - t)^{-3/8}$
- The abrupt cutoff **is the merger** — the moment they become one black hole and there's nothing left to orbit.

### The horizontal bands — the instrument
Power-line harmonics, violin modes of the mirror suspensions. **Constant frequency, indefinite duration.**

Note they look *nothing* like the arc. **That's the whole point:** a chirp's signature is that it **moves diagonally** through this plane, and almost no instrumental artifact does that.

> **Noise wanders. A chirp has a direction and a deadline.**
> That structure is what the CNN will be learning.

### The colorbar
Normalized energy is scaled so that **pure noise averages 1**. A value of 49 means *"49× the energy you'd expect from noise here."*

We plot the **√** of it purely for visual contrast (it compresses the peak so the faint early inspiral stays visible), so the displayed 0–7 range corresponds to a real energy of **0–49**. Axis is labelled `√(normalized energy)` accordingly.

---

## Why this matters for the project

This time-frequency view is essentially **what Gravity Spy shows its classifiers.** Glitches are *named by their shape here*:

- **Blip** — short, broadband, symmetric
- **Scattered Light** — low-frequency arches
- **Whistle** — swooping tones
- **Violin Mode Harmonic** — those horizontal lines

**Our [[GW Signal Classifier - Brainstorm|Stage 3]] hard negatives are defined in this space.**

### The design fork it exposes
Do we feed the CNN the **1D time series**, or **this 2D image**?

| | 1D time series | 2D Q-transform image |
|---|---|---|
| Used by | the **detection** literature | the **glitch-classification** literature (Gravity Spy) |
| Closer to | matched filtering | computer vision; can borrow pretrained image models |
| Our choice | ✅ **this one** | (the "YOLO-style" framing in the original brief lives here) |

We picked **1D** — and this plot makes concrete *what information we're asking the network to reconstruct on its own*. The diagonal sweep is plainly visible to us in 2D; the CNN has to infer it from a 1D wiggle.

*Worth revisiting if the 1D model underperforms.*
