"""Stage 1, Step 5's sanity check — does the model fire on real GW150914?

The plan's rule: *if the model can't fire on GW150914, the model is broken.* Every
training positive was synthetic; this is the one place Stage 1 touches a real black hole
merger. It is a SANITY CHECK, not a result — the model was trained on simulated Gaussian
noise at design sensitivity, and real O1 noise is neither Gaussian nor at design
sensitivity. A pass here says the pipeline plumbing and the learned features survive
contact with reality; what performance on real noise actually looks like is Stage 2, and
the plan expects it to be worse.

HOW THE REAL DATA IS CONDITIONED — the leak rules apply even harder here:
  - We cannot whiten with the design PSD (O1's noise floor is genuinely different), and
    we MUST NOT estimate the PSD from the segment being scored ([[Stage 1#Footguns|THE
    footgun]]). So the PSD is Welch-estimated from a separate off-source stretch minutes
    before the event, then applied identically to every segment scored — the event and
    all of the background. Same fixed filter for everything, exactly like condition().
  - The global scale constant is likewise measured on off-source segments only.
  - Everything downstream is byte-for-byte the Stage 1 recipe: 4 s buffer, whiten with
    the fixed PSD (0.5 s inverse-spectrum truncation), 30-350 Hz zero-phase FIRs, crop
    the central 1 s, one global scale.

THE CHECK THAT CAN FAIL: the event's score must beat every one of ~100 off-source 1 s
background segments drawn from the same conditioned stream. Ranking against a measured
background is the honest form of "it fired" — an absolute p > 0.5 would be meaningless,
since the sigmoid saturates and the background sets the scale of false alarms.

GW150914, H1: merger GPS 1126259462.423 ([[Stage 0]] — .4 puts the chirp 23 ms
off-centre). H1 optimal SNR ~20: inside the training range (4-20), so the model has seen
signals this loud, in this band, at these masses (36+29 brackets inside [10, 50]).

Data comes from GWOSC via gwpy (network needed on first run; cached to ~/ligo-data
afterwards, where .gitignore keeps it out of the vault).

Run (from WSL2, after stage1_train.py):
    source ~/venvs/ligo/bin/activate
    cd "/mnt/c/Users/locke/Documents/LIGO-ML/1. Stages/Stage 1"
    python stage1_gw150914.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside WSL2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.types import TimeSeries

from stage1_condition import BUFFER_LEN, BUFFER_N, CROP, CROP_N, CROP_START, FIR_ORDER, INV_SPEC_LEN
from stage1_model import load_checkpoint
from stage1_noise_check import BAND, FLOW, SAMPLE_RATE
from stage1_train import DEFAULT_CKPT

OUT = Path(__file__).parent / "outputs"
CACHE = Path.home() / "ligo-data" / "gw150914_h1_2048.npz"

GPS_MERGER = 1126259462.423  # Stage 0's number — .4 is 23 ms off
MERGER_POS = 0.85  # merger at 85% of the crop, mid-range of the training placement

# Three disjoint stretches of H1 data, all ending well before the event:
#   PSD stretch     [merger-256, merger-128)  -> the fixed whitening PSD
#   background      [merger-124, merger-8)    -> scale constant + background scores
#   event buffer    4 s around the merger
FETCH = (GPS_MERGER - 258.0, GPS_MERGER + 6.0)
PSD_STRETCH = (GPS_MERGER - 256.0, GPS_MERGER - 128.0)
BG_STRETCH = (GPS_MERGER - 124.0, GPS_MERGER - 8.0)


def fetch_strain() -> tuple[np.ndarray, float]:
    """H1 strain for FETCH, resampled to 2048 Hz. (values, gps_of_first_sample)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["strain"], float(z["t0"])
    print(f"fetching H1 {FETCH[0]:.0f}-{FETCH[1]:.0f} from GWOSC (~264 s, first run only) ...")
    from gwpy.timeseries import TimeSeries as GwpyTS

    ts = GwpyTS.fetch_open_data("H1", *FETCH)  # 4096 Hz
    ts = ts.resample(SAMPLE_RATE)
    strain, t0 = ts.value.astype(np.float64), float(ts.t0.value)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, strain=strain, t0=t0)
    print(f"  cached -> {CACHE}")
    return strain, t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 — GW150914 sanity check")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt)
    print(f"checkpoint: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}\n")

    strain, t0 = fetch_strain()

    def buffer_at(t_start: float) -> TimeSeries:
        i0 = int(round((t_start - t0) * SAMPLE_RATE))
        if i0 < 0 or i0 + BUFFER_N > len(strain):
            raise ValueError(f"buffer at {t_start} falls outside the fetched stretch")
        return TimeSeries(strain[i0 : i0 + BUFFER_N].copy(), delta_t=1.0 / SAMPLE_RATE)

    # --- the fixed whitening PSD, from a stretch that contains no event ---------------
    i0 = int(round((PSD_STRETCH[0] - t0) * SAMPLE_RATE))
    i1 = int(round((PSD_STRETCH[1] - t0) * SAMPLE_RATE))
    psd_data = TimeSeries(strain[i0:i1].copy(), delta_t=1.0 / SAMPLE_RATE)
    psd = welch(psd_data, seg_len=4 * SAMPLE_RATE, seg_stride=2 * SAMPLE_RATE)
    psd = interpolate(psd, 1.0 / BUFFER_LEN, BUFFER_N // 2 + 1)
    psd = inverse_spectrum_truncation(
        psd, max_filter_len=int(INV_SPEC_LEN * SAMPLE_RATE),
        low_frequency_cutoff=FLOW, trunc_method="hann",
    )

    def condition_real(buf: TimeSeries) -> np.ndarray:
        """Stage 1's condition(), with the measured off-source PSD in place of the design
        curve. Same fixed filter for every segment — event and background alike."""
        white = (buf.to_frequencyseries() / psd ** 0.5).to_timeseries()
        band = highpass_fir(white, BAND[0], FIR_ORDER)
        band = lowpass_fir(band, BAND[1], FIR_ORDER)
        return band.numpy()[CROP]

    # --- background: disjoint 1 s crops from the off-source stretch --------------------
    # Buffers hop by 1 s, so consecutive CROPS are disjoint (each crop is the central 1 s
    # of its 4 s buffer). The same segments set the global scale — off-source only.
    bg_starts = np.arange(BG_STRETCH[0], BG_STRETCH[1] - BUFFER_LEN, 1.0)
    bg = np.stack([condition_real(buffer_at(t)) for t in bg_starts])
    norm = 1.0 / float(np.mean(bg.std(axis=1)))
    bg = (bg * norm).astype(np.float32)

    # --- the event: merger at MERGER_POS of the crop -----------------------------------
    event_start = GPS_MERGER - (CROP_START + MERGER_POS * CROP_N) / SAMPLE_RATE
    x_event = (condition_real(buffer_at(event_start)) * norm).astype(np.float32)

    with torch.no_grad():
        bg_logits = model(torch.from_numpy(bg).unsqueeze(1)).numpy().ravel()
        ev_logit = float(model(torch.from_numpy(x_event).reshape(1, 1, -1)))
    ev_p = 1.0 / (1.0 + np.exp(-ev_logit))

    n_bg = len(bg_logits)
    n_above = int((bg_logits >= ev_logit).sum())
    print(f"conditioned background: {n_bg} disjoint 1 s segments, "
          f"std of stream = {bg.std():.3f} (want ~1)")
    print(f"\n  event logit : {ev_logit:+8.2f}   (p = {ev_p:.4f})")
    print(f"  background  : median {np.median(bg_logits):+.2f}, "
          f"max {bg_logits.max():+.2f}")
    print(f"  rank        : {n_above} of {n_bg} background segments score above the event")

    ok = n_above == 0 and ev_p > 0.5
    print(f"\n  {'PASS' if ok else 'FAIL'}: the model {'fires' if ok else 'does NOT fire'} "
          f"on real GW150914 — trained on synthetic injections only")
    print("  (sanity check, not a result: O1 noise is not the design-sensitivity Gaussian"
          "\n   this model was trained on. Real-noise performance is Stage 2's question.)")

    # --- figure ------------------------------------------------------------------------
    fig, (axt, axh) = plt.subplots(1, 2, figsize=(13, 5))

    t_ms = (np.arange(CROP_N) / SAMPLE_RATE - MERGER_POS) * 1000  # 0 = merger
    axt.plot(t_ms / 1000 + MERGER_POS, x_event, lw=0.8, color="tab:blue")
    axt.axvline(MERGER_POS, color="tab:red", lw=1.2, ls="--",
                label=f"merger GPS {GPS_MERGER}")
    axt.set_xlabel("time in crop [s]")
    axt.set_ylabel("whitened strain [σ]")
    axt.set_title("GW150914 (H1), conditioned exactly like the training data\n"
                  "(off-source Welch PSD — never the segment's own)")
    axt.legend(loc="upper left")
    axt.grid(alpha=0.3)

    axh.hist(bg_logits, bins=30, color="tab:gray", alpha=0.8,
             label=f"{n_bg} off-source 1 s segments")
    axh.axvline(ev_logit, color="tab:red", lw=2,
                label=f"GW150914: logit {ev_logit:+.1f} (p = {ev_p:.4f})")
    axh.set_xlabel("CNN score (logit)")
    axh.set_ylabel("count")
    axh.set_title(f"The event vs the measured background — "
                  f"{'above all' if n_above == 0 else f'beaten by {n_above}'} of {n_bg}")
    axh.legend(loc="upper center")
    axh.grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "8_gw150914.png"
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
