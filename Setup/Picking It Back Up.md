---
tags: [setup, ligo, checklist]
status: active
created: 2026-10-29
updated: 2026-10-29
---

# Picking it back up

The re-entry guide for coming back after a break. **The research is done.** Stages 0–5 are
complete and written up, and nothing is half-finished. Whatever you do next is either
presentation work or a new follow-on stage.

## 1. Two-minute recap

- **Question:** on real, glitch-ridden LIGO noise, at equal false-alarm probability, can a small
  CNN close the gap that the Gaussian assumption opens in matched filtering, and at what cost?
- **Answer, honours split.** The clip-fixed CNN (arm D+) beats the χ²-vetoed matched filter (arm F)
  at glitch rejection (0.058 ± 0.022 vs 0.100 @ FAP 10⁻²). F still leads on faint-signal detection
  (0.850 vs 0.801 at SNR 6–8 @ 10⁻³). Nothing beats the oracle (+0.000). The CNN has ~20× lower
  latency and ~3,700× higher throughput.
- **Boundary of the claim:** fed raw strain the CNN is a coin flip (AUC 0.4997), so the
  conditioning pipeline, mainly the bandpass, does load-bearing work.
- **Read in this order:** `README.md` (the whole story) → [[Stage 5]] §4b and §7 (the final verdict
  and the clip bug) → [[GW Signal Classifier - Brainstorm]] (the original plan and follow-ons).

## 2. Get both machines in sync

- [ ] **Laptop** — open the vault in Obsidian; obsidian-git pulls on startup. Then check it's clean:
      ```powershell
      cd $env:USERPROFILE\Documents\LIGO-ML
      git status -sb      # want: "## main...origin/main" and nothing else
      ```
      If you see `conflict-files-obsidian-git.md` at the vault root or `UU` lines, a merge is
      stuck. Back the folder up before touching it. (This happened once already: the laptop sat
      mid-merge from July to October 2026 and was reset to `origin/main` on 2026-10-29. The backup
      is at `Documents\LIGO-ML-backup-2026-10-29`; delete it once you're happy.)
- [ ] **Desktop PC** — same check on the PC's `Documents\LIGO-ML`. The PC did most of the commits,
      so it is probably clean, but check before editing.
- [ ] **`Explained/` notes** are gitignored, so they **don't sync**. The laptop copy is from
      2026-07-13. If you edited them on the PC later, copy them over by hand.

## 3. Check the PC environment still works

All of this runs **inside WSL2** on the desktop PC. Nothing should need reinstalling.
```bash
source ~/venvs/ligo/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   → True NVIDIA GeForce RTX 3070
python -c "import lal, lalsimulation, pycbc; print('lalsuite ok', pycbc.__version__)"
ls -lh ~/ligo-data/        # datasets + checkpoints: stage1.h5 … stage5_s2.h5, *_cnn.pt
```
If `torch.cuda` says `False` after a Windows driver update, run `wsl --shutdown` from
PowerShell and try again. **Never install a Linux NVIDIA driver inside WSL2** ([[Setup - Desktop PC]]).

## 4. Prove it still runs

Cheapest first:

- [ ] **Stage 0, on the laptop** (native Windows, ~1 min). Rebuilds the GW150914 plots, so a clean
      `git diff` afterwards means nothing changed.
      ```powershell
      cd "Stages\Stage 0"
      python -m venv .venv
      .\.venv\Scripts\pip install -r requirements-stage0.txt
      .\.venv\Scripts\python stage0_gw150914.py
      ```
- [ ] **Stage 5's final verdict, on the PC** (reuses saved datasets and checkpoints, so nothing is
      re-fetched). Run it from the stage folder so the relative imports resolve:
      ```bash
      cd "/mnt/c/Users/locke/Documents/LIGO-ML/Stages/Stage 5"
      python stage5_rerun_stage4.py     # regenerates outputs/7_stage4_rerun.png
      ```
      Each stage note lists its scripts in run order. The `*_check.py` files are not optional.

## 5. Open threads, if you want a Stage 6

None of these are needed for the project to count as finished. Ordered by value per hour:

- [ ] **H1 + L1 as a 2-channel input.** Coincidence is most of real detection power, and
      [[Stage 0]]'s +6.9 ms overlay is the argument. This is the biggest scientific gap.
- [ ] **Off-bank signals.** Inject waveforms the 62-template bank doesn't cover (spinning, higher
      mass) to test the "template coverage" assumption that Stage 4 deliberately left untested.
- [ ] **More background.** 1.48 h of test noise caps FA/h claims at ~0.7/h, and the 10⁻³ rows
      ride on ~10 counts. More O3a strain would tighten every deep-threshold number.
- [ ] **Audit the MF side as hard as the CNN side.** Stage 5 found a bug that favoured MF. A fair
      referee would rebuild F from scratch too.
- [ ] **2D Q-transform arm** (deferred since [[Stage 3]]) and **chirp-mass regression** (Brainstorm,
      out of scope) are both possible, but they are new projects more than next stages.
