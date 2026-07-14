
# CLAUDE.md

Guidance for Claude Code when working in this vault.

## What this is

An **Obsidian vault that also carries the project's code.** It is mostly notes — planning for a
LIGO gravitational-wave signal classification project — and the notes are the source of truth for
project decisions. But as of 2026-07-12 the stage code lives here too, **inside the stage folders
themselves**: each stage is one folder holding its note, its scripts and its plots.

The vault is synced between **two machines via git** (laptop + desktop PC). See "Two-machine
workflow" below — it changes how you should start and end a session.

## Layout

- `1. Stages/` — the plan, the per-stage logs, **and the code.**
  - `GW Signal Classifier - Brainstorm.md` — the master planning note. Environment decisions,
    staged plan (Stage 0–4), known failure modes, reference papers. **Read this first.**
  - `Stage 0/`, `Stage 1/` — **one folder per stage, holding everything about that stage:** the
    write-up (`Stage N.md`), the scripts, `requirements-stageN.txt`, a gitignored `.venv/`, and
    `outputs/` with the committed plots — which the note embeds with `![[plot.png]]`. Each stage
    folder is self-contained; Stage 0 is complete.
    - `Stage 0/` is pure gwpy/scipy and **runs natively on Windows.** Stages 1+ need WSL2.
- `2. Explained/` — standalone explainers (Q-transform, matched filtering, 1D-vs-2D).
- `3. Setup/` — machine setup checklists, and this file.
- `.obsidian/` — Obsidian config. Tracked in git so both machines share settings.

## Code conventions

- **Plots are deliverables and belong in git.** They're small. *Data* does not — datasets,
  checkpoints and HDF5 stay on the PC, and `.gitignore` enforces it.
- **Stage 0 pins `igwn-segments==2.0.0`** in its requirements. This is not cruft: without the pin,
  `pip install gwpy` fails on native Windows trying to compile a C extension. Don't unpin it.
- Each stage script should be runnable end-to-end from its own directory with no arguments, and
  should overwrite its plots in place so a re-run shows up as a clean diff.

## Obsidian conventions

- **Frontmatter**: notes use YAML frontmatter with `tags`, `status`, `created`, `updated`.
  Preserve it when editing, and bump `updated` (ISO `YYYY-MM-DD`) when you make a real change.
- **Wikilinks**: `[[Note Name]]` and `[[#Heading]]`, not Markdown links, for internal
  references. Obsidian resolves these by filename — renaming a note breaks inbound links.
- **Bases** (`.base` files): Obsidian's database views. None exist yet. If you add one, it's
  a YAML file defining views over note frontmatter properties — which is why consistent
  frontmatter keys matter more than they look like they do.
- **Task checkboxes**: `- [ ]` / `- [x]` are live project state, not decoration. Check them
  off when the corresponding work is genuinely done.

## Project context (from the brainstorm note — don't re-derive)

- **Two machines, deliberately split**: laptop = Stage 0 + all editing/reading. Desktop PC
  (RTX 3070, WSL2) = Stages 1–4, training, and *all* data. The note explicitly says **do not
  split the pipeline across machines** — code in git, data on the PC only.
- **Never suggest installing a Linux NVIDIA driver inside WSL2.** It breaks CUDA passthrough.
  Windows driver + `wsl-ubuntu` CUDA toolkit only.
- **`lalsuite` is Linux-only** (no Windows wheels), which is why Stages 1–4 need WSL2. Stage 0
  is pure `gwpy`/`scipy` and runs natively on Windows.
- **The expected failure mode is data leakage in preprocessing**, not modeling error. If a
  result looks suspiciously good, suspect leakage first. The note lists the three mechanisms.
- **The CNN is not supposed to beat matched filtering** on known waveforms in Gaussian noise —
  that's a theorem. If it does, it's a bug.

## Two-machine workflow

A `SessionStart` hook runs `git pull --rebase --autostash` automatically, so the vault should
be current when you start. But:

- **Before ending a session with edits, commit and push.** Uncommitted notes on one machine
  are invisible to the other, and the next session there will diverge.
- **Conflicts land as `<<<<<<<` markers inside the Markdown.** If you see them in a note, that
  is a merge conflict, not content — stop and surface it rather than reasoning over it.
- Don't commit datasets or model checkpoints. `.gitignore` covers the usual extensions, but
  data belongs on the PC only.
