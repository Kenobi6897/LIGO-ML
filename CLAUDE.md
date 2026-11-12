
# CLAUDE.md

Guidance for Claude Code when working in this vault.

## What this is

An **Obsidian vault that also carries the project's code.** It is a completed, six-stage LIGO
gravitational-wave project (a 1D CNN benchmarked against matched filtering at equal false-alarm
rate), and the notes are the source of truth for project decisions. Each stage is one folder
holding its note, its scripts and its plots. `README.md` is the public lab report.

**Status (2026-10-29): research complete, Stages 0–5 done.** Anything new is a follow-on, not
unfinished work. To resume after a break, read `Setup/Picking It Back Up.md` first.

The vault is synced between **two machines via git** (laptop + desktop PC). See "Two-machine
workflow" below — it changes how you should start and end a session.

## Layout

- `README.md` — the lab report: abstract, methods, results for all six stages, threats to validity.
- `Stages/` — the plan, the per-stage logs, **and the code.**
  - `GW Signal Classifier - Brainstorm.md` — the master planning note. Environment decisions,
    staged plan, known failure modes, reference papers.
  - `Stage 0/` … `Stage 5/` — **one folder per stage, holding everything about that stage:** the
    write-up (`Stage N.md`), the scripts, `requirements-stageN.txt`, and `outputs/` with the
    committed plots — which the note embeds with `![[plot.png]]`. Later stages import earlier
    stages' modules via `sys.path` (`parents[1] / "Stage 1"`), so folder names are load-bearing.
    - `Stage 0/` is pure gwpy/scipy and **runs natively on Windows.** Stages 1+ need WSL2.
- `Setup/` — machine setup (`Setup - Desktop PC.md`) and the re-entry guide (`Picking It Back Up.md`).
- `Explained/` — personal physics explainers. **Gitignored on purpose**: they exist only on disk,
  never in the public repo. Don't commit them and don't link to them from the README.
- `.obsidian/` — Obsidian config, including the obsidian-git plugin's code. Tracked in git so
  both machines share settings; removing it from git would delete it on the other machine.

## Code conventions

- **Plots are deliverables and belong in git.** They're small. *Data* does not — datasets,
  checkpoints and HDF5 live on the PC at `~/ligo-data/` (inside WSL2), and `.gitignore` enforces it.
- **Stage 0 pins `igwn-segments==2.0.0`** in its requirements. This is not cruft: without the pin,
  `pip install gwpy` fails on native Windows trying to compile a C extension. Don't unpin it.
- Each stage script should be runnable end-to-end from its own directory with no arguments, and
  should overwrite its plots in place so a re-run shows up as a clean diff.
- Every pipeline step has a paired `*_check.py` that compares against something independently
  known and **can fail**. New steps get one too.

## Obsidian conventions

- **Frontmatter**: notes use YAML frontmatter with `tags`, `status`, `created`, `updated`.
  Preserve it when editing, and bump `updated` (ISO `YYYY-MM-DD`) when you make a real change.
- **Wikilinks**: `[[Note Name]]` and `[[#Heading]]`, not Markdown links, for internal
  references. Obsidian resolves these by filename — renaming a note breaks inbound links.
  (The README is the exception: it is read on GitHub, so it uses relative Markdown links.)
- **Task checkboxes**: `- [ ]` / `- [x]` are live project state, not decoration. Check them
  off when the corresponding work is genuinely done.

## Project context (from the brainstorm note — don't re-derive)

- **Two machines, deliberately split**: laptop = Stage 0 + all editing/reading. Desktop PC
  (RTX 3070, WSL2) = Stages 1–5, training, and *all* data. **Do not split the pipeline across
  machines** — code in git, data on the PC only.
- **Never suggest installing a Linux NVIDIA driver inside WSL2.** It breaks CUDA passthrough.
  No CUDA toolkit is needed either — torch's wheels bundle their own runtime.
- **`lalsuite` is Linux-only** (no Windows wheels), which is why Stages 1+ need WSL2. The venv
  is Python 3.13 (`~/venvs/ligo`) because pycbc has no 3.14 wheel.
- **The expected failure mode is a silent wrong answer**, usually preprocessing leakage. If a
  result looks suspiciously good, suspect leakage first.
- **The CNN is not supposed to beat matched filtering** on known waveforms in Gaussian noise —
  that's a theorem. If it beats the oracle, it's a bug.

## Two-machine workflow

A `SessionStart` hook runs `git pull --rebase --autostash` automatically, so the vault should
be current when you start. But:

- **Before ending a session with edits, commit and push.** Uncommitted notes on one machine
  are invisible to the other, and the next session there will diverge.
- **Conflicts land as `<<<<<<<` markers inside the Markdown**, or as an obsidian-git
  `conflict-files-obsidian-git.md` at the vault root. Either means a merge is in progress —
  stop and surface it rather than reasoning over it.
- Don't commit datasets or model checkpoints. `.gitignore` covers the usual extensions, but
  data belongs on the PC only.
