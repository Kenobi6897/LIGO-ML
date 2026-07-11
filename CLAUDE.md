
# CLAUDE.md

Guidance for Claude Code when working in this vault.

## What this is

An **Obsidian vault**, not a codebase. It holds the notes and planning for a LIGO
gravitational-wave signal classification project. Notes are plain Markdown; treat them
as the source of truth for project decisions.

The vault is synced between **two machines via git** (laptop + desktop PC). See "Two-machine
workflow" below — it changes how you should start and end a session.

## Layout

- `GW Signal Classifier - Brainstorm.md` — the master planning note. Environment decisions,
  staged plan (Stage 0–4), known failure modes, reference papers. Read this first.
- `.obsidian/` — Obsidian config. Tracked in git so both machines share settings.

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
