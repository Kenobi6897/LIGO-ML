---
tags: [setup, ligo, machine-learning, checklist]
status: todo
created: 2026-07-11
updated: 2026-07-11
---

# Setup — Desktop PC

Written from the **laptop** (`THEOS-LAPTOP`, i7-1255U / Iris Xe) on 2026-07-11, when the PC
wasn't reachable. Everything below runs **on the desktop PC** (Ryzen 7 3700X / RTX 3070 / 16 GB).

The laptop is already fully set up: vault cloned, git remote configured, auto-pull hook live.
Nothing here needs doing twice.

**Why the PC needs more than the laptop:** per [[GW Signal Classifier - Brainstorm]], the PC is
the *training* machine — Stages 1–4. That means `lalsuite` (Linux-only, no Windows wheels), which
means WSL2, which means CUDA passthrough. The laptop only ever needed the vault.

Do Part 1 first — it's five minutes and gets your notes syncing. Parts 2–4 are the long pole and
can wait for a separate sitting.

---

## Part 1 — Vault sync (~5 min)

- [ ] **Clone the vault.** In PowerShell:
      ```powershell
      cd $env:USERPROFILE\Documents
      git clone https://github.com/Kenobi6897/LIGO-ML.git
      ```
      Repo is **private** — expect a GitHub auth prompt. If `gh` is installed, `gh auth login`
      first and the clone goes through without a browser detour.

- [ ] **Open it as a vault.** Obsidian → *Open folder as vault* → `Documents\LIGO-ML`.
      `.obsidian/` is tracked in git, so appearance, enabled plugins, and hotkeys arrive already
      configured. Only `workspace.json` (pane layout) is machine-local by design.

- [ ] **Install the Obsidian Git plugin.** Settings → Community plugins → Browse → "Git".
      This is a GUI-only step; it can't be scripted. Configure:
      - Auto-pull on startup: **on**
      - Auto-commit-and-sync interval: **10 minutes**
      - Push on auto-commit-and-sync: **on**

      Without this you are back to pushing by hand, and the first time you forget, git writes
      `<<<<<<<` conflict markers **directly into a note**. That is the whole failure mode this
      setup exists to prevent.

- [ ] **Verify the Claude auto-pull hook.** `.claude/settings.json` came down with the clone, so
      it's already there. Start Claude Code inside `Documents\LIGO-ML` and confirm it pulls on
      launch. If it doesn't fire, open `/hooks` once — Claude's settings watcher only tracks
      directories that had a settings file when the session began.

- [ ] **Prove the round-trip.** Edit a note on the PC, wait for auto-push, then pull on the
      laptop and confirm it lands. Do this *before* you have work worth losing.

---

## Part 2 — WSL2 + CUDA (the long pole)

> [!danger] The one that will actually bite you
> **NEVER install a Linux NVIDIA driver inside WSL2.** It overwrites the Windows driver stubs and
> breaks the passthrough chain — and it's the natural thing to do, because it's what you'd do on
> real Linux. Windows driver **only**; inside WSL2, the **CUDA toolkit only**.

- [ ] **Install the NVIDIA driver on Windows.** Normal GeForce driver, from Windows. Nothing special.

- [ ] **Install WSL2.** From an *admin* PowerShell, then reboot:
      ```powershell
      wsl --install
      ```

- [ ] **Cap WSL2 memory before anything else.** The PC has 16 GB and WSL2 grabs up to half by
      default, while the dataset is what actually wants RAM. Create `%UserProfile%\.wslconfig`:
      ```ini
      [wsl2]
      memory=10GB
      ```
      Then `wsl --shutdown` to apply. Retrofitting this later is annoying — do it now.

- [ ] **Install the CUDA toolkit inside WSL2** using NVIDIA's **`wsl-ubuntu`** packages, which are
      built to skip the driver: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
      Do not substitute the generic Linux CUDA package — that's the footgun above, wearing a hat.

- [ ] **Confirm passthrough works** before installing anything else:
      ```bash
      nvidia-smi          # should report the RTX 3070
      ls /usr/lib/wsl/lib # should contain libcuda.so.1
      ```
      If `nvidia-smi` fails here, stop and fix it. Everything downstream depends on it.

---

## Part 3 — Python environment (inside WSL2)

- [ ] Create the venv and install the stack:
      ```bash
      python3 -m venv ~/venvs/ligo && source ~/venvs/ligo/bin/activate
      pip install gwpy pycbc numpy scipy matplotlib h5py
      pip install torch --index-url https://download.pytorch.org/whl/cu124
      ```

- [ ] **Confirm PyTorch sees the GPU:**
      ```bash
      python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
      ```
      Expect `True NVIDIA GeForce RTX 3070`. `False` means Part 2 isn't actually done, regardless
      of what `nvidia-smi` said.

- [ ] **Confirm `lalsuite` imports** — this is the entire reason for WSL2, so verify it directly
      rather than assuming pycbc pulled it in cleanly:
      ```bash
      python -c "import lal, lalsimulation; print('lalsuite ok')"
      ```

*(Heavyweight alternative if the pip route fights you: the official IGWN conda distribution ships
gwpy/pycbc/lalsuite/bilby pre-integrated — https://computing.docs.ligo.org/conda/)*

---

## Part 4 — Where the data lives

- [ ] **Datasets stay on the PC. Never commit them.** `.gitignore` already excludes `*.hdf5`,
      `*.h5`, `*.npy`, `*.pt`, `data/`, `checkpoints/`, `runs/`. GitHub hard-rejects files over
      100 MB, and a 1–3 GB HDF5 would bloat the repo permanently — even after you delete it.

- [ ] Keep the dataset **outside the vault** entirely (e.g. `~/ligo-data/` inside WSL2). The vault
      is notes; the repo is notes. Don't blur that line — the `.gitignore` is a safety net, not a
      plan.

The brainstorm note is explicit that the pipeline should **not** be split across the two machines:
you'd shuttle a multi-GB HDF5 between boxes every time a preprocessing decision changes, and those
decisions change constantly, because that's where the leakage bugs live.

---

## Done when

- [ ] Notes edited on either machine show up on the other without manual git commands.
- [ ] `nvidia-smi` reports the 3070 from inside WSL2.
- [ ] `torch.cuda.is_available()` is `True`.
- [ ] `import lal` works.

At that point the PC is ready for **Stage 1** and this note can be archived.

Stage 0 (pull GW150914, whiten, bandpass, see the chirp) needs none of this — it's pure
gwpy/scipy and runs natively on Windows, on the laptop, today.
