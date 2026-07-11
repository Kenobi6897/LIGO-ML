---
tags: [setup, ligo, machine-learning, checklist]
status: in-progress
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

> [!tip]- 👉 RESUME HERE — written 2026-07-11 on the PC, immediately before the `wsl --install` reboot
> **Where we actually are: Part 1 is done. Part 2 has not started. Parts 3–4 untouched.**
> [[Stage 1]] is **gated shut** — its gate was verified, not assumed, and it **fails**: no WSL2.
>
> **The single next action** — from an **Administrator** PowerShell, then reboot:
> ```powershell
> wsl --install
> ```
> Ubuntu prompts for a Linux username + password on first launch. `.wslconfig` (10 GB) is
> **already written**, so the cap applies from first boot — no `wsl --shutdown` needed.
>
> **After the reboot, the order is:** CUDA `wsl-ubuntu` toolkit → `nvidia-smi` inside WSL →
> Part 3 venv → `torch.cuda.is_available()` → `import lal`. When `import lal` works, the
> [[Stage 1]] gate opens. Nothing before that point can be skipped.
>
> **Does WSL2 endanger the Windows boot? No.** It is not a dual-boot: no bootloader entry, no
> partition, no boot menu. Ubuntu is a VM in a file, started on demand. The reboot only exists to
> activate two Windows optional features (WSL + Virtual Machine Platform). *One real caveat:* those
> features enable the Hyper-V layer, which can upset older VirtualBox/VMware and some kernel-level
> anti-cheat games. If you run neither, you'll notice nothing.
>
> **Still unverified (deliberately left unticked, don't assume them):**
> - The **Claude auto-pull hook** — the session that would have proved it started *outside* the
>   vault (`C:\Users\locke`), so the vault's `SessionStart` hook never fired. No evidence either way.
> - Everything in Parts 2–4.
>
> **Don't "fix" [[Stage 0]]'s `C:\Users\tmloc\...` paths.** They look stale but are correct: Stage 0
> is a log of work done on the *laptop*, where that genuinely was the username. Only [[Stage 1]]'s
> path needed correcting to `locke`, because it describes *this* PC. (Already done.)

> [!note] Progress — 2026-07-11, run **on the PC** (`DESKTOP-P6POAN4`)
> The PC was not the clean slate this note assumed. Corrections, so the next reader isn't misled:
> - **The vault was already cloned** here, clean and tracking `origin/main`. Someone started Part 1
>   and stopped. Clone step was a no-op.
> - **Obsidian itself was *not* installed** — this note never had a step for it, because the laptop
>   already had it. Installed 1.12.7 via `winget install --id Obsidian.Obsidian -e`.
>   Do **not** pass `--scope user`: the installer crashes with an access violation (`0xC0000005`).
>   Plain flags work.
> - **`gh` is not needed.** The note suggests it for the private-repo auth prompt, but Windows
>   Credential Manager already had the credential — `git ls-remote` authenticated fine.
> - **Part 2's NVIDIA-driver step is already done**: driver **596.49**, RTX 3070 detected. Part 2
>   really starts at `wsl --install`.
> - Still open: everything in Part 1 that needs the Obsidian GUI, and all of Parts 2–4.
>
> *Superseded later the same day — Part 1 finished and auto-sync proven. See the RESUME callout above.*

> [!success] Round-trip — **PC → laptop confirmed**, 2026-07-11
> Read back on the laptop (`THEOS-LAPTOP`): commit `1ee4a5d` arrived clean, fast-forward, no
> conflict markers. **The PC → laptop direction works.**
>
> This callout was written *on the laptop* and pushed back. **If you are reading it on the PC,
> laptop → PC works too, and the round trip is closed** — go tick the box in Part 1.
>
> Two things the round-trip test surfaced:
> - **Both machines committed via Claude, not via Obsidian Git.** So this proves *git* works, not
>   that the **auto**-sync works. The plugin settings are still unverified (see Part 1) — until
>   you've watched Obsidian push on its own, you are still syncing by hand.
> - **The two machines have different git identities.** The PC commits as `T. M. Locke`, the laptop
>   as `Kenobi6897` — same email, so GitHub attributes both to you and nothing is broken. Worth
>   knowing before you wonder who the second contributor is.

---

## Part 1 — Vault sync (~5 min)

- [x] **Clone the vault.** In PowerShell:
      ```powershell
      cd $env:USERPROFILE\Documents
      git clone https://github.com/Kenobi6897/LIGO-ML.git
      ```
      Repo is **private** — expect a GitHub auth prompt. If `gh` is installed, `gh auth login`
      first and the clone goes through without a browser detour.
      *Done — was already cloned at `Documents\LIGO-ML`. Credential Manager had the auth; no `gh`.*

- [x] **Install Obsidian.** Not in the original checklist — the laptop already had it, the PC didn't.
      ```powershell
      winget install --id Obsidian.Obsidian --exact
      ```
      Lands at `%LOCALAPPDATA%\Programs\Obsidian\Obsidian.exe`. **Don't add `--scope user`** — the
      installer dies with an access violation (`0xC0000005`) if you do.

- [x] **Open it as a vault.** Obsidian → *Open folder as vault* → `Documents\LIGO-ML`.
      `.obsidian/` is tracked in git, so appearance, enabled plugins, and hotkeys arrive already
      configured. Only `workspace.json` (pane layout) is machine-local by design.
      *Done — Obsidian running on the PC, `workspace.json` created.*

- [x] **Enable the Obsidian Git plugin.** It is *already installed* — the plugin's code is
      tracked in this repo and `community-plugins.json` already lists it as enabled, so it comes
      down with the clone. On first open, Obsidian will likely ask you to turn off **Restricted
      Mode** before it will load community plugins. Do that; no download needed.

- [x] **Set the plugin's options by hand.** These do *not* sync: `.obsidian/plugins/*/data.json`
      is deliberately gitignored, because obsidian-git can store a username/password in that file
      and it must never reach GitHub. Match the laptop (Settings → Git):
      - Auto commit-and-sync interval: **10** minutes
      - Auto pull on startup: **on**
      - Pull before push: **on**
      - Push on commit-and-sync: **on** (i.e. leave "disable push" off)

      Skipping this leaves you pushing by hand, and the first time you forget, git writes
      `<<<<<<<` conflict markers **directly into a note**. That is the whole failure mode this
      setup exists to prevent.

      *Pre-staged on the PC:* `data.json` was written directly with those four values
      (`autoSaveInterval: 10`, `autoPullOnBoot`, `pullBeforePush`, `disablePush: false`) — the file
      is gitignored, so this had to be done per-machine anyway. **Still verify them in the UI**: the
      key names were inferred, not read off a running plugin, so if Settings → Git shows defaults
      instead, just set the four by hand as originally written.

- [ ] **Verify the Claude auto-pull hook.** `.claude/settings.json` came down with the clone, so
      it's already there. Start Claude Code inside `Documents\LIGO-ML` and confirm it pulls on
      launch. If it doesn't fire, open `/hooks` once — Claude's settings watcher only tracks
      directories that had a settings file when the session began.
      *File confirmed present on the PC; not yet observed firing, since this session started outside
      the vault.*

- [x] **Prove the round-trip.** Edit a note on the PC, wait for auto-push, then pull on the
      laptop and confirm it lands. Do this *before* you have work worth losing.
      - [x] **PC → laptop** — `1ee4a5d` pulled clean on the laptop. ✅
      - [x] **Laptop → PC** — callout arrived on the PC clean (`3ee5dab`, fast-forward, no conflict
            markers). ✅ Both manual directions now proven.
      - [x] **Auto-sync (the one that matters)** — ✅ **passed, 2026-07-11.** The edit ticking the
            three boxes above was left deliberately uncommitted; obsidian-git picked it up on its
            own interval and landed it as commit `69d69ae`, named **`vault backup: 2026-07-11
            21:30:43`** — the plugin's own default message format, not a hand-written one. It
            pushed too: `main` is level with `origin/main`. **Nobody touched git.**
            This also retroactively validates the pre-staged `data.json`: those four keys were
            *inferred*, not read off a running plugin, and the plugin evidently read them. They
            are correct.

---

## Part 2 — WSL2 + CUDA (the long pole)

> [!danger] The one that will actually bite you
> **NEVER install a Linux NVIDIA driver inside WSL2.** It overwrites the Windows driver stubs and
> breaks the passthrough chain — and it's the natural thing to do, because it's what you'd do on
> real Linux. Windows driver **only**; inside WSL2, the **CUDA toolkit only**.

- [x] **Install the NVIDIA driver on Windows.** Normal GeForce driver, from Windows. Nothing special.
      *Done — `nvidia-smi` on Windows reports RTX 3070, driver **596.49**.*

- [ ] **Install WSL2.** From an *admin* PowerShell, then reboot:
      ```powershell
      wsl --install
      ```
      Verified absent on the PC, 2026-07-11: `wsl --list --verbose` → *"The Windows Subsystem for
      Linux is not installed."* **This is the sole blocker on all of [[Stage 1]].**
      Ubuntu asks for a Linux username + password on first launch — unrelated to your Windows login.
      *Not a dual-boot:* no bootloader entry, no partition, no boot menu. The reboot only activates
      the WSL and Virtual Machine Platform Windows features.

- [x] **Cap WSL2 memory before anything else.** The PC has 16 GB and WSL2 grabs up to half by
      default, while the dataset is what actually wants RAM. Create `%UserProfile%\.wslconfig`:
      ```ini
      [wsl2]
      memory=10GB
      ```
      Then `wsl --shutdown` to apply. Retrofitting this later is annoying — do it now.
      *Done 2026-07-11 — written to `C:\Users\locke\.wslconfig` **before** `wsl --install`, so it
      applies from WSL's first boot and no `wsl --shutdown` is needed. (Confirmed 16 GB physical.)*

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
      pip install gwpy pycbc numpy scipy matplotlib h5py scikit-learn tqdm
      pip install torch --index-url https://download.pytorch.org/whl/cu124
      ```
      (`scikit-learn` for ROC/AUC — [[Stage 1]] evaluates on ROC, *not* accuracy, so it isn't
      optional. `tqdm` because dataset generation is a long loop and you want a progress bar.)

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

- [x] Notes edited on either machine show up on the other without manual git commands. ✅ 2026-07-11
- [ ] `nvidia-smi` reports the 3070 from inside WSL2.
- [ ] `torch.cuda.is_available()` is `True`.
- [ ] `import lal` works.

At that point the PC is ready for **Stage 1** and this note can be archived.

Stage 0 (pull GW150914, whiten, bandpass, see the chirp) needs none of this — it's pure
gwpy/scipy and runs natively on Windows, on the laptop, today.
