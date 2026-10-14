# Set up AITER for Kernel Scout

Set up AITER before discovery or scheduled measurements.
Use a dedicated checkout and virtual environment.
The discovery agent researches installed source with read-only tools.
It cannot install packages or compile kernels.
The controller runs adapters, including their normal JIT compilation, in the configured worker runtime.

## Prepare the environment

1. Check the AMD GPU architecture and ROCm toolchain.
2. Select a ROCm PyTorch build compatible with that toolchain.
3. Build TLX from source for the main Scout runtime.
4. Clone AITER with its pinned submodules.
5. Install AITER in its own virtual environment.

Read the checked revision's [installation instructions](https://github.com/ROCm/aiter#installation) and `setup.py` first.
AITER installation can replace Triton unless `AITER_USE_SYSTEM_TRITON=1` selects an existing installation.
Use that flag when checking AITER with the source-built TLX compiler.
If that compiler lacks required APIs, configure AITER's supported compiler in the separate environment.
Retain the same PyTorch build and harness input contract across both runtimes.

The example below reuses an existing ROCm PyTorch environment through an explicit import path.
It does not change that environment.
A self-contained environment can instead install the selected ROCm PyTorch build locally.

```bash
export SCOUT_AITER_SRC=/path/to/aiter
export SCOUT_AITER_VENV=/path/to/venv/aiter.venv
git clone --recursive https://github.com/ROCm/aiter.git "$SCOUT_AITER_SRC"
python3 -m venv "$SCOUT_AITER_VENV"

# Install build/runtime dependencies in the new environment before borrowing other packages.
"$SCOUT_AITER_VENV/bin/python" -m pip install setuptools wheel setuptools_scm
"$SCOUT_AITER_VENV/bin/python" -m pip install -r "$SCOUT_AITER_SRC/requirements.txt"

export PYTHONPATH=/path/to/triton/python:$SCOUT_AITER_SRC:/path/to/rocm-torch/site-packages
export PATH=$SCOUT_AITER_VENV/bin:$PATH
export MAX_JOBS=8
export GPU_ARCHS=gfx950
AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  "$SCOUT_AITER_VENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$SCOUT_AITER_SRC"
```

Use the actual architecture for `GPU_ARCHS`.
Choose build parallelism that fits the host's available memory.
`PREBUILD_KERNELS=0` leaves native operators for JIT compilation.
The first public attention call can need substantially longer than a warm measurement.

Put the AITER checkout itself on `PYTHONPATH`.
Its parent directory can resolve `aiter` as an empty namespace when an editable installation is present.
Check that `aiter.__file__` names the intended `aiter/__init__.py`.

Do not set `AITER_TRITON_ONLY=1` for a complete backend comparison.
That mode removes public CK/HIP operators from the top-level package.
Likewise, `ENABLE_CK=0` changes attention dispatch.
Keep these choices explicit and retain them with the results.

## Check imports and configure Scout

Choose an idle physical GPU.
Scout checks its PCI identity and competing processes before measurements.
The `--gpu` option uses AMD SMI numbering. Scout maps that index to an HSA UUID before starting workers.
For the manual import check, use the matching GPU UUID from `rocminfo`.

```bash
ROCR_VISIBLE_DEVICES=GPU-<uuid-from-rocminfo> HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  "$SCOUT_AITER_VENV/bin/python" -c \
  'import aiter, torch, triton; print(aiter.__file__, torch.__version__, triton.__file__); print(aiter.flash_attn_varlen_func)'
```

Save a runtime override outside the checkout:

```json
{
  "aiter": {
    "python": "/path/to/venv/aiter.venv/bin/python",
    "pythonpath": "/path/to/aiter-triton/python:/path/to/aiter:/path/to/rocm-torch/site-packages"
  }
}
```

Scout prepends the worker interpreter's directory to `PATH`.
This makes the environment's `ninja` executable available for AITER JIT builds.
A Python `ninja` import alone does not establish executable availability.

From the Kernel Scout directory, run a small weekly comparison:

```bash
/path/to/venv/tlx.venv/bin/python appctl.py run --mode weekly --execute \
  --targets targets/amd/gfx950/varlen_f16_fa.json --case one_x_1000 \
  --gpu 2 --runtimes /path/to/runtimes.json --state /path/to/scout-state \
  --worker-timeout 1800
```

Set the main runtime's source-first `PYTHONPATH` before this command.
Use the same override and environment for nightly reuse.
Keep AITER's compiler checkout pinned during the week.
Use a separate source-built compiler checkout when the main TLX checkout changes nightly.
The worker records the compiler it actually imports.
Check AITER's candidate measurements, input fingerprints, and selected confirmation.
Run its report reproduction command before accepting the setup.
Repeat with dense attention, GEMM, and the full packed-attention target.

The default AITER baseline uses public operators and their own dispatch.
It does not invoke an LLM, even when other backends use agent discovery.
Add `--aiter-search` for alternative implementations.
Also select `--provider claude` or `codex` for source discovery in that mode.
Keep the same AITER mode for weekly and nightly runs.
Scout gives the agent the installed AITER checkout as its research workspace.
The prompt includes the runtime identity, source pin, and measured seed feedback.
Use [backend instructions](instructions.md) for the search sequence.
Use a fresh state directory when repeating an existing weekly run.

## Cold builds, tuning, and failures

- A build failure is a failed candidate, even when a different AITER path succeeds.
- Missing packages are unavailable. Install them before another discovery run.
- Missing `ninja` usually indicates an incomplete installation or incorrect worker `PATH`.
- A missing public API can indicate namespace shadowing or a Triton-only installation.
- Inspect `worker.log` for native build diagnostics and `search.json` for all evaluated paths.
- Warm each needed implementation through correctness checks before comparing steady-state timing.

Record the AITER Git SHA, submodule revisions, package version, compiler, and build environment.
Keep architecture-specific configuration files and their revisions with performance evidence.
Scout hashes source and tuning files while excluding ignored generated JIT output.
Changing `AITER_CONFIG_*` file contents changes runtime identity.
Preserve the source checkout, configuration, environment, and run artifacts for pinned reproduction.

The checked setup uses AITER `3b82abe81ae36b2cfefcf5a8280e2edc15bbf047`, Python 3.12, ROCm PyTorch 2.13,
and the source-built TLX compiler on gfx950.
Recheck installation flags and API signatures when updating AITER.
