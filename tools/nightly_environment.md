# Explicit nightly environment preparation

This helper prepares two separate disposable containers. It never starts a
model server, runs pytest, submits requests, or uses an NPU for validation.
Run it from a frozen vllm-ascend checkout; its `--vllm-sha` must exactly equal
that checkout's `.github/vllm-main-verified.commit`.

```bash
python3 -m tools.nightly_environment --role server \
  --vllm-sha <40-character-frozen-SHA> --dep-dir /opt/nightly/server

python3 -m tools.nightly_environment --role client \
  --vllm-sha <40-character-frozen-SHA> --dep-dir /opt/nightly/client
```

Each dependency directory must be new. Successful setup creates `activate.sh`
there. Each subsequent job must source its role's activation script before
calling `python`, `vllm`, or `ais_bench`. Use separate task output directories
for the CLI's generated configuration, dataset copies, and result files.

## Server: limited baseline-image policy

The server preserves the existing image's Torch, Torch NPU, TorchVision,
TorchAudio, NumPy, OpenCV, Triton Ascend, SciPy, and Numba versions. It clones
the frozen vLLM and Ascend sources into the dependency directory and installs
them with `--no-deps --no-build-isolation`. Ascend custom kernels are built
normally; no old kernel binary is relabelled as a new build. Installed CANN
environment scripts supply build and runtime paths. The installed
`/usr/local/Ascend/nnal/atb/set_env.sh`, when present, is also loaded and
`/usr/local/lib` is added to the process's library path. Any vendor script
failure aborts setup or activation.

FastAPI, Starlette and setuptools bounds are read from the selected vLLM
source, including Python version markers. They may be updated while the
protected stack is constrained to its original versions. Audio and development
extras are not installed.

**This is not a fully consistent pip dependency solution.** In the measured
baseline, Triton Ascend 3.2.2 requires NumPy 1.26.4, while the selected vLLM's
OpenCV >=4.13 requirement implies NumPy >=2. The existing OpenCV 4.11 / NumPy
1.26 ABI stack is deliberately retained for this text-only case. Existing
image-level pip-check failures are recorded in full, never suppressed or
rewritten. The helper rejects any new failure, altered actual dependency
version in a failure, or change to a protected package. The owner package
version alone is normalized when comparing a frozen editable SHA build with
the base image. `environment-report.json` explicitly records this limitation.

Imports are checked after installation, but only the separately supervised
`vllm serve` and original performance case can establish runtime compatibility.
Do not advertise this image policy as validation of other model types or of
the whole dependency graph. A failed installation requires a new environment;
there is no automatic rollback or hidden retry.

## Client: isolated API dependencies

The client clones AISBench commit
`0da56eadb2ac85c31c2540f4f5b69af3ec5717a5` from the existing source repository
specified by `--benchmark-source` (default
`/vllm-workspace/vllm-ascend/benchmark`). It never modifies that source.

A venv without system site packages installs the official CPU Torch 2.10.0
wheel and the fixed AISBench `[api]` dependency set plus PyYAML. The client
constraints retain NumPy 1.26.4, pandas 2.2.3, OpenCV 4.11.0.86, and Transformers
5.14.1. No vLLM, Ascend Python plugin, or audio extras are installed. CANN's
inherited `PYTHONPATH` is removed from client install/run processes. Client
`pip check` must pass, and the native `ais_bench --help` import is verified.

Python is `<dep-dir>/venv/bin/python`; AISBench is
`<dep-dir>/venv/bin/ais_bench`, with private sources in `<dep-dir>/benchmark`.
The official CPU index provides the Python 3.12 aarch64 Torch 2.10.0+cpu wheel:
https://download.pytorch.org/whl/cpu/torch/ .

## Evidence and bounds

The helper keeps clone, build, dependency install, import, and pip-check logs
inside its private dependency directory. `environment-report.json` contains
fixed source identities and dependency-audit results. Commands have bounded
timeouts; the workflow supervisor must additionally enforce its environment
deadline. Building custom kernels and downloading a clean client environment
can approach or exceed a 3600-second environment budget on a slow node/network;
an actual deadline failure must remain a failure and be reviewed before retry.
Git may use a task-local, verified source cache, but neither global Git config
nor another user's Python environment is modified by this helper.

Run the isolated interface checks without an NPU runtime:

```bash
python -m unittest tests.ut.tools.test_nightly_environment tests.ut.tools.test_nightly_cli
```
