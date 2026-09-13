# Explicit nightly preparation and verification

`python -m tools.nightly_cli` reads a named performance benchmark from the
existing single-node YAML. It does not import pytest, instantiate
`AisbenchRunner`, download a model/dataset, start a server, or launch AISBench.
The process supervisor remains responsible for service readiness, execution,
cancellation, deadlines, and collecting output.

The native `AisbenchRunner` retains its existing lifecycle and now uses the same
configuration-rendering and performance-check functions as this CLI. Output
throughput, optional input throughput, and optional TPOT checks preserve the
original formulas and raise `AssertionError` on failure, including under
`python -O`.

## Generate the server command

Run from the vllm-ascend repository checkout. Use a fresh output directory for
each execution; generated files are never overwritten.

```bash
python -m tools.nightly_cli server-command \
  --config tests/e2e/nightly/single_node/models/configs/Qwen3-30B-A3B-W8A8.yaml \
  --case Qwen3-30B-A3B-W8A8-TP1 --benchmark perf \
  --model-path /models/local --host 0.0.0.0 --port 18123 \
  --output-dir /tmp/nightly/run-unique/server
```

This writes `server.json` and `server.sh`. Tensor parallelism, model length,
batching options, and environment variables come from the selected YAML.
The supplied host, port, and local model path override runtime locations;
`--served-model-name` uses the YAML model name to match the benchmark requests.
Only declared YAML environment references are substituted; unresolved references
are rejected. EPD and KV-pool lifecycle configurations are not supported here.

The supervisor may explicitly run `bash /tmp/nightly/run-unique/server/server.sh`.
Its final command is `exec vllm serve ...`, with every argument shell-quoted.
The helper itself never executes this generated file.

## Prepare the direct AISBench command

```bash
python -m tools.nightly_cli prepare \
  --config tests/e2e/nightly/single_node/models/configs/Qwen3-30B-A3B-W8A8.yaml \
  --case Qwen3-30B-A3B-W8A8-TP1 --benchmark perf \
  --benchmark-home /opt/benchmark \
  --model-path /models/local --dataset-path /datasets/local \
  --host 192.0.2.8 --port 18123 \
  --output-dir /tmp/nightly/run-unique/client
```

This reads the installed AISBench request/dataset templates and writes a complete
`benchmark.py`, `benchmark.sh`, and `manifest.json` outside `benchmark-home`.
The templates are not modified. The generated configuration includes `models`,
`datasets`, and the fixed checkout's official `summarizers/perf/default_perf.py`
configuration, preserving its calculator and statistics along with dataset
pre/postprocessors. The native performance workflow supplies its inference
defaults; preparation does not instantiate or execute that workflow.
The generated command is equivalent to:

```bash
ais_bench /tmp/nightly/run-unique/client/benchmark.py \
  --mode perf --num-prompts 180 \
  --work-dir /tmp/nightly/run-unique/client/results --debug
```

To retain the current run's log, keep the benchmark in the foreground and
preserve its exit status through the logging pipeline:

```bash
set -euo pipefail
bash /tmp/nightly/run-unique/client/benchmark.sh 2>&1 \
  | tee /tmp/nightly/run-unique/client/benchmark.log
```

The request count and model configuration values come from YAML. Default warmup
behavior is retained. A directory dataset mapping is passed through. For a
single GSM8K JSONL file, the helper copies identical bytes to private
`dataset/train.jsonl` and `dataset/test.jsonl` files for the GSM8K directory
loader, recording the input SHA256; it does not edit the original mapping.
Single-file adaptation for other dataset types is rejected.

## Verify the exact result files

AISBench adds a timestamp below `--work-dir`. Use the **current execution's**
`Performance Result files located in ...` output to identify the result
directory. Do not select an arbitrary latest result from a reused directory.
The manifest records the expected JSON/CSV basenames.

The optional `locate-results` action validates a unique directory from this
log, confined to the manifest's private work directory. It also checks the
source YAML, benchmark definition, and prepared configuration hashes. It copies
the exact JSON/CSV bytes to stable private filenames and writes an explicit
verification script; it does not execute the script or rerun the benchmark.

```bash
python -m tools.nightly_cli locate-results \
  --config tests/e2e/nightly/single_node/models/configs/Qwen3-30B-A3B-W8A8.yaml \
  --case Qwen3-30B-A3B-W8A8-TP1 --benchmark perf \
  --manifest /tmp/nightly/run-unique/client/manifest.json \
  --log /tmp/nightly/run-unique/client/benchmark.log \
  --output-dir /tmp/nightly/run-unique/collected
bash /tmp/nightly/run-unique/collected/verify.sh
```

The collected directory contains `result.json`, `result.csv`, and
`result-location.json` with original locations and byte hashes. `verify.sh`
explicitly supplies the copied paths to `verify`, producing `verification.json`.
Successful result location means only that the files were identified; a failed
AISBench process must remain failed in the supervisor even if it left results.
Alternatively, pass the exact existing result files directly:

```bash
python -m tools.nightly_cli verify \
  --config tests/e2e/nightly/single_node/models/configs/Qwen3-30B-A3B-W8A8.yaml \
  --case Qwen3-30B-A3B-W8A8-TP1 --benchmark perf \
  --result-json /exact/current/result/directory/gsm8k.json \
  --result-csv /exact/current/result/directory/gsm8k.csv \
  --output-file /tmp/nightly/run-unique/verification.json
```

Exit codes are `0` for passed checks, `1` for failed performance thresholds, and
`2` for invalid input, missing files, or unsupported configuration. The optional
verification JSON records the verdict and reason. Manifests and verification
reports include the source YAML SHA256, benchmark configuration SHA256, and
baseline/threshold. Keep the original YAML and native JSON/CSV alongside them.

## Dependencies and tests

Preparation and verification require Python 3.11+ and PyYAML, with no NPU runtime
imports. Executing generated configurations requires the normal AISBench
dependencies, local model/dataset inputs, and a separately prepared vLLM service.
Installing environments remains a separate action.

```bash
python -m unittest tests.ut.tools.test_nightly_cli
ruff check tools/aisbench.py tools/aisbench_config.py tools/nightly_cli.py \
  tests/ut/tools/test_nightly_cli.py
```

The CLI tests install subprocess audit guards that reject service connections,
child process creation, and imports of lifecycle wrappers. They cover complete
configuration output, safe server arguments, unchanged input templates and
dataset bytes, explicit result verification, optional thresholds, and optimized
Python. These CPU tests do not claim a successful NPU performance run.

## Explicit image runtime reuse

`tools/nightly_environment.py` defaults to the existing `source-install` policy.
Opt in to `--runtime-mode image-reuse` to use an image's installed runtime rather
than installing the requested PR's vLLM/Ascend revisions:

```bash
python tools/nightly_environment.py --role server --runtime-mode image-reuse \
  --vllm-sha "$(cat .github/vllm-main-verified.commit)" \
  --dep-dir /tmp/nightly/environment-unique/server
source /tmp/nightly/environment-unique/server/activate.sh
```

This performs no server dependency installation, source clone, or compilation.
It loads installed CANN/ATB environment scripts, tests imports and the vLLM CLI,
and captures `pip check`. The report explicitly separates the requested helper
source and vLLM revision from actual imported versions, paths, Git revisions,
dirty status, and tracked-diff hashes. An originally modified image may be used;
its modifications are recorded. This is not proof that the PR runtime executed,
nor a complete dependency solution. Unknown runtime source or an import from the
requested PR checkout is rejected.

Client reuse checks out the frozen AISBench revision from an existing local Git
repository into a private directory. It ignores and never edits the source
repository's working tree, recording its state. No dependencies are installed
unless the separate client-only flag is supplied:

```bash
python tools/nightly_environment.py --role client --runtime-mode image-reuse \
  --install-client-dependencies \
  --vllm-sha "$(cat .github/vllm-main-verified.commit)" \
  --benchmark-source /mnt/share/c00814587/benchmark \
  --dep-dir /tmp/nightly/environment-unique/client
source /tmp/nightly/environment-unique/client/activate.sh
```

The optional installation uses a private `venv --system-site-packages`, reuses
the image's installed Torch/NumPy/NPU stack, and constrains existing protected
packages and Transformers to their observed versions. Only the fixed AISBench
API dependencies and PyYAML are installed as needed. The HTTP client explicitly
uses private OpenCV 4.11.0.86 and Pillow 11.2.1 to satisfy the frozen AISBench
requirements; the image's original packages must remain unchanged. Reports
record the image baseline and effective private environment. Installation,
dependency audit, or CLI failures prevent successful activation. Without this
flag, missing client dependencies fail rather than trigger an installation.

One client-only dependency exception is recorded explicitly: the observed image's
`vllm 0.28.0+empty` metadata requires `opencv-python-headless>=4.13.0`, while this
HTTP client uses private OpenCV 4.11.0.86 with NumPy 1.x. This exact conflict is
permitted because the client does not run vLLM; all other new or worsened
conflicts still fail setup. Original image conflicts and the exception remain
visible in the report; successful setup does not claim a clean `pip check`.

Activation selects private command wrappers and, when present, the private
venv's Python. It removes PR paths from `PYTHONPATH` and enters a neutral working
directory. Render YAML commands from the frozen helper checkout, then return to
a neutral directory before executing the service or benchmark. The AISBench
wrapper imports `ais_bench.benchmark.cli.main:main` from the private fixed
checkout; it does not instantiate `AisbenchRunner`.
