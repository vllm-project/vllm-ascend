# GLM-5.3-Flash Triton operator comparison

This standalone tool compares the KPool indexer, tail compressor and gated norm
in PR #17542 with its mainline baseline. It needs one idle Ascend NPU and an
existing environment with CANN, PyTorch, torch_npu, Triton-Ascend, vLLM and pytest.
It loads bundled operator sources without changing installed operators or loading
model weights. It can run on A3 or A5; hardware results must be collected on each.

## Build and run

From a checkout containing both pinned commits:

```bash
python3 benchmarks/scripts/glm53_triton_check/build_bundle.py --output /tmp/glm53-triton-check
# Select an idle device visible inside your container.
ASCEND_RT_VISIBLE_DEVICES=0 bash /tmp/glm53-triton-check/run.sh
```

The builder also creates `/tmp/glm53-triton-check.tar.gz` for another machine.
Use a new output directory for each build. The target machine does not need Git.
If a shallow clone lacks either commit, fetch it before building. Defaults are
baseline `ffc1c7a82bfef1d9b434da1206e08589387fdb28` and candidate
`faaa46690d751b22ce5ddd8673ca56a5fabcf776`; overrides are recorded as full commit IDs.
The builder keeps operator sources intact and extracts four norm functions
verbatim from `kda.py` to avoid unrelated recurrent-KDA imports.

```bash
# Expanded shapes, or correctness without latency measurements.
ASCEND_RT_VISIBLE_DEVICES=0 bash /tmp/glm53-triton-check/run.sh --suite full
ASCEND_RT_VISIBLE_DEVICES=0 bash /tmp/glm53-triton-check/run.sh --check-only
# Bundle integrity needs no NPU libraries; the regression check needs CPU PyTorch.
python3 /tmp/glm53-triton-check/run_a5.py --check-bundle
python3 /tmp/glm53-triton-check/check_score_regression.py
```

Use `PYTHON=/path/to/python bash .../run.sh` to select an environment. The selected
logical NPU defaults to `0`; `--device` changes it after visibility mapping.

## Coverage and interpretation

- Reuses 27 existing operator cases, changing only their import paths.
- Compares **valid** indexer score rows and ordered outputs exactly (`rtol=atol=0`).
  Candidate padding scores must equal the lowest finite FP32 value and output
  indices must equal `-1`. Mainline's discarded padding scores are not an oracle:
  in a 64-row bucket with 8 live rows and 875 pools, all 49000 padded score cells
  may differ legitimately. The earlier standalone harness compared them by mistake.
- Replays a 64-row graph bucket with 8/16/8 live rows using
  `allow_cache_packing=False`, checking output aliasing and padding. This checks
  operator graph behavior, not the model's FULL-mode routing.
- Checks compressor tail/cache equality and gated-norm dtype, activation,
  residual and statistics branches. Norm row counts are flattened rows, not requests.

Each `results-*` directory contains `run.log`, `results.json`, `summary.csv` and
`operators.xml`. JSON retains software versions, source hashes, the command and
every timing sample. Both variants compile and warm up before alternating A/B
measurements in NPU graphs; seven samples are retained by default.

Positive reduction means lower latency. Changes within 5% are `NEUTRAL`; a
slowdown above 5% is `REGRESSION`. This screening threshold does not establish
statistical significance. Recheck suspect cases on an idle device with
`--samples 15 --replays 50`. Exit codes are `0` for correctness passed without
flagged regressions, `1` for correctness/environment errors, and `2` for flagged
latency regressions. Zero does not mean every case improved.

These are isolated operator latencies, not model TPOT, throughput or GSM8K.
Use EvalScope or `vllm bench` for model performance with identical model, MTP,
parallelism and request settings for both variants.

The padding fix has CPU regression coverage. A3 device rerun and A5 confirmation
are pending; publishing this harness does not claim either passed.
