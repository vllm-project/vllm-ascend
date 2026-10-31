# GLM-5.1 A5 TP8/EP8 runner reproduction

`GLM5_1_W4A4_A5_TP8_runner_smoke.yaml` uses the existing single-node E2E
loader and test. Its two cases run sequentially with identical model arguments
and environment apart from `VLLM_USE_V2_MODEL_RUNNER`.
It adds no collector, comparison framework or scheduled nightly job.

## Recorded full-model baseline

| Component | Recorded value |
| --- | --- |
| Model | `Eco-Tech/GLM-5.1-w4a4c8-mxfp4` |
| ModelScope revision | `b54a6d54803fa2cb4c28a73423061d882d9c06b3` |
| vLLM Ascend | `5bd7714b5ef39315cc26867bc67826d42ac0760a` plus this PR's shared FP8 fallback |
| vLLM | `544f182d9f0c9bfe8bd7645525951d5e3692a542` |
| Hardware | Eight visible Ascend 950DT devices, 98304 MB HBM per device |
| Python / torch / torch-npu | `3.12.13` / `2.10.0+cpu` / `2.10.0.post7` |
| CANN | `9.2.0`, `V100R001C12SPC002B200` |
| transformers / Triton | `5.19.0` / `3.6.0` |
| Parallelism | TP8, EP8; DP/PP/PCP/DCP1 |
| Graph | `FULL_DECODE_ONLY`, capture sizes 1/2/4/8/16/32 |
| Quantization | Checkpoint's mixed MXFP4/MXFP8; FP8 KV and indexer cache |

The recorded NPU captures use the frozen stack above, not a claim about an
arbitrary current-main image. The shared FP8 fix is required in both workers.
The original candidate was `072f1bf710f37590d31e096e359ae3bc0a3a87d3`;
later CPU/package changes do not retag the original hardware runs.

## Run the minimal E2E cases

Use the matching runtime and the complete, unmodified revision above, mounted
at `/models/GLM-5.1-w4a4c8-mxfp4`. The local-path model argument does not verify
weight identity: verify the snapshot/manifest before running. The explicit
`--revision` also records the intended revision. Use eight idle A5 devices.

From the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
CONFIG_BASE_PATH=tests/e2e/cases/models/configs/GLM \
CONFIG_YAML_PATH=GLM5_1_W4A4_A5_TP8_runner_smoke.yaml \
PYTEST_ADDOPTS='' \
python -m pytest -q -s tests/e2e/common/single_node/test_single_node.py
```

Do not enable xdist for this profile: each case uses all eight devices and the
existing harness must release one server before starting the next. The loader
allocates the port. The endpoint uses the default model name. Those transport
details differ from the standalone capture launcher; model-execution arguments
are otherwise the shared graph baseline. No APC, async scheduling, EPLB or MTP
is enabled by this minimal profile.

This entry is a short generation smoke with three completion prompts per arm,
greedy decoding, seed 1024 and logprobs requested. The existing completion test
checks a nonempty first response; it does not implement cross-runner token or
logprob comparison. Passing it must not be reported as full precision or
performance parity.

## Paired precision, performance and boundaries

Full paired results, raw data, commands and their original version provenance
are delivered independently in the
[task evidence](https://github.com/vllm-project/vllm-ascend/pull/20387#issuecomment-6526361475).
The executable collection/comparison tools and their configuration are frozen
at [48dd063](https://github.com/romavec/vllm-ascend/tree/48dd063a74ecacaabb0d40ed9272fa1cbbd612e0/tests/e2e/runner_parity).
Keep `tests/__init__.py` and `tests/e2e/__init__.py` from that snapshot and run
its documented `python -m tests.e2e.runner_parity.<module>` commands from the
archive root. These tools are not dependencies of this YAML smoke entry.

The evidence includes greedy/seeded sampling, fixed-history logprobs, long
inputs, real-generated reasoning/tool parsing, MTP k3 eager and boundary cases.
The repeated serving differences remain unresolved in
[issue 20388](https://github.com/vllm-project/vllm-ascend/issues/20388).
No graph-plus-MTP throughput, stable speedup or no-regression claim is made.
