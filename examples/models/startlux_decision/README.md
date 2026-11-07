# StartLux v1 and v1.1 Instant on Ascend

This main-branch candidate adds group-local encoder-only attention metadata
and complete-prompt launch settings for the StartLux v1.1 Instant dense/MoE
readout. **The refreshed main combination has CPU-only supplemental evidence;
its NPU/model evaluation is still pending.** AutoThink is not covered.

## Current candidate and historical acceptance

| Item | Version / evidence |
| --- | --- |
| Candidate vLLM base | `83c9a547bf9a3b4f2653fb5766886bdc5dba35be` |
| Paired vLLM eager candidate | `a3a20232602267e3ebf9e1defb0a1d0a9c94f9ad`; full runtime validation pending |
| Candidate Ascend base | `ee29cc6fc24d20d47948776764059f05601170a6` |
| Ascend verified vLLM | `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`; candidate pairing is not yet verified |
| Historical v1.1 vLLM eager / graph | `fbd23b169f4b5558b388f40dd038bd47d3f0a1fe` / `2be401d683a214bcbf521bdcf4fb529760e7568f` |
| Historical v1.1 Ascend | `9c88a4e71953f122aecc4275ef2cde0bce5685b1` |
| Historical seven-suite mean, 4B / 27B Instant | 91.07% / 93.03%; these are historical delivery results, not this main candidate's results |

Both main snapshots require their corresponding candidate patches. Follow the
Ascend build documentation to install matching sources and compiled extensions
in a new container. Do not reuse the historical v0.30 image as evidence that
this new pair is compatible. Preserve checkpoint tokenizer, prompt and
calibration metadata.

## v1.1 Instant classification and native decisions

After building the paired candidate sources, use a local checkpoint and
explicitly assigned devices. These are reproduction commands, not a claim that
the refreshed main versions have been run on an NPU.

```bash
export MODEL_PATH='<LOCAL_4B_OR_27B_INSTANT_CHECKPOINT>'
export ASCEND_RT_VISIBLE_DEVICES='<ASSIGNED_NPU_IDS>'
export TENSOR_PARALLEL_SIZE='<TP_SIZE>'
bash examples/models/startlux_decision/serve_instant.sh --enforce-eager
```

Graph execution is not part of this paired eager recipe. Exact-token-count
PIECEWISE replay requires a separate vLLM dispatcher change that is not included
in the paired eager candidate above. Do not enable graph mode on the assumption
that this Ascend change supplies that dispatcher support. The launcher tests
only check argument forwarding; they do not establish graph replay or accuracy.

Instant uses bidirectional full-attention layers, so the launcher disables
chunked prefill and prefix caching. Each request must fit in a single prefill;
`max-num-batched-tokens` is therefore at least `max-model-len`. Do not override
these settings to enable chunking or prefix reuse: representations depend on
the complete prompt, including later questions and candidates. The launcher
rejects these CLI overrides before starting Python. Set `MAX_MODEL_LEN` to
change both length budgets together; protected settings are appended after
extra CLI arguments so a configuration file cannot re-enable them.

`/classify` returns 26 raw A–Z logits in its `probs` field. Clients of this
low-level interface must use the official prompt renderer, restrict candidates
and apply the checkpoint-specific temperature before softmax.

The paired vLLM candidate also registers a text-only single-pass strategy for
its current native `/v1/decisions` route. It preserves choice/predicate/score
formats and type-specific temperatures; see `docs/models/startlux_decision.md`
in that candidate. This is not the historical `/v1/systemone` route. The
`serve.sh` below uses the independent v1 text example and is not a v1.1/image
gateway. AutoThink, images, wide-choice aggregation and none-check passes are
outside the native candidate's scope. CPU handler fixtures do not establish
HTTP startup or NPU accuracy.

The Ascend change uses `EncoderOnlyAttentionSpec` to override only the matching
attention group's causal metadata; shared metadata and recurrent groups stay
unchanged. UT and device regression cases cover both incoming causal flags.

## Historical v1 recipe (separate version pins)

This recipe uses native vLLM classification models for the dense 4B and MoE
35B-A3B checkpoints. The models inherit the Qwen3.5 implementations in vLLM;
Ascend supplies its existing attention and MoE operators. No model class is
registered or patched in the hardware plugin.

## Revisions and environment

The following pins describe the validated v0.30.0 runtime. The upstream model
submission is maintained separately on main; its CI and runtime validation must
complete before substituting that revision into this recipe.

- vLLM fork branch: `Liuchenbing-2026/vllm:startlux_decision_npu_eager`, based on
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0).
- Ascend fork branch: `Liuchenbing-2026/vllm-ascend:startlux_decision_npu_eager`, based
  on `a8fcedb03d93e60efceddbfc912406f7fa491d57` with the existing packed-GDN
  normalization change `eb68728c5989645edcd5f43db82d711ff7412aef`.
- These pins describe the historical deployment only. Current Ascend verified
  vLLM is listed in the candidate table above; it is a different revision.
- Base image: `quay.io/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85`.
- Official prompt and evaluation source:
  `StartLuxLabs/StartLux-Decision@0e7a2e81b9c92756e26d8edd843a44d50e362669`.
- Checkpoints: `startlux-models/StartLux-Decision-4B` and
  `startlux-models/StartLux-Decision-35B-A3B` on Hugging Face, or the same
  checkpoint names under `StartLuxAI` on ModelScope. Preserve model licenses.
- The tested ModelScope weight revisions are
  `9113683823edce9311109fff84cf9719394835d8` (4B) and
  `e6dafa13615ba6858e9489416216a9b46c36e45c` (35B-A3B).
- Runtime: Python 3.12.13, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4,
  Transformers 5.14.1 and triton-ascend 3.2.2. Check actual import paths:
  installed package metadata can describe the image rather than mounted source.

Create a new container and a virtual environment for this recipe. Install the
paired editable repositories following the Ascend installation guide. The
vLLM install uses `VLLM_TARGET_DEVICE=empty`; build Ascend C++ extensions when
switching away from the compiled revision supplied by the image. Installing
CUDA dependencies from the official StartLux requirements is unnecessary:
only its renderer and answer protocol are imported.

The initial experiment reused matching, previously built Ascend extensions.
Those binaries differ from the base image's plugin binaries. On a fresh machine,
build the selected Ascend checkout; do not mix its Python sources with arbitrary
image extensions. No C++ or operator source is changed by this recipe.

## Run

Set paths to the checkouts and downloaded checkpoint inside your container:

```bash
export MODEL_PATH=/models/StartLux-Decision-4B
export VLLM_SOURCE=/workspace/vllm
export STARTLUX_SOURCE=/workspace/StartLux-Decision
export PYTHON_BIN=/workspace/.venv/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0
bash examples/models/startlux_decision/serve.sh --enforce-eager
```

For the 35B-A3B checkpoint, set `TENSOR_PARALLEL_SIZE=4` and select four assigned
NPUs. The default HTTP listener is loopback. Configure `PORT` to run separate
services. This recipe uses eager execution. Graph-mode validation is a separate change.

The launcher defaults the existing vLLM setting
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. In-process scheduling queues the questions
before executing the batch. The background engine can execute the first question
before the rest arrive, causing additional forward passes. Set the variable to
`1` to reproduce that baseline. Tensor parallel workers remain separate when
TP is greater than one. Recheck both probabilities and latency when changing
scheduling or graph shapes.

```bash
curl -s http://127.0.0.1:18190/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Paris is the capital of France.","questions":{"city":{"type":"choice","instructions":"Which city is the capital of France?","criteria":{"Paris":null,"Berlin":null}}}}'
```

Each question is rendered independently with the official thinking-off prompt.
The native pooler returns the last token's raw A–Z logits, with the selected
output-head rows projected in FP32. The reference protocol then restricts the
candidates, applies checkpoint-specific temperatures, and returns probabilities.
It also retains score ordering, yes/no semantics and wide-choice tournaments.
This text example explicitly rejects images; image and 256K-context support
are not established by text-suite validation.

## Validate

Run the focused model tests from the vLLM checkout:

```bash
.venv/bin/python -m pytest tests/models/multimodal/pooling/test_startlux_decision.py
```

Use the official pinned evaluation bundle, without changing labels or prompts:

```bash
cd "$STARTLUX_SOURCE"
bash eval/fetch_benchmarks.sh
"$PYTHON_BIN" eval/suites.py predict --endpoint http://127.0.0.1:18190 \
  --out outputs/startlux-npu
"$PYTHON_BIN" eval/suites.py score outputs/startlux-npu \
  --json outputs/startlux-npu-metrics.json
"$PYTHON_BIN" eval/latency.py http://127.0.0.1:18190/v1/systemone 200
```

The latency script warms up 20 times, then times 200 serial HTTP requests, each
with choice, yes/no and score fields. Report its actual processed token count;
request-content tokens and separately rendered question tokens are different.
Before reporting the seven-suite average, require all 12,351 decisions, zero
missing rows and zero invalid results. Use a fresh output directory for each run.

Keep full predictions, final metrics, exact weight hashes and timing evidence
in the local experiment archive. Serving startup and CPU unit tests alone do
not establish seven-suite accuracy, tensor-parallel equivalence or latency.

## Launcher regression

The CPU-only launcher tests check argument quoting, physical device selection,
source precedence and the in-process scheduling default. They are collected by
the standard `tests/ut` CI selection:

```bash
python -m pytest tests/ut/test_startlux_recipe.py -q
```

These tests do not replace the two-checkpoint NPU accuracy evaluation above.
