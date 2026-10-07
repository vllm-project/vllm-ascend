# StartLux-Decision inference on Ascend

This recipe uses native vLLM classification models for the dense 4B and MoE
35B-A3B checkpoints. The models inherit the Qwen3.5 implementations in vLLM;
Ascend supplies its existing attention and MoE operators. No model class is
registered or patched in the hardware plugin.

## Revisions and environment

- vLLM fork branch: `Liuchenbing-2026/vllm:startlux_decision_npu`, based on
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0).
- Ascend fork branch: `Liuchenbing-2026/vllm-ascend:startlux_decision_npu`, based
  on `a8fcedb03d93e60efceddbfc912406f7fa491d57` with the existing packed-GDN
  normalization change `eb68728c5989645edcd5f43db82d711ff7412aef`.
- The Ascend Dockerfile and `.github/vllm-main-verified.commit` specify the
  above vLLM base. Use paired revisions, not arbitrary latest branches.
- Base image: `quay.io/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85`.
- Official prompt and evaluation source:
  `StartLuxLabs/StartLux-Decision@0e7a2e81b9c92756e26d8edd843a44d50e362669`.
- Checkpoints: `startlux-models/StartLux-Decision-4B` and
  `startlux-models/StartLux-Decision-35B-A3B` on Hugging Face, or the same
  checkpoint names under `StartLuxAI` on ModelScope. Preserve model licenses.

Create a new container and a virtual environment for this recipe. Install the
paired editable repositories following the Ascend installation guide. The
vLLM install uses `VLLM_TARGET_DEVICE=empty`; build Ascend C++ extensions when
switching away from the compiled revision supplied by the image. Installing
CUDA dependencies from the official StartLux requirements is unnecessary:
only its renderer and answer protocol are imported.

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
services. Begin with eager execution; removing `--enforce-eager` must be
validated against eager outputs before making graph-mode accuracy claims.

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
```

Keep full predictions, final metrics, exact weight hashes and timing evidence
in the local experiment archive. Serving startup and CPU unit tests alone do
not establish seven-suite accuracy, tensor-parallel equivalence or latency.
