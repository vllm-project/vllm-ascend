# StartLux native text decisions on Ascend

This recipe uses the native vLLM decision model and `/v1/decisions` strategy
with group-local encoder-only attention metadata on Ascend.
It requires paired vLLM and vLLM-Ascend checkouts containing the StartLux model,
its typed decision strategy, and the encoder-only metadata fix.
The legacy standalone `serve_startlux_decision.py` script is not required.

## Paired code and validation scope

The model implementation and typed request strategy belong to vLLM. The Ascend
plugin keeps per-group `causal` metadata local to `EncoderOnlyAttentionSpec`
layers so hybrid linear/full-attention pooling checkpoints do not mutate shared
metadata or run full-attention layers causally.

The refreshed paired baseline for this branch is:

- vLLM base `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` with
  `Liuchenbing-2026/vllm` branch `startlux_decision_eager_pr-g`
  (`5ead79dded61ee73f8c5419c1bb3dcc71e682f09`).
- vLLM-Ascend base `a38556f69529595e70598d0d6203009f5ac3293e` with this
  branch (`startlux_decision_eager_pr-g`).

**Validation status on this refreshed baseline:** only static syntax checks and
launcher CLI/argument tests have been run locally; CPU unit tests, CI/lint, and
NPU model/E2E evaluation on this paired baseline are pending. Do not treat
historical v0.30.0rc2 or earlier experimental candidate results as acceptance of
this refreshed main pair.

Before deployment, inspect the selected Ascend revision's `Dockerfile`, image
build workflow, and `.github/vllm-main-verified.commit`; install matching
`<VLLM_SOURCE_DIR>` and compiled `<ASCEND_SOURCE_DIR>` sources into `<PYTHON_BIN>`
in a fresh container and record the installed commits and image digest.

## Start the service

Keep `decision_config.json`, the tokenizer, and the official chat template with
the local checkpoint. Activate the environment containing the paired packages.

```bash
export MODEL_PATH='<WEIGHTS_DIR>'
export ASCEND_RT_VISIBLE_DEVICES='<ASSIGNED_NPU_IDS>'
export PYTHON_BIN='<PYTHON_BIN>'
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=8192
bash examples/models/startlux_decision/serve.sh --enforce-eager
```

Use the same launcher for v1 dense checkpoints. For the v1 MoE checkpoint,
select the corresponding architecture explicitly:

```bash
bash examples/models/startlux_decision/serve.sh --enforce-eager \
  --hf-overrides '{"architectures":["StartLuxDecisionMoeForSequenceClassification"],"num_labels":26}'
```

The launcher binds localhost (`127.0.0.1`) and preserves the assigned device
selection. `PORT`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `PYTHON_BIN`
configure the service. Both `serve.sh` and `serve_instant.sh` disable prefix
caching and chunked prefill and set `--max-num-batched-tokens` to the full
prompt budget (`MAX_MODEL_LEN`), rejecting overrides that would split a
bidirectional prompt.

Call the native `/v1/decisions` endpoint using its choice, predicate, and score
schema. `/classify` returns raw candidate logits, not calibrated probabilities.

## Validation boundaries

This recipe targets text-only, single-pass reads. It does not implement
AutoThink, images, graph replay dispatcher changes, or long-running service
acceptance. Launcher tests verify argument forwarding and reject partial-prompt
scheduling overrides; they do not prove that a real checkpoint started or passed
a model benchmark.
