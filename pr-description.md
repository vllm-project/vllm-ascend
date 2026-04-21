## New Parameter Compatibility: --language-model-only

### Parameter Info
- **Parameter**: `--language-model-only`
- **Value**: `true`
- **Config Class**: `MultiModalConfig`
- **Tree Root**: `MultiModalConfig`
- **Node ID**: `MultiModalConfig/language-model-only`

### Test Result
- **Status**: PASS
- **Startup Time**: 320.1s
- **Endpoints Verified**: 9/9
  - GET /health: L0=PASS L1=PASS L2=PASS
  - GET /v1/models: L0=PASS L1=PASS L2=PASS
  - GET /load: L0=PASS L1=PASS L2=PASS
  - GET /version: L0=PASS L1=PASS L2=PASS
  - POST /tokenize: L0=PASS L1=PASS L2=PASS
  - POST /detokenize: L0=PASS L1=PASS L2=PASS
  - POST /v1/chat/completions: L0=PASS L1=PASS L2=PASS
  - POST /v1/completions: L0=PASS L1=PASS L2=PASS
  - POST /v1/responses: L0=PASS L1=PASS L2=PASS
- **Stress Test**: 100/100 requests passed (p50=0.855s, p99=0.966s, avg=0.849s)

### Code Changes
- `vllm_ascend/worker/model_runner_v1.py`: Fixed dummy-run branch condition at line 2574. Changed `self.is_multimodal_model` to `self.supports_mm_inputs` to align with upstream `gpu_model_runner.py` logic. When `--language-model-only` is set, `supports_mm_inputs` is `False` (multimodal inputs disabled), but `is_multimodal_model` remains `True` (model architecture is still VL). The old condition caused the dummy run to take the multimodal path (setting `input_ids=None` with uninitialized `inputs_embeds`), crashing with `AttributeError: 'NoneType' object has no attribute 'size'`.
- `tools/api-compatibility/tree/definition.yaml`: Added `--language-model-only` node under `MultiModalConfig` root.

### Debug Rounds
1 (fixed on first debug attempt)
