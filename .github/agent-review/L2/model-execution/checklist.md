# Model Execution Domain Checklist

Rules for PRs touching the inheritance surfaces where vllm-ascend extends
upstream vLLM: `vllm_ascend/worker/` (NPUWorker(GPUWorker) in `worker.py`,
NPUModelRunner(GPUModelRunner) in `worker/v2/model_runner.py`), the
self-contained 310P stack in `vllm_ascend/_310p/` (NPUWorker310(NPUWorker),
NPUModelRunner310V2(NPUModelRunner) in `_310p/worker/v2/`),
`vllm_ascend/attention/` and `vllm_ascend/compilation/`. These classes
override upstream base classes (GPUWorker, GPUModelRunner,
AttentionBackend, AttentionImpl, ...); the overrides track upstream
signatures closely, so signature drift breaks the plugin even when the
plugin's own code is self-consistent.

- `L2-exec#override-contract`: When a PR changes the signature, return
  type or call order of a method that any Ascend override implements
  (init_device,_make_model_runner, load_model, determine_available_memory,
  execute_model, sample_tokens,_prepare_inputs,_build_attention_metadata,
  _dummy_run, _allocate_kv_cache_tensors, capture_model, ...), ALL Ascend
  overrides must be updated together: `worker/worker.py`,
  `worker/v2/model_runner.py`, `_310p/worker_310p.py` and
  `_310p/worker/v2/model_runner.py`. One missed override is a MUST-FIX.
- `L2-exec#attention-positional`: The attention stack (AttentionBackend,
  AttentionMetadataBuilder, AttentionImpl, MLA/SFA/DSA/FA backends)
  relies on positional `__init__` and `forward` contracts. Adding or
  reordering parameters without updating every Ascend backend
  (attention_v1, mla_v1, sfa_v1, dsa_v1, dsa_v41, fa_v1, fa3_v1, flashmla)
  breaks all kernels — MUST-FIX unless every implementation is updated.
- `L2-exec#310p-parity`: A behavioral change to the main stack
  (`worker/v2/model_runner.py`, `attention/`) should be checked against the
  self-contained 310P stack (`_310p/`) — it deliberately duplicates parts of
  the implementation (NPUModelRunner310V2 inherits NPUModelRunner but
  `_310p/` also carries its own attention, fused_moe, ops and quantization).
  Silent divergence between the two is a SHOULD-FIX.
- `L2-exec#hardware-profile`: Hardware-specific branches (A2/A3/A5/310P)
  belong in `device/hardware_profile.py` (HardwareProfile/HardwareCapability),
  not as scattered chip checks at call sites. A new inline chip branch is a
  SHOULD-FIX.
- `L2-exec#310p-containment`: 310P-specific logic lives in the
  self-contained `vllm_ascend/_310p/` implementation. New 310P branches
  outside that directory need justification in the PR.
