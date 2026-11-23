# Model Execution Domain Checklist

Rules for PRs touching the inheritance surfaces where vllm-ascend extends
upstream vLLM: `vllm_ascend/worker/` (NPUWorker, NPUModelRunner V1/V2),
`vllm_ascend/attention/`, `vllm_ascend/compilation/`, `vllm_ascend/sample/`,
`vllm_ascend/spec_decode/`. These classes override upstream base classes
(WorkerBase, GPUModelRunner, AttentionBackend, AttentionImpl, Sampler, ...);
the overrides track upstream signatures closely, so signature drift breaks
the plugin even when the plugin's own code is self-consistent.

- `L2-exec#override-contract`: When a PR changes the signature, return
  type or call order of a method that any Ascend override implements
  (init_device,_make_model_runner, load_model, determine_available_memory,
  execute_model, sample_tokens,_prepare_inputs,_build_attention_metadata,
  _dummy_run, _allocate_kv_cache_tensors, capture_model, ...), ALL Ascend
  overrides must be updated together: `worker/worker.py`,
  `worker/model_runner_v1.py`, `worker/v2/model_runner.py` and
  `vllm_ascend/_310p/worker_310p.py`. One missed override is a MUST-FIX.
- `L2-exec#attention-positional`: The attention stack (AttentionBackend,
  AttentionMetadataBuilder, AttentionImpl, MLA backends MLA/SFA/DSA/FA)
  relies on positional `__init__` and `forward` contracts. Adding or
  reordering parameters without updating every Ascend backend
  (attention_v1, mla_v1, sfa_v1, dsa_v1, fa3_v1) breaks all kernels —
  MUST-FIX unless every implementation is updated.
- `L2-exec#v1v2-parity`: A change to `worker/model_runner_v1.py` should be
  mirrored in `worker/v2/model_runner.py` (and checked against the 310P
  runner) unless the divergence is intentional and commented. Silent
  divergence is a SHOULD-FIX.
- `L2-exec#hardware-profile`: Hardware-specific branches (A2/A3/A5/310P)
  belong in `device/hardware_profile.py` (HardwareProfile/HardwareCapability),
  not as scattered chip checks at call sites. A new inline chip branch is a
  SHOULD-FIX.
- `L2-exec#310p-containment`: 310P-specific logic lives in the
  self-contained `vllm_ascend/_310p/` implementation. New 310P branches
  outside that directory need justification in the PR.
