# ACL Graph

## Overview

ACL Graph is the Ascend realization of vLLM static graph execution. Upstream vLLM and PyTorch documents already describe the generic graph model, including `CUDAGraphMode`, runtime dispatch, batch descriptors, bucketing and padding, and the definitions of full graph and piecewise graph. This document focuses on what is specific to Ascend in `vllm-ascend`: the platform integration points, the extra constraints introduced by ACL graph capture, and the mechanisms used to keep attention parameters correct during replay.

On Ascend, the design goal is the same as upstream static graph execution: reduce host launch overhead for small and medium runtime shapes. The implementation boundary is different. vLLM provides the generic dispatch path, while `vllm-ascend` supplies the platform wrapper, capture-size trimming, and attention-specific update logic needed by ACL graph replay.

## Prerequisites and References

- Upstream vLLM design doc for generic graph concepts: [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html).
- PyTorch graph documentation for generic capture and replay semantics: [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
- Ascend user guide for operational enablement: [Graph Mode Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html).
- Existing repo design note: `docs/source/developer_guide/Design_Documents/ACL_Graph.md`.

This document intentionally does not re-explain upstream topics such as graph mode selection, dispatcher behavior, batch descriptor construction, capture bucketing, padding policy, or the generic meaning of full versus piecewise execution.

## How ACL Graph Fits into vLLM

vLLM owns the generic static graph flow. On Ascend, `NPUPlatform.get_static_graph_wrapper_cls()` returns `vllm_ascend.compilation.acl_graph.ACLGraphWrapper`, which is the platform-specific wrapper used when vLLM enables static graph mode.

`ACLGraphWrapper` is responsible for:

- reading the runtime mode and `batch_descriptor` from the forward context,
- deciding whether to run eagerly, capture a new ACL graph, or replay a cached ACL graph,
- caching graph entries per batch descriptor,
- preserving the graph pool and replay bookkeeping needed by the Ascend backend.

The wrapper does not define the upstream dispatch policy. It assumes the runtime mode and batch descriptor have already been chosen correctly by vLLM, then applies Ascend capture or replay to that concrete runtime shape.

## Capture Sizes and Bucketing

vLLM graph replay requires stable runtime shapes, so vLLM does not try to capture every possible batch shape. Instead, it prepares a finite set of capture sizes and dispatches a runtime batch to the nearest supported size. If the runtime batch is larger than the largest configured capture size, graph mode is skipped and execution falls back to eager mode.

By default, vLLM builds capture sizes as:

- `1`, `2`, `4`
- multiples of `8` from `8` up to `255`
- multiples of `16` from `256` up to `max_cudagraph_capture_size`

Conceptually, the default list looks like:

```text
[1, 2, 4, 8, 16, 24, 32, ..., 248, 256, 272, 288, ...]
```

The smaller step at small batch sizes reduces padding overhead where latency is most sensitive, while the larger step at bigger sizes keeps the number of captured graphs under control.

On Ascend, this generic upstream bucketing strategy is still the starting point, but the final capture sizes may be reduced further by platform-specific constraints:

- sequence-parallel filtering may remove unsupported sizes,
- stream-budget trimming may reduce the number of sizes that can be captured,
- some runtime modes may be normalized before capture begins.

## Ascend-Specific Design Constraints

### Stream budget constrains capture breadth

Unlike CUDA Graph, ACL graph capture is limited by stream resources. The current implementation treats graph count as a stream budget problem and trims capture sizes accordingly in `vllm_ascend.utils.update_aclgraph_sizes()`. The trimming logic starts from the configured capture sizes, estimates per-graph resource cost from model depth and communication structure, and samples a smaller representative size set when the requested range would exceed the supported budget.

The current implementation uses a practical maximum graph count budget of about 1800, below the device stream limit, and further reduces the budget for communication-heavy cases such as context parallel execution. Piecewise mode is more constrained because each captured segment consumes resources independently, roughly one graph per layer.

The communication execution mode also matters. `update_aclgraph_sizes()` uses different formulas depending on `HCCL_OP_EXPANSION_MODE`. In practice, `HCCL_OP_EXPANSION_MODE=AIV` can increase the number of supported capture sizes, while the default communication unfolding path is more restrictive and reduces the supported runtime shape range.

### Platform mode normalization is stricter than generic upstream behavior

Ascend currently narrows some generic upstream modes in `vllm_ascend.platform.NPUPlatform.check_and_update_config()`.

- `FULL_AND_PIECEWISE` is normalized to `PIECEWISE`.
- Encoder-decoder models are forced to `PIECEWISE`.
- `use_inductor` is disabled for ACL graph paths.
- `ASCEND_LAUNCH_BLOCKING=1` is rejected when ACL graph is enabled.
- Xlite graph mode can disable ACL graph full mode or fall back to `FULL_DECODE_ONLY`, depending on configuration.

These checks document the subset of upstream graph behavior that the current Ascend backend can execute safely. Some of them are long-term platform constraints, while others are clearly transitional in the current implementation.

## Key Ascend-Specific Mechanisms

### Host-side attention parameter update for full graph replay

Full graph replay on Ascend has an extra problem that upstream generic documentation does not cover in detail: some attention operators need runtime metadata updates even when the overall graph is static. The Ascend implementation handles this by separating graph capture from host-side task parameter updates.

The flow is:

1. During capture, attention backends record per-graph task handles, events, workspaces, and weak references to the tensors or metadata that must be refreshed.
2. Before replay, `update_full_graph_params()` calls the backend specific `update_graph_params()` implementation.
3. That backend runs parameter refresh on an update stream with `torch.npu.graph_task_update_begin(...)` and `torch.npu.graph_task_update_end(...)` around the underlying attention operator launch.
4. `torch.npu.ExternalEvent` objects are used to enforce ordering between the host-side update stream and the replay stream.

This mechanism is implemented in attention backends such as:

- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/attention/mla_v1.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/mla_cp.py`

The important design point is that Ascend full graph support depends on backend-provided `update_graph_params()` hooks. Without that hook, capture alone is not enough to replay the correct attention state.

### Replay ordering and synchronization

`ACLGraphWrapper` synchronizes the current stream before replay in the common path to ensure that host-side parameter updates stay aligned with the graph execution that will consume them. This is especially relevant in asynchronous scheduling or multi-threaded execution.

If ordering is not preserved, the parameter update for iteration *i* can be observed by the replay of iteration *i-1*, or the replay of iteration *i* can start before its own parameter update has completed. In practice, this means the attention operator may run with mismatched runtime metadata, which can cause incorrect results, precision issues, or even hangs. The code keeps a narrower path for the main full-graph eagle case, but the general design assumption is the same: replay must not overtake pending parameter update work.

## Full vs Piecewise on Ascend

Upstream docs already define full graph and piecewise graph semantically. On Ascend, the practical difference is driven by backend support and resource cost.

### Piecewise mode

Piecewise mode is the conservative path. It relies on the generic vLLM split execution strategy, then applies ACL graph capture to the non-attention segments selected by the compilation path. On Ascend, this mode is currently the more widely supported option, but it is also the most sensitive to stream pressure because the number of captured graphs scales with model depth.

### Full graph mode

Full graph mode is the more performance-oriented path when the attention backend can support runtime parameter patching through `update_graph_params()`. On Ascend, full graph support is tied to those attention-specific update hooks, workspace caching, and replay ordering guarantees.

## Diagnostics and Operational Notes

- The simplest way to confirm that graph mode is active is to enable cudagraph metrics and keep log stats enabled. In CLI usage, use `--cudagraph-metrics` and do not pass `--disable-log-stats`. In Python usage, set `cudagraph_metrics=True` and `disable_log_stats=False`. Then inspect the emitted metrics and logs.
- Profiling can also confirm whether replay is happening, and developers can add temporary prints before replay when debugging locally, but those are secondary methods and are not expanded here.
- `update_aclgraph_sizes()` is the main implementation point for stream-budget-driven capture-size trimming.
- In debug mode, `ACLGraphWrapper` asserts that replay uses the same tensor addresses recorded during capture.
- `ASCEND_LAUNCH_BLOCKING=1` is incompatible with ACL graph enablement in the current implementation.
- For debugging inside graph execution, the repo also provides graph-aware print helpers in `vllm_ascend.utils`, but those are developer diagnostics rather than part of the execution design.

## Related Files

- `vllm_ascend/platform.py`, mode normalization, platform hooks, and static graph wrapper selection.
- `vllm_ascend/compilation/acl_graph.py`, ACL graph wrapper, capture and replay cache, graph parameter containers, and full graph update dispatch.
- `vllm_ascend/utils.py`, capture size adjustment through `update_aclgraph_sizes()`.
- `vllm_ascend/attention/attention_v1.py`, full graph attention parameter capture and update logic.
- `vllm_ascend/attention/mla_v1.py`, MLA (Multi-Head Latent Attention) specific full graph parameter capture and update logic.
- `vllm_ascend/attention/context_parallel/attention_cp.py`, context parallel attention update path.
- `vllm_ascend/attention/context_parallel/mla_cp.py`, context parallel MLA update path.
