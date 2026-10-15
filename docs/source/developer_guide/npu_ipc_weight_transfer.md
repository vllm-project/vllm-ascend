# NPU IPC weight transfer inheritance

The NPU IPC backend extends the upstream IPC engines and overrides the device-specific operations. It requires the trainer and inference worker to share a physical NPU. The public backend name remains `npu_ipc`, with the existing init/update dataclasses and wire format.

This refactor was checked against local vLLM commit `91ed2859c7345dc160956781b4662dbf99870005` and vLLM Ascend base commit `23d1d9055f2161b3c15613c7fc309f7d5d2ef629`. It needs no changes to upstream vLLM.

## What is inherited

| Ascend class | Upstream base | Reused behavior |
| --- | --- | --- |
| `NPUIPCWeightTransferInitInfo` | `IPCWeightTransferInitInfo` | Worker-side packed-mode handshake fields |
| `NPUIPCWeightTransferUpdateInfo` | `IPCWeightTransferUpdateInfo` | Metadata validation and serialized-handle decoding |
| `NPUIPCWeightTransferEngine` | `IPCWeightTransferEngine` | Handshake, START/FINISH layerwise reload, mapping cleanup on FINISH/shutdown, model target management and per-update synchronization |
| `NPUIPCTrainerWeightTransferEngine` | `IPCTrainerWeightTransferEngine` | `trainer_init`, `send_weights`, `_send`, `_send_unpacked`, `_do_send` and shutdown |
| `NPUPackedBufferImporter` | `PackedBufferImporter` | Cache initialization and mapping release through `close` |

The trainer constructor initializes `TrainerWeightTransferEngine` directly because the immediate upstream constructor queries CUDA device properties. The NPU UUID is also assigned to the upstream `gpu_uuid` attribute, which the inherited unpacked sender uses only as an opaque handle-dictionary key. No CUDA properties are required.

## What stays specific to Ascend

- The worker receives handles through `rebuild_npu_tensor`. Its assigned device index selects both the physical UUID and the rebuild destination; an unrelated ambient current device cannot select the wrong handle.
- The trainer retains NPU synchronization, collective handle aggregation and packed-buffer production. The upstream equivalents currently call CUDA primitives directly, so inheriting them would execute the wrong backend.
- Packed chunks share one `NPUPackedBufferImporter` on the worker. Repeated chunks of the same export reuse the mapping; a new export replaces it. The inherited FINISH and shutdown methods release the mapping.
- Unpacking reuses upstream `unpack_tensor`. Each returned tensor is cloned before the producer can overwrite the shared buffer, preserving data retained by layerwise reload.
- The receive operation uses the upstream MTP completeness-check scope during each chunk load and restores the scope on exceptions.

The packed producer flattens before changing dtype, which supports scalar weights. A final chunk is emitted when it has parameter names, including an all-empty tensor payload. An empty source still emits no chunks.

## Validation

The focused unit tests cover packed and unpacked transfers, sender and non-sender ranks, two successive trainer/worker updates, non-contiguous tensor lifetime through FINISH and synchronization, collective merging, per-chunk synchronization before buffer reuse, mapping reuse/release, device visibility mapping, MTP scope restoration, oversized tensors, and scalar/empty/single-element/matrix shapes.

Run in the normal vLLM Ascend development environment:

```bash
pytest -q tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py tests/ut/distributed/weight_transfer/test_packed_tensor.py
```

The local Windows validation used an isolated loader with CPU PyTorch 2.2.1: it loaded the real upstream base/IPC/packing/MTP modules and the Ascend implementations, while replacing configuration imports and NPU/CUDA runtime boundaries. It bypassed the repository-wide UT conftest and did not validate plugin discovery or actual IPC handle exports. The result was **63 passed**; Ruff checks and formatting checks were also run on the four changed Python files.

Real multi-process NPU IPC, HCCL collectives, and model-output correctness still require an Ascend environment with the supported dependency versions. The existing model-aware end-to-end test remains the hardware validation entry point:

```bash
pytest -q tests/e2e/pull_request/one_card/rlhf/state_transitions/test_npu_ipc_weight_transfer.py
```
