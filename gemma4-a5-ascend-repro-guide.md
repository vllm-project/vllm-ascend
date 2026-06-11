# Gemma-4 on Ascend950PR (A5) — Full Reproduction Guide

> Standalone guide. With this file alone, you can build the environment from scratch,
> start the inference service, and complete precision testing.

---

## 1. Environment Information

| Item | Value |
|------|-------|
| NPU Device | Ascend950PR, soc_version=260 (`AscendDeviceType.A5` in vllm-ascend) |
| CANN Version | 9.1.T560 (torch-npu 2.10.0 requires CANN 9.1+) |
| Driver | 25.7.rc1.1 |
| Python | 3.11.14 |
| GCC | 11.4.0 |
| torch | 2.10.0+cpu (**must be CPU-only wheel**, CUDA runtime conflicts with torch-npu) |
| torch_npu | 2.10.0 (**must `--no-deps` install**, otherwise pip pulls non-existent CPU torch) |
| vllm | 0.20.2 (**must `--no-deps` install**, prevents dependency chain upgrading torch) |
| vllm-ascend | 0.19.1rc2.dev94 (editable install, `COMPILE_CUSTOM_KERNELS=0`) |
| transformers | 5.5.3 (Gemma4 architecture requires 5.x+) |
| triton | 3.2.0 |
| triton-ascend | 3.5.1.dev (**must `--no-deps` install**) |
| numpy | 1.26.4 (forced version) |
| evalscope | 1.8.0 (precision evaluation framework) |
| SOC_VERSION (build) | ascend950pr_957b |
| vllm-ascend device_type | A5 (hardcoded in `_build_info.py`) |
| HBM per card | 114688 MB (8 cards) |
| Model | gemma-4-26B-A4B-it at `/home/models/gemma-4-26B-A4B-it` |

---

## 2. Source Code Patches (Required for A5)

All patches below are mandatory. Missing any one causes crash or garbled output.

### 2.1 vllm-ascend patches (7 modifications, 6 files)

#### 2.1.1 `_build_info.py` — Set device type to A5

```bash
cat > vllm_ascend/_build_info.py << 'EOF'
__device_type__ = 'A5'
EOF
```

#### 2.1.2 `platform.py` — Preserve user `custom_ops:["none"]`

Find (~line 470):
```python
if get_ascend_device_type() != AscendDeviceType._310P:
    compilation_config.custom_ops = ["all"]
```
Change to:
```python
if get_ascend_device_type() != AscendDeviceType._310P:
    if compilation_config.custom_ops != ["none"]:
        compilation_config.custom_ops = ["all"]
```

> Reason: User-passed `custom_ops:["none"]` should not be overwritten. Triton kernels crash on A5.

#### 2.1.3 `ascend_forward_context.py` — A5 MoE comm forced ALLGATHER

Find A5 branch in `select_moe_comm_method`:
```diff
     elif soc_version in {AscendDeviceType.A5}:
-        if num_tokens <= mc2_tokens_capacity and vllm_config.parallel_config.world_size_across_dp > 1:
-            moe_comm_type = MoECommType.MC2
-        else:
-            moe_comm_type = MoECommType.ALLTOALL
+        moe_comm_type = MoECommType.ALLGATHER
```

> Reason: `npu_moe_distribute_dispatch_v2` (MC2) triggers NPU device error (561000) on A5. ALLTOALL also incompatible. ALLGATHER is the only viable MoE communication on A5. This affects both eager and graph mode (crashes in `_dummy_run` profile run).

#### 2.1.4 `device/device_op.py` — 2 modifications

**Mod 1: BaseDeviceAdaptor.reshape_and_cache — param name `slot_mapping` → `slot_indices`**

```python
class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache,
            slot_indices=slot_mapping
        )
```

> Reason: CANN 9.1 changed `_npu_reshape_and_cache` param name from `slot_mapping` to `slot_indices`.

**Mod 2: A5DeviceAdaptor.reshape_and_cache — manual index scatter replaces `npu_scatter_pa_kv_cache`**

```python
class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        block_size = key_cache.shape[1]
        slot_mapping_long = slot_mapping.long()
        block_indices = slot_mapping_long // block_size
        block_offsets = slot_mapping_long % block_size
        key = key.contiguous()
        value = value.contiguous()
        key_cache[block_indices, block_offsets] = key
        value_cache[block_indices, block_offsets] = value
```

> Reason: `npu_scatter_pa_kv_cache` triggers NZ 32-byte format constraint on A5 (arch35): "the last dim of key cache must be 32 Byte". Gemma4 head_dim=256 (fp16=512 bytes) violates this. Manual scatter via PyTorch index assignment bypasses this.

#### 2.1.5 `attention/attention_v1.py` — 3 modifications

**Mod 1: get_cudagraph_support — A5 returns NEVER**

```python
@classmethod
def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    if get_ascend_device_type() == AscendDeviceType.A5:
        return AttentionCGSupport.NEVER
    return AttentionCGSupport.ALWAYS
```

> `npu_fusion_attention` does not support aclgraph/cudagraph capture on A5.

**Mod 2: New `_forward_decode_via_fusion_attention` — A5 decode path**

```python
def _forward_decode_via_fusion_attention(self, query, attn_metadata, output):
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    if get_ascend_device_type() != AscendDeviceType.A5:
        return self.forward_paged_attention(query, attn_metadata, output)

    dense_key, dense_value = self._gather_paged_kv_to_dense(
        self.key_cache, self.value_cache,
        attn_metadata.block_tables, attn_metadata.seq_lens_list,
    )

    num_tokens = query.shape[0]
    actual_seq_lengths_kv = torch.tensor(attn_metadata.seq_lens_list, dtype=torch.int32).cumsum(0).tolist()
    actual_seq_lengths_q = [1] * len(attn_metadata.seq_lens_list)
    actual_seq_lengths_q_cumsum = torch.tensor(actual_seq_lengths_q, dtype=torch.int32).cumsum(0).tolist()

    sparse_mode = 4 if self.sliding_window is not None else 3 if attn_metadata.causal else 0
    pre_tokens = self.sliding_window if self.sliding_window is not None else SWA_INT_MAX
    next_tokens = 0

    attn_mask = attn_metadata.attn_mask
    if attn_mask is not None and attn_mask.dtype not in (torch.bool, torch.uint8):
        attn_mask = attn_mask.bool()

    attn_output = torch_npu.npu_fusion_attention(
        query=query[:num_tokens], key=dense_key, value=dense_value,
        head_num=self.num_heads, input_layout="TND",
        atten_mask=attn_mask, scale=self.scale,
        pre_tockens=pre_tokens, next_tockens=next_tokens,
        actual_seq_qlen=actual_seq_lengths_q_cumsum,
        actual_seq_kvlen=actual_seq_lengths_kv,
        sparse_mode=sparse_mode,
    )[0]
    output[:num_tokens] = attn_output[:num_tokens]
    return output
```

> A5 decode uses `npu_fusion_attention` (TND) + `_gather_paged_kv_to_dense` to gather paged KV into dense tensor. `attn_mask` must be bool (npu_fusion_attention only supports bool/uint8). This is required because `FIA_TND_SUPPORTED_HEAD_SIZES = {64, 128, 192}` does not include Gemma4's head_size=256 or global_head_dim=512, and `_npu_paged_attention` is not supported on A5.

**Mod 3: `forward_impl` — Restore complete dispatch**

```python
def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    num_tokens = query.shape[0]
    is_a5 = get_ascend_device_type() == AscendDeviceType.A5
    is_large_head = self._should_use_large_head_attention_fallback()

    if (self.kv_sharing_target_layer_name is not None
        and key is not None and value is not None
        and query.shape[0] == key.shape[0]
        and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)):
        shared_key, shared_value = self._get_current_token_shared_kv(attn_metadata)
        if shared_key is not None and shared_value is not None:
            return self._forward_large_head_prefill_attention(query, shared_key, shared_value, attn_metadata, output)

    if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        if is_a5:
            output = self._forward_decode_via_fusion_attention(query, attn_metadata, output)
        elif using_paged_attention(num_tokens, self.vllm_config) and self.sliding_window is None:
            output = self.forward_paged_attention(query, attn_metadata, output)
        elif is_large_head:
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
    elif (not _EXTRA_CTX.capturing and is_large_head
          and self.kv_sharing_target_layer_name is None
          and key is not None and value is not None
          and query.shape[0] == key.shape[0]
          and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)):
        output = self._forward_large_head_prefill_attention(query, key, value, attn_metadata, output)
    else:
        output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
    return output
```

> Restores kv_sharing and large_head branches (previously simplified/deleted causing A5 garbled output).

#### 2.1.6 `worker/block_table.py` — Pure PyTorch replaces triton kernel

Delete:
```python
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
```

Add at file top:
```python
def _compute_slot_mapping_pytorch(
    num_tokens, max_num_tokens, query_start_loc, positions, block_table,
    block_table_stride, block_size, slot_mapping,
    TOTAL_CP_WORLD_SIZE, TOTAL_CP_RANK, CP_KV_CACHE_INTERLEAVE_SIZE,
    PAD_ID, BLOCK_SIZE=1024,
) -> None:
    slot_mapping[:num_tokens] = PAD_ID
    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    num_reqs = query_start_loc.shape[0] - 1
    for req_idx in range(num_reqs):
        start_idx = query_start_loc[req_idx].item()
        end_idx = query_start_loc[req_idx + 1].item()
        if start_idx >= end_idx:
            continue
        req_positions = positions[start_idx:end_idx]
        block_indices = req_positions // virtual_block_size
        block_numbers = block_table[req_idx, block_indices.long()]
        virtual_block_offsets = req_positions - block_indices.long() * virtual_block_size
        if TOTAL_CP_WORLD_SIZE > 1:
            is_local = (
                (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE
                == TOTAL_CP_RANK
            )
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_ids = torch.where(is_local, slot_ids, PAD_ID)
        else:
            local_block_offsets = virtual_block_offsets % block_size
            slot_ids = block_numbers * block_size + local_block_offsets
        slot_mapping[start_idx:end_idx] = slot_ids.long()
    slot_mapping[num_tokens:] = PAD_ID

class _SlotMappingKernelWrapper:
    def __getitem__(self, grid):
        return _compute_slot_mapping_pytorch

_compute_slot_mapping_kernel = _SlotMappingKernelWrapper()
```

> Reason: Triton kernel compilation fails on A5 (MLIRCompilationError). Pure PyTorch replacement. `_SlotMappingKernelWrapper.__getitem__` returns the function ignoring grid, so `BlockTable.compute_slot_mapping` call `[_compute_slot_mapping_kernel[(num_reqs+1,)]](...)` works unchanged.

### 2.2 vllm package patches (2 modifications)

#### 2.2.1 `gemma4.py` — routing_function add `**kwargs`

File: `/usr/local/python3.11.14/lib/python3.11/site-packages/vllm/model_executor/models/gemma4.py`

Find `def routing_function(` (~line 332), add `**kwargs`:
```python
def routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
```

> Reason: vllm-ascend passes `num_experts` parameter; without `**kwargs` raises TypeError.

#### 2.2.2 gelu activation fix — `chunk(2, dim=-1)` → slicing + `approximate="tanh"`

In `Gemma4MoE` class:
```python
# Original:
gate_up = gate_up.chunk(2, dim=-1)
gate = torch.nn.functional.gelu(gate[0])

# Fixed:
gate_up = gate_up[..., :gate_up.shape[-1] // 2]
gate = torch.nn.functional.gelu(gate_up, approximate="tanh").contiguous()
```

> Reason: `chunk(2, dim=-1)` + default gelu produces non-contiguous output on A5. `approximate="tanh"` is the Gemma4 paper-specified gelu version.

### 2.3 torch package patch (1 modification)

#### 2.3.1 `_guards.py` — Delete ConstraintViolationError exception handler

File: `/usr/local/python3.11.14/lib/python3.11/site-packages/torch/_guards.py`

Find (~line 370-375) and delete these 5 lines:
```python
        except ConstraintViolationError:
            log.exception("Constraint violation:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise
```

> Reason: This exception class is undefined in CPU-only torch + torch-npu environment, causing NameError at runtime.

---

## 3. Problem Causal Chain (4 core problems)

```
Problem 1: npu_scatter_pa_kv_cache crash (NZ 32-byte constraint)
  → Fix: manual index scatter
  → Triggers Problem 2: FIA TND reads paged cache with large head_size → garbled output
    → Cause: forward_impl simplified, all decode goes FIA TND + block_table,
             but FIA_TND_SUPPORTED_HEAD_SIZES = {64, 128, 192},
             Gemma4 head_size=256 and global_head_dim=512 not in list
    → Fix: restore complete dispatch + add A5 decode fallback
    → Triggers Problem 3 (two sub-problems):
      3a: attn_mask dtype incompatible (DT_INT8)
        → Fix: convert attn_mask.bool() in _forward_decode_via_fusion_attention
      3b: A5 decode can't use _npu_paged_attention or FIA TND block_table
        → Fix: new _forward_decode_via_fusion_attention,
               using npu_fusion_attention (TND) + _gather_paged_kv_to_dense

Problem 4: graph mode npu_moe_distribute_dispatch_v2 crash (error 561000)
  → Cause: ascend_forward_context.py A5 MoE comm selection is MC2/ALLTOALL
  → Fix: A5 branch forces MoECommType.ALLGATHER
  → This affects both eager and graph mode (crashes in _dummy_run profile run)
```

### Full Problem List (18 problems)

| # | Problem | Symptom | Root Cause | Fix | Scope |
|---|---|---|---|---|---|
| 1 | CUDA runtime conflict | import torch_npu → torch crash | CUDA torch wheel, CUDA runtime conflicts with torch-npu | Install CPU-only torch wheel (`+cpu`) | Env setup |
| 2 | torch version rewritten | After vllm/vllm-ascend install torch becomes CUDA | pip dependency chain upgrades torch | All packages `--no-deps` install, check torch version after | Env setup |
| 3 | HCCL error code 4 | Distributed comm failure | CANN 9.0.0 incompatible with torch-npu 2.10.0 | Upgrade CANN to 9.1.T560, clear old versions | Env setup |
| 4 | Docker ENV residual old CANN | Runtime still loads old CANN libs | Docker ENV layer hardcoded 9.0.0/8.5.1 paths, docker commit doesn't modify ENV | `docker commit --change ENV` override all variables | Env setup |
| 5 | triton nvidia backend crash | `0 active drivers` error | nvidia/amd backends crash on NPU environment | Delete `triton/backends/nvidia/` and `amd/` | Env setup |
| 6 | MLIRCompilationError | Triton kernel fails on A5 | A5 doesn't support some triton MLIR ops | `custom_ops:["none"]` + block_table.py pure PyTorch replacement | Inference |
| 7 | torch._guards ConstraintViolationError | NameError runtime crash | Exception class undefined in CPU-only torch + torch-npu | Delete handler in `_guards.py` | Inference |
| 8 | TypeError: unexpected 'num_experts' | gemma4 routing_function doesn't accept num_experts | vllm-ascend passes `num_experts` but no `**kwargs` | Add `**kwargs` to routing_function | Inference |
| 9 | gelu output non-contiguous | `chunk(2, dim=-1)` + gelu → garbled output | A5 chunk result non-contiguous, default gelu not Gemma4 specified version | slicing + `approximate="tanh"` + `.contiguous()` | Precision |
| 10 | **npu_scatter_pa_kv_cache crash** | NZ 32-byte constraint "last dim must be 32 Byte" | Gemma4 head_dim=256 (fp16=512B) violates A5 NZ format | Manual index scatter: `key_cache[block_indices, block_offsets] = key` | **Core** |
| 11 | **FIA TND decode garbled** | Decode output like `s/s/s/` repetitive garbage | `FIA_TND_SUPPORTED_HEAD_SIZES={64,128,192}`, Gemma4 head_size=256 unsupported; forward_impl simplified deleted kv_sharing/large_head | Restore complete dispatch + A5 decode fallback | **Core** |
| 12 | **attn_mask dtype incompatible** | `invalid attenMask dtype[DT_INT8]` crash | `npu_fusion_attention` only supports bool/uint8 mask | Convert `attn_mask.bool()` in decode fallback | **Core** |
| 13 | **graph mode MC2 crash** | `npu_moe_distribute_dispatch_v2` error 561000 | A5 doesn't support MC2/ALLTOALL MoE communication | A5 branch forces `MoECommType.ALLGATHER` | **Core** |
| 14 | EngineCore zombie | Background process killed | Shell session timeout kills background process | Use `setsid nohup` to start | Startup |
| 15 | NPU device 6,7 zombie | ~106GB HBM occupied, new service can't allocate KV cache | Leftover processes not cleaned | Change to `ASCEND_RT_VISIBLE_DEVICES=0,1` | Startup |
| 16 | Squid proxy intercepts localhost | curl/evalscope requests intercepted | System has Squid proxy | curl add `--noproxy localhost`; evalscope add `os.environ['no_proxy']` | Testing |
| 17 | _npu_reshape_and_cache param name changed | slot_mapping param error | CANN 9.1 changed param name slot_mapping → slot_indices | Use `slot_indices=slot_mapping` in BaseDeviceAdaptor | Inference |
| 18 | _C_ascend has no attribute | `torch.ops._C_ascend.npu_moe_init_routing` not found | torch-npu 2.10.0 emptied `_C_ascend` namespace | Use `torch_npu.npu_moe_init_routing` instead | Inference |

---

## 4. Short Sequence Precision Issues

### 4.1 Short-Prompt Repetition / Garbled Output

Without system prompt + `temperature=0.0`, short prompts cause repetition/garbling. This is a **Gemma-4 inherent weakness** related to `<|channel>thought\n<channel|>` in the prompt suffix.

**Resolution**: Add system prompt + `temperature > 0.7`.

### 4.2 --reasoning-parser and --tool-call-parser Issues

Using `--reasoning-parser gemma4` or `--tool-call-parser gemma4` corrupts Chinese output because they force `skip_special_tokens=False`, leaking channel markers into decoded text.

### 4.3 Maxwell Equations Answer

Correct answer for Maxwell equations MCQ is **D** (not A). Previous "A" result was coincidental from wrong swiglu activation.

---

## 5. Graph Mode Limitation

Ascend950PR (A5) **does NOT support** graph capture for:
1. `npu_paged_attention` — PagedAttentionOperation setup fails
2. `aclnnFusedInferAttentionScoreV5` — error 161002 → 107033 during capture
3. `npu_moe_distribute_dispatch_v2` — error 561002 shape mismatch

**But**: With the ALLGATHER MoE fix + A5 decode fallback, graph mode **can work** via `_forward_decode_via_fusion_attention` which runs eagerly within the aclgraph wrapper (acl_graph.py line 115-122: `runtime_mode != self.runtime_mode` → direct runnable call).

---

## 6. Inference Service Startup

### 6.1 Eager Mode (26B MoE, basic verification)

```bash
source /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/set_env.sh

CUDA_VISIBLE_DEVICES="" \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  HCCL_OP_EXPANSION_MODE=AIV \
  HCCL_BUFFSIZE=256 \
  vllm serve /home/models/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 \
    --served-model-name gemma-4-26B-A4B-it \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enforce-eager \
    --max-model-len 10010 \
    --compilation-config '{"cudagraph_mode":"NONE","custom_ops":["none"]}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}'
```

Key params:
- `CUDA_VISIBLE_DEVICES=""` — mask CUDA
- `ASCEND_RT_VISIBLE_DEVICES=0,1` — select free NPU (adjust per environment)
- `--enforce-eager` — disable torch.compile and NPU Graph
- `--compilation-config '{"cudagraph_mode":"NONE","custom_ops":["none"]}'` — disable triton custom ops
- `--max-model-len 10010` — default 262144 causes excessive KV cache allocation

Must use `setsid nohup` to start, otherwise background process killed by shell timeout.

### 6.2 Graph Mode (26B MoE, DP4+TP2, 8 cards)

```bash
source /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/set_env.sh

CUDA_VISIBLE_DEVICES="" \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  HCCL_OP_EXPANSION_MODE=AIV \
  HCCL_BUFFSIZE=256 \
  setsid nohup vllm serve /home/models/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 \
    --served-model-name gemma-4-26B-A4B-it \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser functiongemma \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
    > /tmp/gemma_graph_run.log 2>&1 &
```

### 6.3 Process Cleanup

```bash
pkill -9 -f "vllm serve"
pkill -9 -f "VLLM::"
pkill -9 -f "launch_online_dp.py"
# Wait ~30s for NPU HBM to release before restarting
```

### 6.4 Manual Verification

```bash
curl --noproxy localhost http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26B-A4B-it","messages":[{"role":"user","content":"Hi"}],"max_tokens":50}'
```

**Important**: Use `--noproxy localhost` to bypass Squid proxy.

---

## 7. Error Logs

### 7.1 `npu_moe_distribute_dispatch_v2` crash (graph mode, 26B MoE)

Full log: `/home/tongpan/gemma/error.log`

```
RuntimeError: npu_moe_distribute_dispatch_v2:
  NPU function error: device error type 0, error code is 561002
EZ9999: The input of xActiveMask dim0 = 16 is not equal to x dim0 = 1024.
  [moe_distribute_dispatch_v2_tiling.cpp:350]
  params shape is invalid. [moe_distribute_dispatch_v2_tiling.cpp:1374]
  Tiling check param failed. [moe_distribute_dispatch_v2_tiling.cpp:1674]
```

Call chain: `profile_run` → `_dummy_run` → `_model_forward` → `acl_graph.__call__` → `npu_moe_distribute_dispatch_v2` FAIL → WorkerProc dies → EngineCoreProc fails → APIServer RuntimeError

Other operator errors observed:

**1. `aclnnFusedInferAttentionScoreV5` (FIA) graph capture failure**:
- Workspace mismatch: allocated 55664128 vs required 104871936 bytes
- Root error 161002 → cascading 107033 (`rtStreamEndCapture` failed, task group status error)
- Traceback: `forward_impl → forward_fused_infer_attention → full_graph_fia → aclnnFusedInferAttentionScoreV5 fails inside torch.npu.graph() capture → capture_end error 107033`
- This is the primary reason graph mode cannot work with FIA on A5

**2. `npu_paged_attention` not supported on A5**:
- `PagedAttentionOperation setup failed!` — A5 (soc_version=260) is `AscendDeviceType.A5`, `using_paged_attention()` returns False (attention/utils.py line 48)

**3. `copy_between_host_and_device_opapi` error 107030**:
- CPU→NPU `.to(device)` inside graph capture is not allowed; attempted during block_table shift workaround

**4. `svm_dbi_query_npage_size Invalid device (devid=1)`**:
- DRV-level NPU device state issue, likely residual from zombie processes

**Framework-level cascading failure sequence**:
`NPUModelRunner init failed` → `Engine core initialization failed` → `EngineCoreProc dies` → `APIServer RuntimeError`

### 7.2 Successful eager mode startup (31B Dense)

Full log: `/home/tongpan/gemma/gemma_graph_run_31.log`

Key milestones: weights 30.39 GB/TP rank, KV cache 66.87 GiB, 1,562,140 tokens, service on `http://0.0.0.0:8001`

---

## 8. Immutable Constraints

1. **torch must be CPU-only wheel** (`+cpu`) — CUDA runtime conflicts with torch-npu
2. **torch_npu must `--no-deps` install** — pip pulls non-existent CPU torch
3. **vllm/vllm-ascend/triton must `--no-deps` install** — prevents dependency chain upgrading torch
4. **CANN must be 9.1+** — torch-npu 2.10.0 requires, old versions (9.0.0/8.5.1) must be cleared
5. **Eager mode**: `--enforce-eager` + `custom_ops:["none"]` — triton compilation crashes on A5
6. **Startup must use `setsid nohup`** — background process killed by shell timeout otherwise
7. **A5 MoE comm must be ALLGATHER** — MC2/ALLTOALL crash (error 561000)
8. **A5 reshape_and_cache must be manual index scatter** — NZ 32-byte constraint vs head_dim=256
9. **BaseDeviceAdaptor.reshape_and_cache param name `slot_indices`** — CANN 9.1 change
10. **gemma4 routing_function must have `**kwargs`** — vllm-ascend passes `num_experts`
11. **A5 decode must use `npu_fusion_attention` + dense KV gather** — head_size=256/512 not in FIA_TND_SUPPORTED_HEAD_SIZES={64,128,192} and `_npu_paged_attention` unsupported
12. **curl must `--noproxy localhost`** — Squid proxy intercepts
13. **Docker ENV old CANN paths must be overridden with `docker commit --change ENV`**

---

## 9. Git Commit History

| Commit | Description | Files |
|--------|-------------|-------|
| c540a81b | [Feature] Support Gemma4 inference on Ascend (PR #9222 cherry-pick) | Multiple |
| 5075b1ce | Simplify forward_impl: remove kv_sharing and large_head branches | attention_v1.py |
| fb7dd11a | A5: bypass npu_scatter_pa_kv_cache and add decode fusion attention fallback | device_op.py, attention_v1.py |
| d4252c39 | A5: force MoE comm to ALLGATHER (MC2/ALLTOALL crash) | ascend_forward_context.py |

Branch: `pr-9222-gemma4-support`, repo: `0moyi0-2024/vllm-ascend_tp`, remote HEAD: `d4252c39`

---

## 10. Model Key Parameters

| Parameter | Value |
|-----------|-------|
| Model | gemma-4-26B-A4B-it |
| Total params | 26B (MoE, 128 experts, top_k=8) |
| Active params | ~4B per token |
| head_dim (local) | 256 |
| global_head_dim | 512 |
| num_kv_heads (local) | 8 |
| num_kv_heads (global) | 2 |
| num_experts | 128 |
| top_k | 8 |
| sliding_window | 1024 |
| Activation | gelu_tanh |
| FIA_TND_SUPPORTED_HEAD_SIZES | {64, 128, 192} (**excludes 256/512**) |

> head_size=256 and global_head_dim=512 not in FIA_TND_SUPPORTED_HEAD_SIZES — this is the fundamental reason A5 decode must use `_forward_decode_via_fusion_attention` + dense KV gather.

---

## 11. Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| torch version | `python3 -c "import torch; print(torch.__version__)"` | `2.10.0+cpu` |
| torch-npu | `python3 -c "import torch_npu; print(torch_npu.__version__)"` | `2.10.0` |
| vllm | `python3 -c "import vllm; print(vllm.__version__)"` | `0.20.2` |
| device_type | `python3 -c "from vllm_ascend._build_info import __device_type__; print(__device_type__)"` | `A5` |
| CANN 9.1 | `ls /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/bin/ascendc` | exists |
| CANN old cleared | `ls /usr/local/Ascend/cann-9.0.0` | not exists |
| NPU visible | `npu-smi info` | Ascend950PR |
| Service started | `grep "Application startup complete" /tmp/gemma_run.log` | exists |
| Inference response | `curl --noproxy localhost http://127.0.0.1:8000/v1/chat/completions -d '{"model":"gemma-4-26B-A4B-it","messages":[{"role":"user","content":"Hi"}],"max_tokens":50}'` | JSON with `"choices"` |