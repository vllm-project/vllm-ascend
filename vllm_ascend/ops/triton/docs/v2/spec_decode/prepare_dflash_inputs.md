# prepare_dflash_inputs

## Description

- **Function**: Prepares the metadata and fixed-capacity buffers required by DFlash speculative decoding on Ascend NPU. For each request, the operator converts the current target-model token span into draft-model context KV positions/slots, constructs the next DFlash query tokens and query slots, builds sampling mappings, copies per-request sampling state, and pads graph-visible buffers.
- **Supported vLLM ABI**: This optimized Ascend implementation targets the DFlash `prepare_dflash_inputs` ABI in vLLM 0.27.1, which is the version currently adapted by vLLM-Ascend for this path.
- **Implementation**:
    - `vllm_ascend/ops/triton/v2/spec_decode/prepare_dflash_inputs.py` contains the Triton kernel `_prepare_dflash_inputs_kernel` and launcher `prepare_dflash_inputs_triton`.
    - `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py` provides `prepare_dflash_inputs` with the vLLM 0.27.1 ABI and forwards to the optimized launcher.
- **Formula**:
    - Request context range:
      `ctx_start = query_start_loc[req]`,
      `ctx_end = query_start_loc[req + 1]`,
      `valid_ctx_end = ctx_end - num_rejected[req]`,
      `last_valid_pos = positions[valid_ctx_end - 1]`.
    - Context/query KV slot:
      `logical_block = min(position // block_size, block_table_stride - 1)`,
      `physical_block = block_table[req, logical_block]`,
      `slot = physical_block * block_size + position % block_size`.
    - Query construction:
      `query_pos = last_valid_pos + 1 + query_offset`;
      query offset `0` uses the request bonus token, and subsequent offsets use
      `parallel_drafting_token_id`.
    - Bonus token:
      `last_sampled[req_state_idx]` when `num_sampled[req] > 0`, otherwise
      `next_prefill_tokens[req_state_idx]`.
    - Sampling mapping:
      the vLLM 0.27.1 DFlash path uses `sample_from_anchor=False`,
      `num_query_per_req = num_speculative_steps + 1`,
      and sample row `s` maps to query offset `s + 1`.
- **Algorithm flow**:
  1. Read `input_batch.num_reqs` and the maximum scheduled context length.
  2. Read the Ascend VectorCore count and choose `workers_per_req` as the maximum of:
     - the target VectorCore parallelism per request;
     - the minimum workers needed to keep Context work within 256 elements per worker;
     - the minimum workers needed to keep Query work within 16 elements per worker;
     - the minimum workers needed to keep Sample work within 16 elements per worker.
  3. Launch a 2-D grid `(num_reqs, workers_per_req)`.
  4. Split each request's Context range across request-local workers with quotient/remainder balancing. Each worker processes one contiguous range and writes `context_positions` and `context_slot_mapping`.
  5. Independently split `num_query_per_req` across the same workers. Worker 0 owns query offset 0 and loads the bonus token; all other query rows use `parallel_drafting_token_id`.
  6. Independently split `num_speculative_steps` across the same workers and write `sample_indices`, `sample_pos`, and `sample_idx_mapping`.
  7. Worker 0 writes the per-request scalar outputs `query_start_loc`, `seq_lens`, `temperature`, and `seeds`.
  8. Flatten the complete launch grid and quotient/remainder-balance graph-safety padding for `query_start_loc`, `seq_lens`, sample buffers, and `query_slot_mapping`.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950

## Parameters

> [!NOTE]
>
> The parameters below describe the vLLM 0.27.1 DFlash ABI supported by this optimized implementation.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input_buffers` | Output | Preallocated DFlash query buffers containing `input_ids`, `positions`, `query_start_loc`, and `seq_lens`, written in place | `InputBuffers` | N/A |
| `query_slot_mapping` | Output | Physical KV slot for each DFlash query token; graph padding uses `PAD_SLOT_ID` | int32 | ND |
| `context_positions` | Output | Absolute target positions used for DFlash context KV precompute | int64 | ND |
| `context_slot_mapping` | Output | Physical KV slot for each target context token | int32 | ND |
| `sample_indices` | Output | Flattened query-row indices selected from DFlash hidden states for sampling | int64 | ND |
| `sample_pos` | Output | Absolute position corresponding to each speculative sample | int64 | ND |
| `sample_idx_mapping` | Output | Maps each speculative sample to its persistent request-state index; padded entries are `-1` | int32 | ND |
| `temperature` | Output | Per-request-state sampling temperatures copied from `input_temperature` | float32 | ND |
| `seeds` | Output | Per-request-state sampling seeds copied from `input_seeds` | int64 | ND |
| `input_batch` | Input | Target-batch metadata; the launcher reads `num_reqs`, `num_scheduled_tokens`, `positions`, `query_start_loc`, and `idx_mapping` | `InputBatch` | N/A |
| `num_sampled` | Input | Number/state flag of sampled tokens per active request; selects `last_sampled` or `next_prefill_tokens` | int32 | ND |
| `num_rejected` | Input | Number of rejected tokens at the end of each request's current target span | int32 | ND |
| `last_sampled` | Input | Persistent last-sampled token ID per request-state slot | int64 | ND |
| `next_prefill_tokens` | Input | Persistent next-prefill token ID per request-state slot, used when `num_sampled == 0` | int32 | ND |
| `input_temperature` | Input | Source sampling temperature per request-state slot | float32 | ND |
| `input_seeds` | Input | Source sampling seed per request-state slot | int64 | ND |
| `block_table` | Input | Request-to-physical-KV-block table, shape `[max_num_reqs, max_num_blocks]` | int32 | ND |
| `block_size` | Input/Attribute | Number of tokens in one KV block | int | scalar |
| `parallel_drafting_token_id` | Input/Attribute | Token ID written to non-anchor DFlash query positions | int | scalar |
| `num_query_per_req` | Input/Attribute | Number of DFlash query rows produced per request | int | scalar |
| `num_speculative_steps` | Input/Attribute | Number of speculative sample rows produced per request | int | scalar |
| `max_num_reqs` | Input/Attribute | Fixed graph-visible request capacity used for output padding | int | scalar |
| `max_num_tokens` | Input/Attribute | Fixed graph-visible token capacity used for query/output padding | int | scalar |
| `max_model_len` | Input/Attribute | Maximum model sequence length; query positions are clamped to `max_model_len - 1` | int | scalar |
| `sample_from_anchor` | Input/Attribute | Whether query offset 0 is sampled; the vLLM 0.27.1 DFlash integration uses `False` | bool | scalar |

## Constraints

- The operator is inference-only and requires Ascend NPU Triton execution.
- This optimized implementation targets the vLLM 0.27.1 `prepare_dflash_inputs` ABI.
- `input_batch.num_reqs > 0` and `input_batch.num_reqs <= max_num_reqs`.
- `input_batch.query_start_loc` contains at least `num_reqs + 1` int32 entries, is non-decreasing, and delimits the flattened `input_batch.positions` tensor.
- `input_batch.positions` is int64. Every position used for KV lookup must resolve to a valid logical block; the implementation clamps the logical block index to `block_table_stride - 1`.
- `input_batch.idx_mapping` contains at least `num_reqs` request-state indices, and each active value must index a valid entry in the persistent request-state buffers.
- `input_batch.num_scheduled_tokens` contains one host-side scheduled-token count per request. Its maximum is used by the launcher to select Context parallelism and `BLOCK_SIZE`.
- `num_sampled` and `num_rejected` are int32 and contain at least `num_reqs` entries. For every request, `0 <= num_rejected[req] < ctx_len[req]`.
- `last_sampled`, `next_prefill_tokens`, `input_temperature`, and `input_seeds` contain at least `max_num_reqs` request-state entries.
- `block_table` is int32 with shape `[max_num_reqs, max_num_blocks]`; `block_size > 0`.
- `query_slot_mapping` and `context_slot_mapping` are int32. Query/context positions and `sample_indices`/`sample_pos` are int64. `sample_idx_mapping` is int32.
- The vLLM 0.27.1 DFlash path uses `sample_from_anchor=False` and `num_query_per_req == num_speculative_steps + 1`.
- `num_reqs * num_query_per_req <= max_num_tokens`.
- The launcher uses fixed vector widths of 256 for Context and 16 for Query/Sample. `workers_per_req` is increased when necessary so every request-local range fits these widths.
- After `workers_per_req` is selected, `BLOCK_SIZE = min(256, next_power_of_2(ceil(max_target_query_len / workers_per_req)))`.
- The operator writes graph-safety padding for fixed-capacity request, sample, and query-slot buffers and is compatible with graph replay/capture paths that consume those buffers.
- The operator performs integer indexing and direct data movement only; all tested outputs are required to be bit-exact.

## Upstream Compatibility

This optimization intentionally follows the DFlash `prepare_dflash_inputs` behavior and ABI used by vLLM 0.27.1, which is the version currently adapted by vLLM-Ascend for this path.

Upstream vLLM later changed the shared DFlash-family input-preparation ABI in `vllm-project/vllm#52188` while adding DSpark support with Decode Context Parallelism (DCP). That interface adds `cp_rank`, `cp_size`, and `cp_interleave` and requires additional functional adaptation beyond the vLLM 0.27.1 path.

The newer upstream ABI is not implemented by this optimized Ascend path. `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py` keeps an explicit compatibility entry for that signature and raises `NotImplementedError`. Support can be added in the future if the corresponding DFlash/DSpark DCP scenario is required by vLLM-Ascend.

## Origin and Differences

- **Origin**: Optimized from the vLLM 0.27.1 DFlash `prepare_dflash_inputs` / `_prepare_dflash_inputs_kernel` behavior already adapted in vLLM-Ascend.
- **Differences**:
    - NPU adaptation for performance: replaces the previous one-effective-program-per-request scalar execution with a VectorCore-aware 2-D launch.
    - Context, Query, and Sample domains are independently partitioned with quotient/remainder balancing.
    - Graph-padding work is distributed across the complete launch grid instead of being serialized by a single request/program.
    - `workers_per_req` is derived from the detected VectorCore count and the Context/Query/Sample vector-width requirements rather than a hard-coded device core count.
    - The optimization does not extend the supported vLLM ABI or add new DFlash/DSpark functionality; non-0.27.1 upstream ABI adaptation is outside the scope of this implementation.

## Test Cases

The accuracy test covers the captured vLLM 0.27.1 DFlash inference shapes plus branch-specific cases:

- `num_reqs=8`, `target_positions.shape=[1626]`, `num_query_per_req=9`, `num_speculative_steps=8`, `max_num_reqs=64`, `max_num_tokens=8192`, `block_size=128`;
- `num_reqs=64`, `target_positions.shape=[576]`, `num_query_per_req=9`, `num_speculative_steps=8`, `max_num_reqs=64`, `max_num_tokens=8192`, `block_size=128`;
- chunk-prefill path with `num_sampled == 0`, validating that query offset 0 uses `next_prefill_tokens`;
- rejected-context path validating that `num_rejected` changes `last_valid_pos` and therefore the generated Query/Sample positions.

The test compares Context mapping, Query construction, Sample mapping, per-request sampling state, and all graph-padding regions against an independent Python reference. Since this operator performs integer indexing and direct state copies, the unified precision requirement is bit-exact (`rtol=0, atol=0`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_dflash_inputs.py
```

## Example

The optimized launcher is used through the vLLM 0.27.1-compatible worker wrapper:

```python
prepare_dflash_inputs_triton(
    input_buffers,
    query_slot_mapping,
    context_positions,
    context_slot_mapping,
    sample_indices,
    sample_pos,
    sample_idx_mapping,
    temperature,
    seeds,
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    input_temperature,
    input_seeds,
    block_table,
    block_size=128,
    parallel_drafting_token_id=151669,
    num_query_per_req=9,
    num_speculative_steps=8,
    max_num_reqs=64,
    max_num_tokens=8192,
    max_model_len=8192,
    sample_from_anchor=False,
)
```
