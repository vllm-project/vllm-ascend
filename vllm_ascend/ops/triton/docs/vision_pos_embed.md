# Vision position embedding interpolation

## Description

- **Function**: Interpolate Qwen's learned visual position table, repeat each spatial grid for every temporal frame, and emit spatial merge block order. Production caller: the existing `patch_qwen3vl.py` worker patch.
- **Formula**: Four neighboring table entries multiplied by bilinear weights, summed per hidden channel. Coordinates span `[0, num_grid_per_side - 1]` inclusively; a singleton dimension uses coordinate zero.
- **Algorithm flow**: Flatten output into fixed 1024-element tiles; recover merge block row/column; compute clipped neighbors and weights; gather four entries; round each product to the table dtype; accumulate in FP32; store in output dtype. Time frames reuse spatial coordinates. No weight/result caching or input mutation.
- **Supported modes**: Intended for Triton-capable Atlas A2, A3 and 950PR/950DT. Hardware compilation, accuracy and performance are **unvalidated**. 310P retains native execution. Eager metadata preparation and metadata generation outside encoder graph replay use the same wrapper.

## Parameters

All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `embed_weight` | Input | Learned table, `[side*side, D]` | FP32, FP16, BF16 | Contiguous NPU ND |
| `t`, `h`, `w` | Attribute | Temporal and spatial grid dimensions | Positive Python int | Scalar |
| `num_grid_per_side` | Attribute | Table side length | Positive Python int | Scalar |
| `m_size` | Attribute | Spatial merge size | Positive Python int | Scalar |
| `dtype` | Attribute | Output dtype, must equal table dtype | torch.dtype | N/A |
| return | Output | `[t*h*w, D]`, merge block order per frame | Same as table | Contiguous NPU ND |

## Constraints

- Both spatial dimensions must be divisible by `m_size`. `D > 0`; table rows must equal the square of the supplied side length. Flattened input and output indexing must fit int32.
- The model dispatch retains upstream native for absent Triton, 310P, non-NPU/noncontiguous tables, unsupported dtypes or differing input/output dtype. Invalid grids/table dimensions on the fused path raise explicitly; compilation failures are not swallowed.
- Coordinate operations and weights use FP32. Multipliers and products are explicitly rounded to the output dtype before FP32 summation, matching native's in-place product/reduction structure. FP fusion is disabled. Bit identity is not promised.
- Fixed tile size bounds vector storage independently of `D`; this initial launch policy is not hardware tuned. There is no new custom-op registration because this eager metadata wrapper is called outside compiled encoder forward.
- `NUM_GRID`, `MERGE`, `HIDDEN`, tile size and pointer dtype specialize JIT. `h`, `w`, total elements and coordinate scales use `do_not_specialize`; grid contents alone should not create variants. Backend specialization/cache behavior still needs NPU observation. No explicit warmup registry is added; the first metadata call may compile before encoder capture. Compilation counts and serving latency remain a merge gate.

## Origin and Differences

- **Origin**: [vLLM #37948](https://github.com/vllm-project/vllm/pull/37948), `qwen3_vl.py`. Reference used: vLLM commit `acb87186fa011e273f2d77af4a0517664e2c35a2`; Ascend baseline `9b7a557f2aba8e4e5bd9f1ebccc7d92b86f32c28`.
- **Differences**: Bounded flattened element tiles instead of one GPU program per patch; endpoint-based second-half coordinates; clip lower and upper neighbors; explicit low-precision product rounding. Retains the existing Ascend patch and native fallback. Remove the patch when upstream offers a validated Ascend implementation/backend hook.
- **Scope**: The pinned Qwen3-VL dense/MoE and Qwen3.5 dense/MoE instantiate the shared `Qwen3_VisionTransformer`; Ascend Qwen3.5 wrappers inherit that visual tower. This does not optimize pure text models, RoPE, or image resizing.

## Test Cases

CPU dispatch and contract checks, independent of the full plugin import environment:

```bash
python tests/ut/ops/test_vision_pos_embed.py
```

Real Ascend operator tests (FP32/FP16/BF16; singleton, rectangular, temporal, real tower dimensions, tail masks, exact merge reorder, endpoint, multi-grid concatenation, changed weights and metadata consumer graph replay):

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_vision_pos_embed.py
```

The oracle is upstream `pos_embed_interpolate_native`, independent of the fused coordinate/index implementation. Accuracy gates use the repository criteria: 99% element tolerance plus maximum absolute error, separately for each dtype. These are proposed acceptance gates, not measured NPU results. The graph test captures a metadata consumer; it does not establish full visual tower/encoder ACLGraph correctness.

Ascend operator A/B, with warmup, alternating execution order and first-call cost reported separately:

```bash
python tools/benchmark_vision_pos_embed.py --grid 1 32 32 --dtype bfloat16
python tools/benchmark_vision_pos_embed.py \
    --weight-file /models/Qwen3-VL/model-00001-of-00005.safetensors \
    --weight-key model.visual.pos_embed.weight --grid 2 24 40
```

Use the actual shard and key from the checkpoint index. The optional real table checks position interpolation only. Event intervals can include idle gaps from host launches; use an NPU profiler for isolated kernel duration. Host wall results include synchronization and allocation. Neither metric establishes model throughput.

Still required before merge: real-checkpoint visual tower outputs and downstream quality, eager and full encoder ACLGraph replay under alternating sizes, profiler kernel duration, metadata wall time, encoder images/s or tokens/s, serving output tokens/s and TTFT, and startup/serving compilation counts. Compare native/fused under identical model, software, graph setting, image grids, batch/concurrency and cache hit rate, alternating A/B runs. Fully cached encoder workloads skip this operation. No GPU measurements are Ascend evidence.
