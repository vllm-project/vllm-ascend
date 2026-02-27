# Sequence Parallelism

## What is Sequence Parallelism

Sequence Parallelism (SP) was first introduced in [Megatron](https://arxiv.org/pdf/2205.05198), with the original intention of reducing training activation memory. The core modification was changing `Allreduce->LayerNorm` to `ReduceScatter->LayerNorm->Allgather`. This technique was later applied to inference by vllm. It should be noted that splitting Allreduce into ReduceScatter and Allgather does not inherently bring performance benefits; it reduces the computation load of LayerNorm, but this gain is minimal. The real benefits of SP come from:

1. LLM inference deployment often uses quantization. Taking INT8 quantization commonly used on NPUs as an example, after LayerNorm, a Quant operator quantizes the hidden states from BF16 to INT8. The communication volume of Allgather is halved, and the time consumption is almost halved.
2. ReduceScatter and Allgather can be fused with the preceding and following Matmul operations respectively into communication-computation parallel operators, reducing latency.

## How to Use

Currently, vllm-ascend has implemented Sequence Parallelism for VL-class models based on the Inductor pass. It can be enabled in the following way:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true}}' \
    --additional_config={"sp_threshold": 1000}
```

- `"pass_config": {"enable_sp": true}`: This is the switch for SP. Since SP relies on graph mode, it must be enabled and is not supported in eager mode.
- `--additional_config={"sp_threshold": 1000}`: Based on our experiments, when the number of tokens is small (empirical value is less than 1000), SP can actually bring negative benefits. This is because when the communication volume is small, the fixed overhead of the communication operator becomes the dominant factor. Therefore, when one communication operator (Allreduce) is split into two communication operators (ReduceScatter+Allgather), the end-to-end latency often becomes longer. Thus, we have reserved the `sp_threshold`parameter; SP will only take effect when `num_tokens >= sp_threshold`. **The default value is 1000, which generally does not need to be modified.** `sp_threshold` will be appended into `compile_ranges_split_points`, which is a parameter provided by vllm that splits the graph compilation range `[1, max_num_batched_tokens]` into `{[1, split_points[0]], [split_points[0] + 1, split_points[1]], ..., [split_points[-1] + 1, max_num_batched_tokens]}`, and sequentially checks whether the `is_applicable_for_range` of the pass returns `True`.

Without modifying `sp_threshold`, the simplest way and recommended way to enable SP is:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true}}'
```

## Difference Between SP and Flash Comm V1

[Flash Comm V1 (FC1)](https://gitcode.com/ascend-tribe/ascend-inference-cluster/blob/main/FlashComm/ascend-inference-cluster-flashcomm.md) is an enhanced version of Sequence Parallelism developed based on NPU. The enhancements include:

1. For models using the MLA structure, Allgather is postponed until after QKV projection, further reducing communication volume.
2. For MoE models, Allgather is postponed until after Gating+DynamicQuant, also aiming to reduce communication volume.

FC1 is a unique optimization in vllm-ascend, currently implemented based on Custom OP, but it is difficult to support VL-class models (reasons detailed in [[RFC]: support sequence parallelism by pass](https://github.com/vllm-project/vllm-ascend/issues/5712) ). Therefore, currently FC1 and SP are complementary.

## Support Matrix

### Without Quantization

|                      | VL + Dense | VL + MoE | non-VL + Dense | non-VL + MoE |
| -------------------- | ---------- | -------- | -------------- | ------------ |
| Sequence Parallelism | graph      | x        | x              | x            |
| Flash Comm V1        | x          | x        | eager/graph    | eager/graph  |

### With Quantization

SP currently does not support quantization and is under adaptation.

|                      | VL + Dense | VL + MoE | non-VL + Dense | non-VL + MoE |
| -------------------- | ---------- | -------- | -------------- | ------------ |
| Sequence Parallelism | x          | x        | x              | x            |
| Flash Comm V1        | x          | x        | eager/graph    | eager/graph  |
