# MoE Experts Analysis

This document analyzes the current implementation capabilities of vLLM Ascend at the MoE `experts` layer, focusing on the structural assumptions, weight layout, grouped matmul path, quantified expert path, and boundaries with dispatch/combine of expert MLP.

## 1. What problem does this layer solve?

The experts layer currently mainly solves:

- Is expert internally a standard two-stage MLP?
- How `w13/gate_up` and `w2/down` are organized
- How to perform grouped matmul
- How to access quantized experts

## 2. Overview of current capabilities

The current expert computing center is at:

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py)

The key features are:

- unquant path coexists with quant path
- The first paragraph is usually `gate_up`
- The middle is usually `swiglu`
- The second paragraph is usually `down`

Therefore, the current most mature capability model at the expert level is expert MLP, which is common in standard routed MoE LLM.

## 3. Key capabilities currently implemented

### 3.1 The current default expert is a two-stage MLP

Existing implementations explicitly revolve around:

- `w13_weight` / `gate_up`
- `w2_weight` / `down`

Organize expert weights and calculation order.

You can see it directly in `moe_mlp.py`:

- grouped matmul for `gate_up`
- `swiglu`
- grouped matmul for `down`

See:

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:353)

### 3.2 `gate_proj + up_proj -> w13` is an existing agreement

Whether it is quantitative mapping or expert calculation, the following convention appears repeatedly in the current warehouse:

- `gate_proj`
- `up_proj`
- `down_proj`
- Pack the first two into `w13` before MoE execution

This can be seen in `modelslim_config.py`, `w4a8.py`, `moe_mlp.py`.

Therefore, the current strong capability area of ​​expert calculation is the model that can fall into the `w13 + w2` layout.

### 3.3 grouped matmul is the main execution method of expert

The current main path of unquant expert has a lot of dependencies:

- `torch_npu.npu_grouped_matmul`

See:

- [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:371)

This means that the current experts layer is more like a "batch-expert parallel matmul pipeline" rather than an expert-by-expert for-loop.

### 3.4 quantized experts have become a formal capability

The current quantized MoE expert is not blank. The warehouse has registered multiple types of quantized fused-MoE methods, for example:

- `AscendW8A8DynamicFusedMoEMethod`
- `AscendW4A8DynamicFusedMoEMethod`
- `AscendW4A16FusedMoEMethod`
- MXFP / FP8 variant

See:

- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py:45)

This shows that the current expert layer is not only the functional path of BF16/FP16, but also assumes the formal execution responsibilities of quantitative experts.

## 4. Current structural assumptions

The current experts layer implies these assumptions:

- expert MLP can be reduced to `gate_up -> activation -> down`
- The weight layout should preferably fall to `w13 + w2`
- Multi-expert calculations should be merged through grouped matmul as much as possible
- quantized expert should still follow roughly the same structural contract

## 5. Known boundaries and risks

The current main boundaries are:

- Multi-branch expert, convolution expert, and residual expert do not naturally fall into the current main path
- Expert If it is not a two-stage MLP, the reuse value of the existing implementation will be significantly reduced.
- The loading and execution coupling of quantized expert is strong
- Problems at the experts layer often appear together with shared expert, router, and dispatch and cannot be viewed in isolation.

## 6. What to look for when analyzing this layer

It is recommended to give priority to:

- Whether expert is `gate_proj/up_proj/down_proj` internally
- Whether checkpoint can be reduced to `w13/w2`
- Whether to rely on grouped matmul
- Whether it is necessary to quantify the expert path

## 7. Related code

- [ops/fused_moe/moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py)
- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
