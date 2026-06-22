# Positional / RoPE Analysis

This document analyzes vLLM Ascend's current implementation capabilities in positional encoding and RoPE layers, focusing on rotary embedding type coverage, cos/sin cache management, partial rope, mrope, and collaboration with model runner/attention backend.

## 1. What problem does this layer solve?

The RoPE layer currently not only "rotates q/k", but also undertakes:

- Select specific rotary implementation
- Manage cos/sin cache
- Handle partial rotary / interleaved rotary
- Prepare location data for paths such as MLA / MRoPE / XD-RoPE
- Synchronized with model runner positions

This layer connects the "position encoding definition in the model structure" and the "input form required for attention at runtime".

## 2. Overview of current capabilities

Ascend currently has several rotary implementations registered as OOT custom ops:

- `RotaryEmbedding -> AscendRotaryEmbedding`
- `MRotaryEmbedding -> AscendMRotaryEmbedding`
- `YaRNScalingRotaryEmbedding -> AscendYaRNRotaryEmbedding`
- `DeepseekScalingRotaryEmbedding -> AscendDeepseekScalingRotaryEmbedding`
- `ApplyRotaryEmb -> AscendApplyRotaryEmb`

For the registration entrance, see:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:688)

This illustrates that RoPE is not a single implementation, but a cluster of capabilities that has been bifurcated by model type.

## 3. Key capabilities currently implemented

### 3.1 cos/sin cache has been explicitly split and managed

Maintained in `vllm_ascend/ops/rotary_embedding.py`:

- `_cos_sin_cache`
- `_cos_cache`
- `_sin_cache`
- `_cos_slice`
- `_sin_slice`
- `_cos_mla`
- `_sin_mla`

And provide:

- `set_cos_and_sin(...)`
- `update_cos_sin(...)`
- `get_cos_and_sin_slice()`
- `get_cos_and_sin_mla(...)`

See:

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:42)

This reflects a core fact of the current implementation: many operators on the Ascend side are more suitable for directly consuming the split cos/sin, rather than reusing the unified cache expression of upstream.

### 3.2 The standard GQA path is separate from the MLA path

Explicit differentiation is currently implemented:

- Standard GQA/normal decoder-only model
- MLA model

In `set_cos_and_sin(...)`:

- MLA goes `_cos_mla/_sin_mla`
- Take the normal path that is not VL and has rope `_cos/_sin`

见：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:62)

Therefore the current RoPE layer is not a unified abstraction; different attention subtypes are already branched here.

### 3.3 partial rope has been considered

For normal RoPE paths, the current implementation adjusts rotary_dim based on the model configuration:

- `partial_rotary_factor`
- `rotary_dim`

See:

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:79)

This shows that the current implementation has recognized that "rope_dim can be smaller than head_dim" and specifically handles the scenario of only rotating the first half and passthrough the second half in `rope_forward_oot(...)`.

### 3.4 Triton coexists with NPU native rotary

The execution path of `rope_forward_oot(...)` is:

- Priority when Triton is available `rope_forward_triton(...)`
- Otherwise go `torch_npu._npu_rotary_embedding`

See:

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:153)
- [ops/triton/rope.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton/rope.py)

Currently Triton rope explicitly supports:

- `rope_dim != head_dim`
- neox / non-neox two styles

But when entering NPU fallback, `offsets` still does not support batched rotary.

### 3.5 YaRN, DeepSeek scaling, and MRoPE have all entered the official path

In addition to ordinary `RotaryEmbedding`, the current warehouse also explicitly supports:

- `YaRNScalingRotaryEmbedding`
- `DeepseekScalingRotaryEmbedding`
- `MRotaryEmbedding`
- More specialized DSV4 rope state management in `rope_dsv4.py`

Related entrances:

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:253)
- [rope_dsv4.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rope_dsv4.py)

Therefore, the current RoPE capability boundary does not only cover "standard Llama-style rotary", but has been extended to more complex scaling and grouped rope forms.

### 3.6 model runner has been responsible for position synchronization

RoPE is not just an op layer issue, `model_runner_v1.py` has taken over:

- `uses_mrope`
- `uses_xdrope_dim`
- `_calc_mrope_positions`
- `_calc_xdrope_positions`
- positions copied to GPU

See:

- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:964)

This shows that the current implementation treats "location metadata" as runner-level state, rather than local state within the layer.

## 4. Current structural assumptions

The structural assumptions implicit in the current RoPE layer include:

- rotary still mainly works on q/k
- cos/sin cache can be prepared in advance and reused
- partial rope can be processed separately through rotary_dim/head_dim
- MRoPE/XD-RoPE requires runner preprocessing positions
- MLA is different from the rope data path of ordinary GQA/decoder path

## 5. Known boundaries and risks

Currently clearly visible boundaries are:

- batched rotary `offsets` is still not supported under NPU path
- The rope cache expression of different models may be different, and the Ascend side relies heavily on cache preprocessing.
- The RoPE layer has strong coupling with the attention backend, especially MLA / SFA / DSA
- Explicit synchronization of positions/cos/sin is required in graph mode, and metadata cannot be changed at will during runtime.

## 6. What to look for when analyzing this layer

It is recommended to give priority to:

- `rope_type`
- `rope_theta`
- `partial_rotary_factor`
- `rotary_dim`
- `qk_rope_head_dim`
- Whether to use `mrope` / `xdrope`
- Are positions generated within the layer or pre-generated by the runner?

## 7. Related code

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
- [rope_dsv4.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rope_dsv4.py)
- [ops/triton/rope.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton/rope.py)
- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py)
