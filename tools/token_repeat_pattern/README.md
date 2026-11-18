# token_repeat pattern backend

Sources live in-tree and build with vllm-ascend:

`vllm_ascend/observability/runtime_guard/native_pattern/`

Install / editable install of `vllm-ascend` compiles `_pattern_native` automatically
(pure CPU pybind11; does not require CANN custom kernels). No separate
`pip install` step is needed.
