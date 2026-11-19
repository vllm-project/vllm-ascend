# token_repeat pattern backend

This `tools/token_repeat_pattern/` directory is documentation / helper notes only.
Runtime sources live in-tree and build with vllm-ascend:

`vllm_ascend/observability/runtime_guard/native_pattern/`

Install / editable install of `vllm-ascend` compiles `_pattern_native` automatically
(pure CPU pybind11; does not require CANN custom kernels). No separate
`pip install` package is needed.

User-facing docs: `docs/source/user_guide/configuration/runtime_config.md`
(`### token_repeat`). Alert thresholds: `native_pattern/pattern.json` (read once at init).
