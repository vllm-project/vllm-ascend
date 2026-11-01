# GLM-5.3-Flash weight-transfer fixture

Adapted from [PR #19828](https://github.com/vllm-project/vllm-ascend/pull/19828),
commit `b979486b166647b2317fedfe1f18ac3d4279d124`.

`config.original.json` and `provenance.json` record the published configuration
and checkpoint-header sources. `config.json` retains the text model's production
widths, the first four layers (three KDA layers and one sparse-attention MoE
layer), and eight routed experts. Quantization and MTP are disabled.
`parameter_metadata.json` contains checkpoint names and shapes, not weights.
The weight source generates deterministic BF16 tensors with bounded KDA/mHC
initialization for both startup loading and live updates.

Tokenizer files listed in the upstream provenance are deliberately not vendored.
Both servers use `--tokenizer ZhipuAI/GLM-5.3-Flash`; tokenizer files must be
available from the configured hub or its local cache. No checkpoint weights are
downloaded.

This fixture runs only in the shared TP2 IPC unpacked lane. HCCL and packed
transport coverage use Qwen3.5.
