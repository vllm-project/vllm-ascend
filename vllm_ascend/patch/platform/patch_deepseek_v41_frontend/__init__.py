# SPDX-License-Identifier: Apache-2.0
"""Day0 patch for vLLM DeepSeek V4.1 frontend registries."""


def register_frontend():
    from vllm.reasoning import ReasoningParserManager
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.tool_parsers import ToolParserManager

    package = "vllm_ascend.patch.platform.patch_deepseek_v41_frontend"
    TokenizerRegistry.register("deepseek_v41", f"{package}.tokenizer", "DeepseekV41Tokenizer")
    RENDERER_REGISTRY.register("deepseek_v41", f"{package}.renderer", "DeepseekV41Renderer")
    ReasoningParserManager.register_lazy_module("deepseek_v41", f"{package}.parser", "DeepseekV41ReasoningParser")
    ToolParserManager.register_lazy_module("deepseek_v41", f"{package}.parser", "DeepseekV41ToolParser")


register_frontend()
