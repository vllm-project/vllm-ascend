"""DSA 稀疏卸载支持模型的能力入口。

本模块只描述“当前哪种 vLLM 已解析模型架构可以启用这套 DSA 算法”，
不安装 monkey patch，也不依赖 scheduler、worker 或设备算子。这样 scheduler
补丁和 Ascend worker 可以消费同一份结论，避免执行层反向导入 patch 模块。

这里使用 ``ModelConfig.architecture``，因为它是 vLLM registry 最终解析并
实际加载的架构；原始 Hugging Face ``architectures`` 仅是候选列表，不能作为
运行时能力判断的第二真源。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_ascend.dsa_sparse.dsa_config import is_dsa_sparse_config_enabled

if TYPE_CHECKING:
    from vllm.config import VllmConfig


DSA_SPARSE_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "GlmMoeDsaForCausalLM",
    }
)


def is_dsa_sparse_model_supported(vllm_config: VllmConfig) -> bool:
    """Return whether vLLM resolved the model to a supported DSA architecture."""
    return vllm_config.model_config.architecture in DSA_SPARSE_SUPPORTED_ARCHITECTURES


def is_dsa_sparse_runtime_enabled(vllm_config: VllmConfig) -> bool:
    """Return whether DSA is configured and supported for this model."""
    return is_dsa_sparse_config_enabled(vllm_config) and is_dsa_sparse_model_supported(vllm_config)
