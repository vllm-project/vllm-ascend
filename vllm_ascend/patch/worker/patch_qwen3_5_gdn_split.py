# vllm_ascend/patch/worker/patch_qwen3_5_gdn_split.py
#
# Purpose: Monkey-patch the Qwen3.5 GDN layer to split in_proj_qkvz into
#          standalone modules and update the weight loading mapping accordingly,
#          which is fully equivalent to modifying the vLLM source code directly.
#
# Switch: environment variable VLLM_ASCEND_ENABLE_GDN_QKV_SPLIT, defaults to 0 (disabled).
#         When disabled, this module applies no patch and keeps the upstream fused
#         in_proj_qkvz layout. Set it to 1 to enable the split.

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention as _GDNBaseCls,
)

import vllm_ascend.envs as envs_ascend

_original_gdn_init = _GDNBaseCls.__init__


def _patched_gdn_init(self, *args, **kwargs):
    # Let the original init do all the work (auto-forwards config and every other argument)
    _original_gdn_init(self, *args, **kwargs)

    if hasattr(self, "in_proj_qkv") and not hasattr(self, "in_proj_qkvz"):
        return

    if hasattr(self, "in_proj_qkvz"):
        delattr(self, "in_proj_qkvz")

    hidden_size = getattr(self, "hidden_size", None)
    if hidden_size is None:
        # Fallback: try to infer it from the input dimension of conv1d
        hidden_size = self.conv1d.weight.shape[1] if hasattr(self, "conv1d") else args[0] if args else None

    num_k_heads = getattr(self, "num_k_heads", getattr(self, "num_heads", 1))
    num_v_heads = getattr(self, "num_v_heads", getattr(self, "num_heads", 1))
    head_k_dim = getattr(self, "head_k_dim", 128)
    head_v_dim = getattr(self, "head_v_dim", 128)

    # Compute the global output dimensions (Linear shards them internally by TP size)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    z_global_size = value_dim

    # Infer quant_config from the one set by the original init (if it was set)
    self.quant_config = getattr(self, "quant_config", None)

    # Build the prefix (inferred from the prefix of the original init)
    prefix = getattr(self, "prefix", "")

    # q/k/v must each be sharded on their own: rearrange_mixed_qkv splits the
    # local tensor as [key_dim//tp, key_dim//tp, value_dim//tp], which only
    # holds when every segment is partitioned independently. A plain
    # ColumnParallelLinear would slice the fused [q|k|v] dim as one contiguous
    # block and hand rank i the wrong rows for tp > 1.
    self.in_proj_qkv = MergedColumnParallelLinear(
        input_size=hidden_size,
        output_sizes=[key_dim, key_dim, value_dim],
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.in_proj_qkv" if prefix else "in_proj_qkv",
    )
    self.in_proj_z = ColumnParallelLinear(
        input_size=hidden_size,
        output_size=z_global_size,
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.in_proj_z" if prefix else "in_proj_z",
    )


if envs_ascend.VLLM_ASCEND_ENABLE_GDN_QKV_SPLIT:
    _GDNBaseCls.__init__ = _patched_gdn_init
