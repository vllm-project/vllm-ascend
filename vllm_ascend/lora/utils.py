import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace

from vllm_ascend.ops.linear import (
    AscendQKVParallelLinear,
)

from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE


class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


class AscendMergedQKVParallelLinearWithLoRA(MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendMergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendQKVParallelLinearWithShardedLoRA(QKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1

class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-specific MoE LoRA that uses unified MLP execution path.

    This implementation injects LoRA context into the MoE pipeline, which is
    then consumed by the unified MLP execution path in `unquant_apply_mlp`.
    The LoRA modifications are applied at the correct positions:
        output = W2 @ act((W1 + A1@B1) @ x) + A2@B2 @ act(...)
    """
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 2

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        from vllm.lora.layers.base import BaseLayerWithLoRA
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        self.tp_size = base_layer.tp_size
        self.tp_rank = base_layer.tp_rank
        from vllm.lora.layers.utils import _get_lora_device
        self.device = _get_lora_device(base_layer)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        self.n_slices = base_layer.local_num_experts * (self._w13_slices + 1)
        # Attach a getter to the base_layer so that quant_method.apply()
        # can read the correct per-layer LoRA context without a global
        # monkey-patch (which would leak the last-initialised layer's
        # context to all MoE layers).
        self.base_layer.get_lora_context = self._build_lora_context

    def _build_lora_context(self):
        from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoELoRAContext
        return MoELoRAContext(
            w13_lora_a_stacked=self.w13_lora_a_stacked,
            w13_lora_b_stacked=self.w13_lora_b_stacked,
            w2_lora_a_stacked=self.w2_lora_a_stacked,
            w2_lora_b_stacked=self.w2_lora_b_stacked,
            punica_wrapper=self.punica_wrapper,
            num_experts=self.base_layer.local_num_experts,
        )

    def set_lora(self, index, lora_a, lora_b, embeddings_tensor=None, bias=None):
        if isinstance(lora_a, list) and len(lora_a) > 3:
            num_groups = len(lora_a) // 3
            w1_lora_a = torch.stack([lora_a[i * 3 + 0] for i in range(num_groups)])
            w2_lora_a = torch.stack([lora_a[i * 3 + 1] for i in range(num_groups)])
            w3_lora_a = torch.stack([lora_a[i * 3 + 2] for i in range(num_groups)])
            w1_lora_b = torch.stack([lora_b[i * 3 + 0] for i in range(num_groups)])
            w2_lora_b = torch.stack([lora_b[i * 3 + 1] for i in range(num_groups)])
            w3_lora_b = torch.stack([lora_b[i * 3 + 2] for i in range(num_groups)])
            lora_a = [w1_lora_a, w2_lora_a, w3_lora_a]
            lora_b = [w1_lora_b, w2_lora_b, w3_lora_b]
        super().set_lora(index, lora_a, lora_b)

    def set_mapping(self, punica_wrapper):
        # Skip FusedMoEWithLoRA.set_mapping() which accesses self._moe_kernel
        # that doesn't exist in Ascend. Call BaseLayerWithLoRA directly.
        from vllm.lora.layers.base import BaseLayerWithLoRA
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)


class AscendFusedMoE3DWithLoRA(FusedMoE3DWithLoRA):
    """Ascend-specific 3D MoE LoRA that uses unified MLP execution path.

    This is the 3D (fused gate_up_proj) counterpart of AscendFusedMoEWithLoRA.
    It bypasses the Triton-based kernel initialization and injects LoRA context
    into the Ascend MoE pipeline instead.

    In the 3D format each expert has 2 packed modules (gate_up_proj, down_proj)
    instead of 3 (gate_proj, up_proj, down_proj).
    """

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer is an AscendFusedMoE with 1 packed w13 module."""
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 1

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        from vllm.lora.layers.base import BaseLayerWithLoRA

        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        self.tp_size = base_layer.tp_size
        self.tp_rank = base_layer.tp_rank
        from vllm.lora.layers.utils import _get_lora_device

        self.device = _get_lora_device(base_layer)
        # 3D format: gate_up_proj is a single fused tensor → 1 slice
        self._w13_slices = 1
        self.n_slices = base_layer.local_num_experts * (self._w13_slices + 1)
        # Attach a getter to the base_layer for per-layer LoRA context
        # (see AscendFusedMoEWithLoRA for rationale).
        self.base_layer.get_lora_context = self._build_lora_context

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
        embeddings_tensor: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ):
        """Overwrites lora tensors at index.

        Handles both the pre-stacked format (2-element list from
        ``_stack_moe_lora_weights``) and the flat per-expert list format
        (2 * num_experts entries: w13_e0, w2_e0, w13_e1, w2_e1, …).
        """
        if isinstance(lora_a, list) and len(lora_a) > 2:
            # Flat per-expert format: stack into (w13, w2)
            num_groups = len(lora_a) // 2
            w13_lora_a = torch.stack([lora_a[i * 2] for i in range(num_groups)])
            w2_lora_a = torch.stack([lora_a[i * 2 + 1] for i in range(num_groups)])
            w13_lora_b = torch.stack([lora_b[i * 2] for i in range(num_groups)])
            w2_lora_b = torch.stack([lora_b[i * 2 + 1] for i in range(num_groups)])
            lora_a = [w13_lora_a, w2_lora_a]
            lora_b = [w13_lora_b, w2_lora_b]
        # Delegate to FusedMoE3DWithLoRA.set_lora for TP slicing & copy
        FusedMoE3DWithLoRA.set_lora(self, index, lora_a, lora_b)

    def set_mapping(self, punica_wrapper):
        # Skip FusedMoEWithLoRA.set_mapping() which accesses self._moe_kernel
        # that doesn't exist in Ascend. Call BaseLayerWithLoRA directly.
        from vllm.lora.layers.base import BaseLayerWithLoRA

        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)

    def _build_lora_context(self):
        from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoELoRAContext

        return MoELoRAContext(
            w13_lora_a_stacked=self.w13_lora_a_stacked,
            w13_lora_b_stacked=self.w13_lora_b_stacked,
            w2_lora_a_stacked=self.w2_lora_a_stacked,
            w2_lora_b_stacked=self.w2_lora_b_stacked,
            punica_wrapper=self.punica_wrapper,
            num_experts=self.base_layer.local_num_experts,
        )


def refresh_all_lora_classes():
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    # Filter out both vLLM MoE LoRA classes — the Ascend replacements
    # (AscendFusedMoEWithLoRA and AscendFusedMoE3DWithLoRA) must be tried
    # first so that the 3D len==1 and 2D len==2 checks match correctly.
    vllm_classes = tuple(
        cls
        for cls in vllm.lora.utils._all_lora_classes
        if cls not in (FusedMoEWithLoRA, FusedMoE3DWithLoRA)
    )
    vllm.lora.utils._all_lora_classes = (*vllm_classes, *ascend_classes)
