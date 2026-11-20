import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.logger import logger
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear


def _describe_lora_tensor_shapes(tensors: torch.Tensor | list[torch.Tensor | None]) -> str:
    if isinstance(tensors, torch.Tensor):
        return str(tuple(tensors.shape))
    return str([tuple(t.shape) if isinstance(t, torch.Tensor) else None for t in tensors])


def _format_lora_shape_error(
    layer: nn.Module,
    index: int,
    lora_a: torch.Tensor | list[torch.Tensor | None],
    lora_b: torch.Tensor | list[torch.Tensor | None],
    err: Exception,
) -> str:
    """Build a diagnostic error message with module prefix, slot, and tensor shapes."""
    base_layer = getattr(layer, "base_layer", None)
    module_name = getattr(base_layer, "prefix", "") or type(layer).__name__
    expected_a = [tuple(t.shape[-2:]) for t in getattr(layer, "lora_a_stacked", ())]
    expected_b = [tuple(t.shape[-2:]) for t in getattr(layer, "lora_b_stacked", ())]
    return (
        f"Failed to load LoRA weights for module '{module_name}' at slot {index}: "
        f"got lora_a={_describe_lora_tensor_shapes(lora_a)}, "
        f"lora_b={_describe_lora_tensor_shapes(lora_b)}; "
        f"expected local slice shapes lora_a={expected_a}, lora_b={expected_b} "
        f"(input_size={getattr(layer, 'input_size', '?')}, "
        f"output_slices={getattr(layer, 'output_slices', '?')}). Original error: {err}"
    )


def _apply_packed_lora(layer, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    original_shape = output.shape if output.ndim == 3 else None
    if original_shape is not None and x.ndim == 3:
        x, output = x.flatten(0, 1), output.flatten(0, 1)
    lora_output = layer.punica_wrapper.add_lora_linear(
        output,
        x,
        layer.lora_a_stacked,
        layer.lora_b_stacked,
        1.0,
        layer.output_slices,
        packed_lora_a=getattr(layer, "lora_a_packed", None),
        packed_lora_b=layer.lora_b_packed,
    )
    if not current_platform.can_update_inplace():
        output = lora_output
    return output.reshape(original_shape) if original_shape is not None else output


class _PackedLoRAAWeightsMixin(MergedColumnParallelLinearWithLoRA):
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        self._use_packed_lora = max_loras == 1
        self._lora_slots: set[int] = set()
        if not self._use_packed_lora:
            self.lora_a_packed = None
            self.lora_b_packed = None
            return
        rank = self.lora_a_stacked[0].size(2)
        self.lora_a_packed = torch.zeros(
            max_loras,
            1,
            self.n_slices * rank,
            self.input_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.lora_b_packed = torch.zeros(
            max_loras,
            1,
            self.n_slices * rank,
            sum(self.output_slices),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        if getattr(self, "_use_packed_lora", True):
            if getattr(self, "lora_a_packed", None) is not None:
                self.lora_a_packed[index].zero_()
            if getattr(self, "lora_b_packed", None) is not None:
                self.lora_b_packed[index].zero_()
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is not None:
            lora_slots.discard(index)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        try:
            super().set_lora(index, lora_a, lora_b)
        except RuntimeError as err:
            raise ValueError(_format_lora_shape_error(self, index, lora_a, lora_b, err)) from err
        if (
            getattr(self, "_use_packed_lora", True)
            and getattr(self, "lora_a_packed", None) is not None
            and getattr(self, "lora_b_packed", None) is not None
        ):
            rank = self.lora_a_stacked[0].size(2)
            for slice_index, slice_weight in enumerate(self.lora_a_stacked):
                packed_slice = self.lora_a_packed[index, 0].narrow(0, slice_index * rank, rank)
                packed_slice.copy_(slice_weight[index, 0], non_blocking=True)
            packed_b = self.lora_b_packed[index, 0].zero_()
            offset = 0
            for slice_index, slice_weight in enumerate(self.lora_b_stacked):
                out_size = self.output_slices[slice_index]
                block = packed_b.narrow(0, slice_index * rank, rank).narrow(1, offset, out_size)
                block.copy_(
                    slice_weight[index, 0, :out_size].transpose(0, 1),
                    non_blocking=True,
                )
                offset += out_size
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is None:
            self._lora_slots = {index}
        else:
            lora_slots.add(index)

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_use_packed_lora", True) and getattr(self, "lora_b_packed", None) is not None:
            return _apply_packed_lora(self, x, output)
        return super()._apply_lora_to_output(x, output)


class AscendMergedColumnParallelLinearWithLoRA(_PackedLoRAAWeightsMixin):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras >= 1
            and type(source_layer) is maybe_get_oot_by_class(MergedColumnParallelLinear)
            and len(packed_modules_list) == 2
        )


class AscendMergedQKVParallelLinearWithLoRA(_PackedLoRAAWeightsMixin, MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras >= 1
            and type(source_layer) is AscendQKVParallelLinear
            and len(packed_modules_list) == 3
        )


class _TransposedLoRABMixin(BaseLinearLayerWithLoRA):
    """Keep a contiguous [rank, output] B copy for the Triton expand kernel."""

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        self._use_packed_lora = max_loras == 1
        self._lora_slots: set[int] = set()
        if not self._use_packed_lora:
            self.lora_b_packed = None
            return
        rank = self.lora_a_stacked[0].size(2)
        self.lora_b_packed = torch.zeros(
            max_loras,
            1,
            rank,
            self.output_slices[0],
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        if getattr(self, "_use_packed_lora", True) and getattr(self, "lora_b_packed", None) is not None:
            self.lora_b_packed[index].zero_()
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is not None:
            lora_slots.discard(index)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        try:
            super().set_lora(index, lora_a, lora_b)
        except RuntimeError as err:
            raise ValueError(_format_lora_shape_error(self, index, lora_a, lora_b, err)) from err
        if getattr(self, "_use_packed_lora", True) and getattr(self, "lora_b_packed", None) is not None:
            self.lora_b_packed[index, 0].copy_(
                self.lora_b_stacked[0][index, 0].transpose(0, 1),
                non_blocking=True,
            )
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is None:
            self._lora_slots = {index}
        else:
            lora_slots.add(index)

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_use_packed_lora", True) and getattr(self, "lora_b_packed", None) is not None:
            return _apply_packed_lora(self, x, output)
        return super()._apply_lora_to_output(x, output)


class _SingleLinearLoRAWrapper(_TransposedLoRABMixin):
    base_layer_cls: type[nn.Module]

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return lora_config.max_loras >= 1 and type(source_layer) is maybe_get_oot_by_class(cls.base_layer_cls)


class AscendRowParallelLinearWithLoRA(_SingleLinearLoRAWrapper, RowParallelLinearWithLoRA):
    base_layer_cls = RowParallelLinear


class AscendColumnParallelLinearWithLoRA(_SingleLinearLoRAWrapper, ColumnParallelLinearWithLoRA):
    base_layer_cls = ColumnParallelLinear


class AscendReplicatedLinearWithLoRA(_SingleLinearLoRAWrapper, ReplicatedLinearWithLoRA):
    """Replicated (non-TP) linear with the Ascend packed-B expand kernel.

    ``ReplicatedLinear`` is never split across ranks, so the whole output is
    produced locally. Without this wrapper the generic Ascend runtime leaves
    those layers on the fallback path instead of the single-slot packed kernel
    used by ``wq_a``/``wkv`` projections in DeepSeek compressed attention.
    """

    base_layer_cls = ReplicatedLinear


def warn_unmatched_lora_modules(lora_manager, lora_request) -> list[str]:
    """Warn once when loaded adapter keys do not match any wrapped model module."""
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    lora_id = getattr(lora_request, "lora_int_id", None)
    if adapter_manager is None or lora_id is None:
        return []

    get_adapter = getattr(adapter_manager, "get_adapter", None)
    if callable(get_adapter):
        lora_model = get_adapter(lora_id)
    else:
        lora_model = getattr(adapter_manager, "_registered_adapters", {}).get(lora_id)
    if lora_model is None:
        return []

    loras = getattr(lora_model, "loras", None)
    modules = getattr(adapter_manager, "modules", None)
    if not isinstance(loras, dict) or not isinstance(modules, dict):
        return []

    get_weights = getattr(adapter_manager, "_get_lora_layer_weights", None)
    get_text_name = getattr(adapter_manager, "_get_text_lora_name", None)
    matched_count = 0
    matched_lora_keys: set[str] = set()
    for module_name in modules:
        matched_weight = get_weights(lora_model, module_name) if callable(get_weights) else loras.get(module_name)
        if matched_weight is None:
            continue
        matched_count += 1
        matched_lora_keys.add(module_name)
        matched_lora_keys.add(module_name.removeprefix("model."))
        if callable(get_text_name):
            text_name = get_text_name(lora_model, module_name)
            if text_name is not None:
                matched_lora_keys.add(text_name)

    packed_modules = getattr(adapter_manager, "packed_modules", None) or {}
    known_packed_subkeys: set[str] = set()
    for subkeys in packed_modules.values():
        for subkey in subkeys:
            known_packed_subkeys.add(subkey)
            known_packed_subkeys.add(subkey.removeprefix("model."))

    unmatched = sorted(key for key in loras if key not in matched_lora_keys and key not in known_packed_subkeys)
    lora_name = getattr(lora_request, "lora_name", str(lora_id))
    if unmatched:
        has_expert_key = any("experts" in key or "base_layer" in key for key in unmatched)
        hint = (
            " For 2D MoE models, use per-expert keys "
            "(e.g. '...mlp.experts.<id>.{gate_proj,up_proj,down_proj}' or "
            "HF '...ffn.experts.<id>.{w1,w2,w3}'), or launch with "
            "--enable-mixed-moe-lora-format and set is_3d_lora_weight=true "
            "for fused 3D MoE adapters."
            if has_expert_key
            else ""
        )
        logger.warning(
            "LoRA adapter '%s' (id=%s) matched %d wrapped module(s) on this worker, "
            "but %d key(s) did not match any target module and will be ignored "
            "(e.g. %s).%s",
            lora_name,
            lora_id,
            matched_count,
            len(unmatched),
            unmatched[:5],
            hint,
        )
    elif matched_count == 0 and not any(key in known_packed_subkeys for key in loras):
        logger.warning(
            "LoRA adapter '%s' (id=%s) matched 0 target modules on this worker; "
            "requests using it will produce base model outputs with zero LoRA delta. "
            "Verify adapter key prefixes and target_modules.",
            lora_name,
            lora_id,
        )
    return unmatched


def refresh_all_lora_classes():
    ascend_classes = (
        AscendReplicatedLinearWithLoRA,
        AscendRowParallelLinearWithLoRA,
        AscendColumnParallelLinearWithLoRA,
        AscendMergedColumnParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    existing_classes = tuple(cls for cls in vllm.lora.utils._all_lora_classes if cls not in ascend_classes)
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *existing_classes,
    )
