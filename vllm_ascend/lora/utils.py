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
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import MoERunner
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
    assert_moe_lora_dp1_supported,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear
from vllm_ascend.utils import oproj_tp_enable


def _safe_oproj_tp_enable() -> bool:
    """Return whether fine-grained o_proj TP is enabled, defaulting False if uninitialized."""
    try:
        return bool(oproj_tp_enable())
    except RuntimeError:
        return False


def _get_unsupported_lora_module_reason(
    source_layer: nn.Module,
    lora_config: LoRAConfig | None,
) -> str | None:
    """Return an explicit error reason when a linear projection must stay unwrapped."""
    explicit_reason = getattr(source_layer, "_ascend_lora_unsupported_reason", None)
    if explicit_reason:
        return str(explicit_reason)

    n_local_groups = getattr(source_layer, "_ascend_dsa_n_local_groups", None)
    if n_local_groups is not None:
        if lora_config is not None and getattr(lora_config, "fully_sharded_loras", False):
            return (
                "LoRA on wo_a does not support --fully-sharded-loras. "
                "Disable --fully-sharded-loras or drop wo_a from the adapter."
            )
        otp_enabled = _safe_oproj_tp_enable()
        if otp_enabled or n_local_groups != 1:
            return (
                "LoRA on wo_a requires a single local o_proj group per rank "
                f"(got n_local_groups={n_local_groups}, fine-grained o_proj TP={otp_enabled}). "
                "Use TP equal to o_groups, or drop wo_a from the adapter."
            )
    return None


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
    *,
    expected_full: tuple[int, int] | None = None,
) -> str:
    """Build a diagnostic error message with module prefix, slot, and tensor shapes."""
    base_layer = getattr(layer, "base_layer", None)
    module_name = getattr(base_layer, "prefix", "") or type(layer).__name__
    expected_a = [tuple(t.shape[-2:]) for t in getattr(layer, "lora_a_stacked", ())]
    expected_b = [tuple(t.shape[-2:]) for t in getattr(layer, "lora_b_stacked", ())]
    tp_size = getattr(layer, "tp_size", 1)
    full_hint = f", expected_full=(in={expected_full[0]}, out={expected_full[1]})" if expected_full is not None else ""
    return (
        f"Failed to load LoRA weights for module '{module_name}' at slot {index}: "
        f"got lora_a={_describe_lora_tensor_shapes(lora_a)}, "
        f"lora_b={_describe_lora_tensor_shapes(lora_b)}; "
        f"expected local slice shapes lora_a={expected_a}, lora_b={expected_b} "
        f"(input_size={getattr(layer, 'input_size', '?')}, "
        f"output_slices={getattr(layer, 'output_slices', '?')}, tp_size={tp_size}{full_hint}). "
        "Verify that the adapter's hidden/projection dimensions match the base model checkpoint "
        "(for example, DeepSeek-V4 vs DeepSeek-V4.1 differ in hidden_size and q_lora_rank). "
        f"Original error: {err}"
    )


def _validate_single_linear_lora_shapes(
    layer: nn.Module,
    index: int,
    lora_a: torch.Tensor | list[torch.Tensor],
    lora_b: torch.Tensor | list[torch.Tensor],
) -> None:
    if (
        not isinstance(lora_a, torch.Tensor)
        or not isinstance(lora_b, torch.Tensor)
        or lora_a.ndim != 2
        or lora_b.ndim != 2
    ):
        return
    a_stacked = getattr(layer, "lora_a_stacked", None)
    b_stacked = getattr(layer, "lora_b_stacked", None)
    if not a_stacked or not b_stacked:
        return
    max_rank_a = a_stacked[0].size(2)
    max_rank_b = b_stacked[0].size(3)
    local_in = a_stacked[0].size(3)
    local_out = b_stacked[0].size(2)
    tp_size = getattr(layer, "tp_size", 1)

    if isinstance(layer, RowParallelLinearWithLoRA):
        expected_in, expected_out = local_in * tp_size, local_out
    elif isinstance(layer, ColumnParallelLinearWithLoRA) and not getattr(layer, "is_merged_col_linear", False):
        expected_in, expected_out = local_in, local_out * tp_size
    elif isinstance(layer, ReplicatedLinearWithLoRA):
        expected_in, expected_out = local_in, local_out
    else:
        return

    rank_ok = lora_a.shape[0] <= max_rank_a and lora_b.shape[1] <= max_rank_b and lora_a.shape[0] == lora_b.shape[1]
    dims = (lora_a.shape[1], lora_b.shape[0])
    # Warmup (LoRAModelManager.create_dummy_lora) builds dummy tensors from local
    # sharded stacked buffer sizes even when tp_size > 1.
    dims_ok = dims == (expected_in, expected_out) or (tp_size > 1 and dims == (local_in, local_out))
    if not (rank_ok and dims_ok):
        raise ValueError(
            _format_lora_shape_error(
                layer,
                index,
                lora_a,
                lora_b,
                ValueError("adapter tensor shape does not match target projection"),
                expected_full=(expected_in, expected_out),
            )
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
    _ascend_unsupported_lora_guarded = True

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
            lora_a_packed = getattr(self, "lora_a_packed", None)
            if lora_a_packed is not None:
                lora_a_packed[index].zero_()
            lora_b_packed = getattr(self, "lora_b_packed", None)
            if lora_b_packed is not None:
                lora_b_packed[index].zero_()
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
        lora_a_packed = getattr(self, "lora_a_packed", None)
        lora_b_packed = getattr(self, "lora_b_packed", None)
        if getattr(self, "_use_packed_lora", True) and lora_a_packed is not None and lora_b_packed is not None:
            rank = self.lora_a_stacked[0].size(2)
            for slice_index, slice_weight in enumerate(self.lora_a_stacked):
                packed_slice = lora_a_packed[index, 0].narrow(0, slice_index * rank, rank)
                packed_slice.copy_(slice_weight[index, 0], non_blocking=True)
            packed_b = lora_b_packed[index, 0].zero_()
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
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        if _get_unsupported_lora_module_reason(source_layer, lora_config) is not None:
            return False
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
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        if _get_unsupported_lora_module_reason(source_layer, lora_config) is not None:
            return False
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
        lora_b_packed = getattr(self, "lora_b_packed", None)
        if getattr(self, "_use_packed_lora", True) and lora_b_packed is not None:
            lora_b_packed[index].zero_()
        lora_slots = getattr(self, "_lora_slots", None)
        if lora_slots is not None:
            lora_slots.discard(index)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        _validate_single_linear_lora_shapes(self, index, lora_a, lora_b)
        try:
            super().set_lora(index, lora_a, lora_b)
        except RuntimeError as err:
            raise ValueError(_format_lora_shape_error(self, index, lora_a, lora_b, err)) from err
        lora_b_packed = getattr(self, "lora_b_packed", None)
        if getattr(self, "_use_packed_lora", True) and lora_b_packed is not None:
            lora_b_packed[index, 0].copy_(
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
    _ascend_unsupported_lora_guarded = True
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
        if _get_unsupported_lora_module_reason(source_layer, lora_config) is not None:
            return False
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


def warn_unmatched_lora_modules(lora_manager, lora_request, *, lora_model=None) -> list[str]:
    """Warn once when loaded adapter keys do not match any wrapped model module."""
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    lora_id = getattr(lora_request, "lora_int_id", None)
    if lora_model is None:
        if adapter_manager is None or lora_id is None:
            return []
        get_adapter = getattr(adapter_manager, "get_adapter", None)
        if callable(get_adapter):
            lora_model = get_adapter(lora_id)
        else:
            lora_model = getattr(adapter_manager, "_registered_adapters", {}).get(lora_id)
    elif adapter_manager is None:
        return []
    if lora_id is None:
        lora_id = getattr(lora_model, "id", None)
    if lora_model is None:
        return []

    loras = getattr(lora_model, "loras", None)
    modules = getattr(adapter_manager, "modules", None)
    if not isinstance(loras, dict) or not isinstance(modules, dict):
        return []

    # Lightweight test stubs sometimes pass SimpleNamespace(loras=...) without
    # LoRAModel's check_lora_name/get_lora helpers, which upstream's manager methods require.
    has_lora_model_api = callable(getattr(lora_model, "check_lora_name", None)) and callable(
        getattr(lora_model, "get_lora", None)
    )
    get_weights = getattr(adapter_manager, "_get_lora_layer_weights", None) if has_lora_model_api else None
    get_text_name = getattr(adapter_manager, "_get_text_lora_name", None) if has_lora_model_api else None
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


def _expand_skipped_module_names(model: nn.Module, adapter_manager, name: str) -> set[str]:
    """Collect module key variants (direct, stripped model., and text-backbone mm aliases)."""
    candidates = {name, name.removeprefix("model.")}
    get_text_names = getattr(adapter_manager, "_get_text_lora_names", None)
    if callable(get_text_names):
        for text_name in get_text_names(name):
            candidates.add(text_name)
            candidates.add(text_name.removeprefix("model."))
    get_mm_mapping = getattr(model, "get_mm_mapping", None)
    if callable(get_mm_mapping):
        mm_mapping = get_mm_mapping()
        for mm_prefix in getattr(mm_mapping, "language_model", ()) or ():
            normalized_prefix = f"{mm_prefix.rstrip('.')}." if mm_prefix else ""
            if normalized_prefix and name.startswith(normalized_prefix):
                stripped = name[len(normalized_prefix) :]
                candidates.add(stripped)
                candidates.add(stripped.removeprefix("model."))
    return candidates


def _collect_skipped_dp_moe_prefixes(adapter_manager) -> tuple[int, tuple[str, ...]]:
    model = getattr(adapter_manager, "model", None)
    if model is None or not callable(getattr(model, "named_modules", None)):
        return 1, ()
    wrapped_modules = getattr(adapter_manager, "modules", None) or {}
    moe_cls = maybe_get_oot_by_class(MoERunner)
    moe_types = (MoERunner,) if moe_cls is MoERunner else (MoERunner, moe_cls)

    max_dp_size = 1
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        if not name or not isinstance(module, moe_types) or name in wrapped_modules:
            continue
        moe_config = getattr(module, "moe_config", None)
        dp_size = getattr(moe_config, "dp_size", 1) if moe_config is not None else 1
        if dp_size <= 1:
            continue
        max_dp_size = max(max_dp_size, dp_size)
        prefixes.update(_expand_skipped_module_names(model, adapter_manager, name))
    return max_dp_size, tuple(sorted(prefixes))


def validate_dp_moe_lora_adapter(
    adapter_manager,
    lora_model,
    *,
    skipped_dp_info: tuple[int, tuple[str, ...]] | None = None,
) -> None:
    """Reject adapters targeting MoE expert layers skipped under DP > 1."""
    dp_size, prefixes = (
        skipped_dp_info if skipped_dp_info is not None else _collect_skipped_dp_moe_prefixes(adapter_manager)
    )
    if dp_size <= 1 or not prefixes:
        return
    loras = getattr(lora_model, "loras", None)
    if not isinstance(loras, dict) or not loras:
        return
    dot_prefixes = tuple(f"{prefix}." for prefix in prefixes)
    prefix_set = set(prefixes)
    matched = sorted(key for key in loras if key in prefix_set or key.startswith(dot_prefixes))
    if matched:
        assert_moe_lora_dp1_supported(dp_size, matched_keys=matched)


def _collect_skipped_unsupported_lora_modules(adapter_manager) -> dict[str, str]:
    model = getattr(adapter_manager, "model", None)
    if model is None or not callable(getattr(model, "named_modules", None)):
        return {}
    wrapped_modules = getattr(adapter_manager, "modules", None) or {}
    lora_config = getattr(adapter_manager, "lora_config", None)

    prefix_reasons: dict[str, str] = {}
    logged_reasons: set[str] = set()
    for name, module in model.named_modules():
        if not name or name in wrapped_modules:
            continue
        reason = _get_unsupported_lora_module_reason(module, lora_config)
        if reason is None:
            continue
        if reason not in logged_reasons:
            logged_reasons.add(reason)
            logger.warning_once(
                "Ascend LoRA skipping wrapping unsupported projection '%s' and equivalent layers: %s",
                name,
                reason,
            )
        for candidate in _expand_skipped_module_names(model, adapter_manager, name):
            prefix_reasons.setdefault(candidate, reason)
    return prefix_reasons


def validate_unsupported_lora_adapter(
    adapter_manager,
    lora_model,
    *,
    skipped_unsupported_modules: dict[str, str] | None = None,
) -> None:
    """Reject adapters targeting unwrapped unsupported projections before cache write."""
    prefix_reasons = (
        skipped_unsupported_modules
        if skipped_unsupported_modules is not None
        else _collect_skipped_unsupported_lora_modules(adapter_manager)
    )
    if not prefix_reasons:
        return
    loras = getattr(lora_model, "loras", None)
    if not isinstance(loras, dict) or not loras:
        return
    ordered_prefixes = sorted(prefix_reasons)
    matched_by_reason: dict[str, list[str]] = {}
    for key in sorted(loras):
        reason = prefix_reasons.get(key)
        if reason is None:
            for prefix in ordered_prefixes:
                if key.startswith(f"{prefix}."):
                    reason = prefix_reasons[prefix]
                    break
        if reason is not None:
            matched_by_reason.setdefault(reason, []).append(key)
    if matched_by_reason:
        first_reason, matched_keys = min(matched_by_reason.items(), key=lambda item: (item[1][0], item[0]))
        raise NotImplementedError(f"{first_reason} Matched adapter key(s): {matched_keys[:3]}.")


class _DPMoELoadGuard:
    """Wrap WorkerLoRAManager._load_adapter to validate and warn on every real load."""

    def __init__(
        self,
        orig_load_adapter,
        lora_manager,
        adapter_manager,
        skipped_dp_info: tuple[int, tuple[str, ...]],
        skipped_unsupported_modules: dict[str, str] | None = None,
    ) -> None:
        self._orig_load_adapter = orig_load_adapter
        self._lora_manager = lora_manager
        self._adapter_manager = adapter_manager
        self._skipped_dp_info = skipped_dp_info
        self._skipped_unsupported_modules = skipped_unsupported_modules

    def __call__(self, lora_request):
        lora_model = self._orig_load_adapter(lora_request)
        validate_dp_moe_lora_adapter(
            self._adapter_manager,
            lora_model,
            skipped_dp_info=self._skipped_dp_info,
        )
        validate_unsupported_lora_adapter(
            self._adapter_manager,
            lora_model,
            skipped_unsupported_modules=self._skipped_unsupported_modules,
        )
        warn_unmatched_lora_modules(self._lora_manager, lora_request, lora_model=lora_model)
        return lora_model


def is_lora_load_guard_installed(lora_manager) -> bool:
    """Return whether ``lora_manager._load_adapter`` is wrapped by ``_DPMoELoadGuard``."""
    return isinstance(getattr(lora_manager, "_load_adapter", None), _DPMoELoadGuard)


def install_dp_moe_lora_load_guard(lora_manager) -> None:
    """Guard _load_adapter for DP>1 MoE, unsupported projections, and unmatched-key warnings."""
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return
    orig_load_adapter = getattr(lora_manager, "_load_adapter", None)
    if not callable(orig_load_adapter) or isinstance(orig_load_adapter, _DPMoELoadGuard):
        return
    skipped_dp_info = _collect_skipped_dp_moe_prefixes(adapter_manager)
    skipped_unsupported_modules = _collect_skipped_unsupported_lora_modules(adapter_manager)
    lora_manager._load_adapter = _DPMoELoadGuard(
        orig_load_adapter,
        lora_manager,
        adapter_manager,
        skipped_dp_info,
        skipped_unsupported_modules=skipped_unsupported_modules,
    )


def _guard_upstream_lora_class(cls: object) -> object:
    """Wrap upstream linear LoRA classes so unsupported projections stay unwrapped."""
    if not isinstance(cls, type) or not issubclass(cls, BaseLinearLayerWithLoRA):
        return cls
    if getattr(cls, "_ascend_unsupported_lora_guarded", False):
        return cls

    base_cls: type[BaseLinearLayerWithLoRA] = cls

    def can_replace_layer(
        sub_cls: type[BaseLinearLayerWithLoRA],
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
        **kwargs,
    ) -> bool:
        if _get_unsupported_lora_module_reason(source_layer, lora_config) is not None:
            return False
        for mro_cls in base_cls.__mro__:
            descriptor = mro_cls.__dict__.get("can_replace_layer")
            if descriptor is not None:
                return descriptor.__get__(None, sub_cls)(
                    source_layer=source_layer,
                    lora_config=lora_config,
                    packed_modules_list=packed_modules_list,
                    model_config=model_config,
                    **kwargs,
                )
        raise AttributeError(f"{base_cls.__name__} has no can_replace_layer")

    return type(
        f"_AscendGuarded{base_cls.__name__}",
        (base_cls,),
        {
            "__module__": __name__,
            "_ascend_unsupported_lora_guarded": True,
            "can_replace_layer": classmethod(can_replace_layer),
        },
    )


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
    # Exclude GPU FusedMoE wrappers so skipping MoE wrapping on DP > 1 never falls back to GPU classes,
    # and wrap upstream linear fallback wrappers with guards for unsupported projections.
    excluded_classes = (*ascend_classes, FusedMoEWithLoRA, FusedMoE3DWithLoRA)
    existing_classes: list[object] = []
    seen: set[object] = set(excluded_classes)
    for cls in vllm.lora.utils._all_lora_classes:
        if cls in seen:
            continue
        mapped = _guard_upstream_lora_class(cls)
        if mapped in seen:
            continue
        seen.add(mapped)
        existing_classes.append(mapped)
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *existing_classes,
    )
