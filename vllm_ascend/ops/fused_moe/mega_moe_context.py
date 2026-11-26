# SPDX-License-Identifier: Apache-2.0
"""Share one CANN channel context between serial DeepSeek V4.1 DSpark forwards."""

from contextlib import contextmanager
from importlib import import_module
from inspect import signature
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import torch
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method, use_cann_megamoe
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.quantization.quant_type import A5_SUPPORT_MEGA_MOE_QUANT_TYPES, QuantType
from vllm_ascend.utils import is_deepseek_v41

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class ExpertLayout(NamedTuple):
    quant: QuantType
    num_experts: int
    local_experts: int
    top_k: int
    hidden: int
    intermediate: int

    def fits_target(self, target):
        return (
            self.quant == target.quant
            and self.quant in A5_SUPPORT_MEGA_MOE_QUANT_TYPES
            and self.hidden == target.hidden
            and self.num_experts <= target.num_experts
            and self.local_experts <= target.local_experts
            and self.top_k <= target.top_k
            and self.intermediate <= target.intermediate
        )


class MainOwnedMegaMoEBuffer:
    def __init__(self, factory, owner, params):
        self.factory = factory
        self.params = params
        self.owner = owner
        self.views = []
        self._allocation = owner
        self._failed_factory_group = None

    def fits(self, params):
        owner = self.owner
        missing = object()
        ep_size = getattr(owner, "ep_world_size", 0)
        if not isinstance(ep_size, int) or ep_size <= 0:
            return False
        for field in ("num_experts", "num_topk", "num_max_tokens_per_rank", "intermediate_hidden"):
            if params[field] > getattr(owner, field, 0):
                return False

        # Bind the installed API's defaults (including combine mode and
        # top-k weight format). Cross-server layouts also depend on topology.
        args = signature(self.factory).bind(owner.group, **params)
        args.apply_defaults()
        values = dict(args.arguments, ep_world_size=ep_size, moe_expert_num=params["num_experts"])
        for field in (
            "hidden",
            "dispatch_quant_mode",
            "dispatch_quant_out_dtype",
            "combine_quant_mode",
            "comm_alg",
            "combine_comm_mode",
            "topk_weights_type",
        ):
            if field in values and values[field] != getattr(owner, field, missing):
                return False
        if (
            getattr(owner, "_ctx_manager", None) is None
            or getattr(owner, "context", None) is None
            or not callable(getattr(owner, "destroy", None))
        ):
            return False
        size_query = getattr(import_module(self.factory.__module__), "_get_mega_moe_ccl_buffer_size", None)
        if size_query is None:
            return False
        query_fields = signature(size_query).parameters
        values["server_num"] = 0
        if getattr(owner, "topo_type", 0):
            ranks = getattr(owner, "rank_num_per_server", 0)
            if "server_num" not in query_fields or not isinstance(ranks, int) or ranks <= 0 or ep_size % ranks:
                return False
            values["server_num"] = ep_size // ranks
        if not query_fields.keys() <= values.keys():
            return False
        required = size_query(**{field: values[field] for field in query_fields})
        return 0 < required <= owner.ccl_buffer_size

    def view(self, params):
        # On A5 r148, serial target and draft forwards can use different expert
        # counts on the same communicator and tag when their communication
        # settings match and the draft fits the target's allocation.
        view = SimpleNamespace(**vars(self.owner))
        vars(view).update(params)
        self.views.append(view)
        return view

    def restore(self, scope):
        group = scope.device_group
        owner = self.owner
        allocated = self._allocation is not None and self._allocation.group is group
        states = _agree(scope, (owner.group is group, allocated, self._failed_factory_group is group))
        if any(factory_failed for _, _, factory_failed in states):
            raise RuntimeError("MegaMoE allocation failed on this communicator; recreate the MC2 group before retrying")
        if all(published for published, _, _ in states):
            return
        if any(has_buffer != allocated for _, has_buffer, _ in states):
            # CANN may already have cached the successful ranks' allocations.
            # Calling its collective factory on a subset cannot repair this group.
            raise RuntimeError("Partial MegaMoE channel allocation; recreate the MC2 group before retrying")
        with torch.inference_mode():
            if not allocated:
                # A recreated group proves the old channel is retired. Agree on
                # releasing its HBM before any rank enters CANN's factory.
                with _restore_step(scope, "release"):
                    if self._allocation is not None:
                        self._allocation.destroy()
                        self._allocation = None
            with _restore_step(scope, "validate"):
                if not allocated:
                    # CANN can register the tag before the factory returns a buffer.
                    self._failed_factory_group = group
                    self._allocation = self.factory(group, **self.params)
                    self._failed_factory_group = None
                replacement = self._allocation
                for name in ("ep_world_size", "topo_type", "rank_num_per_server"):
                    if getattr(replacement, name, None) != getattr(owner, name, None):
                        raise RuntimeError(f"MegaMoE {name} changed when restoring the channel window")
                if replacement.ccl_buffer_size < owner.ccl_buffer_size:
                    raise RuntimeError("Restored MegaMoE channel window is smaller than its captured allocation")
            with _restore_step(scope, "copy context"):
                owner.context.copy_(replacement.context)
                device = torch.npu.current_device()
            with _restore_step(scope, "synchronize"):
                torch.distributed.barrier(group=group, device_ids=[device])
                # Publish the owner's group last. A retry can finish publication
                # on every rank over the existing collective allocation.
                for buffer in (*self.views, owner):
                    for name in ("rank_id", "group_name", "_ctx_manager", "ccl_buffer_size"):
                        setattr(buffer, name, getattr(replacement, name))
                    buffer.group = group


@torch.inference_mode(False)
def _agree(scope, local):
    decisions = [None] * scope.world_size
    torch.distributed.all_gather_object(decisions, local, group=scope.cpu_group)
    return decisions


@contextmanager
def _restore_step(scope, name):
    error = None
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    for rank, failure in enumerate(_agree(scope, error)):
        if failure is not None:
            raise RuntimeError(f"MegaMoE channel restore failed during {name} on rank {rank}: {failure}")


def try_share_dspark_mega_moe_buffer(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    vllm_config: "VllmConfig",
    draft_model_config: "ModelConfig | None",
    *,
    context_kv_only: bool = False,
) -> bool:
    """Check the supported DSpark configuration before probing its MoE buffers."""
    draft_hf = getattr(draft_model_config, "hf_config", None)
    # TODO: Validate more A5 DSpark models before relaxing the V4.1 gate.
    # The sharing probe adds a startup collective even for dense drafts
    # and can allocate the target buffer before the final capacity check.
    if not (
        getattr(draft_hf, "architectures", None) == ["DeepseekV41DSparkModel"]
        and not context_kv_only
        and is_deepseek_v41(vllm_config.model_config.hf_config)
    ):
        return False
    if not (
        get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
        and use_cann_megamoe(vllm_config)
        and select_moe_comm_method(1, vllm_config) is MoECommType.FUSED_MC2
    ):
        return False

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    return try_share_mega_moe_buffer(
        fused_layouts(target_model),
        fused_layouts(draft_model),
        get_mc2_group(),
        not vllm_config.parallel_config.use_ubatching,
    )


def fused_layouts(model):
    """Read the routed layer layouts and their registered FUSED_MC2 methods."""
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    layouts = {}
    for layer in model.modules():
        if isinstance(layer, AscendRoutedExperts):
            cfg = layer.moe_config
            shape = ExpertLayout(
                layer.quant_type,
                cfg.num_experts,
                cfg.num_local_experts,
                cfg.experts_per_token,
                cfg.hidden_dim,
                cfg.intermediate_size_per_partition,
            )
            layouts[shape] = get_moe_comm_method(MoECommType.FUSED_MC2, cfg)
    return layouts


@torch.inference_mode()
def try_share_mega_moe_buffer(main, draft, scope, allow_sharing):
    """Attach a shared window only to distinct, compatible expert layouts."""
    from vllm_ascend.ops.fused_moe.moe_utils import _get_cann_mega_moe_quant_settings

    # All ranks must take the same branches before calling CANN's collective factory.
    layout = tuple(main), tuple(draft), allow_sharing
    if any(peer != layout for peer in _agree(scope, layout)):
        return False
    if len(main) != 1 or len(draft) != 1 or not allow_sharing:
        return False

    ((main_shape, target),) = main.items()
    ((draft_shape, proposer),) = draft.items()
    eligible = (
        target is not None
        and proposer is not None
        and main_shape.num_experts != draft_shape.num_experts
        and target is not proposer
        and draft_shape.fits_target(main_shape)
        and target.moe_config.mc2_group is scope
        and proposer.moe_config.mc2_group is scope
        and hasattr(target, "get_symm_buffer_for_mega_moe")
        and hasattr(proposer, "get_symm_buffer_for_mega_moe")
        and proposer.mega_moe_symm_buffer is None
    )
    if not all(_agree(scope, eligible)):
        return False
    mode, dtype, _ = _get_cann_mega_moe_quant_settings(main_shape.quant)
    main_params = target.mega_moe_buffer_params(mode, dtype)
    draft_params = proposer.mega_moe_buffer_params(mode, dtype)
    owner = target.mega_moe_symm_buffer
    params = main_params, draft_params, owner is not None
    if any(peer != params for peer in _agree(scope, params)):
        return False
    if draft_params["num_max_tokens_per_rank"] > main_params["num_max_tokens_per_rank"]:
        return False

    if owner is None:
        owner = target.get_symm_buffer_for_mega_moe(scope.device_group, **main_params)
        target.mega_moe_symm_buffer = owner
    shared_buffer = getattr(target, "_shared_mega_moe_buffer", None)
    if shared_buffer is None:
        shared_buffer = MainOwnedMegaMoEBuffer(target.get_symm_buffer_for_mega_moe, owner, main_params)
    owns_group = getattr(owner, "group", None) is scope.device_group
    try:
        fits = owns_group and shared_buffer.fits(draft_params)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("Could not verify the draft MegaMoE window size: %s", exc)
        fits = False
    checks = _agree(scope, (owns_group, fits))
    if not all(peer_fits for _, peer_fits in checks):
        if any(peer_owns_group for peer_owns_group, _ in checks):
            raise RuntimeError(
                f"The target already owns the MegaMoE buffer on the draft's MC2 group, but sharing could not "
                f"be verified (target experts={main_shape.num_experts}, draft experts={draft_shape.num_experts}). "
                "Disable CANN MegaMoE or use a draft with compatible communication settings and buffer requirements."
            )
        return False
    proposer.mega_moe_symm_buffer = shared_buffer.view(draft_params)
    target._shared_mega_moe_buffer = shared_buffer
    proposer._shared_mega_moe_buffer = shared_buffer
    return True
