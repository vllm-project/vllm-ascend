# SPDX-License-Identifier: Apache-2.0
"""A single CANN channel window for serialized target and draft MoE layers."""

from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from inspect import signature
from types import SimpleNamespace

import torch
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.quant_type import A5_SUPPORT_MEGA_MOE_QUANT_TYPES

# Only populated while the worker constructs a speculative A5 model pair.
_loading_layers: ContextVar[dict | None] = ContextVar("channel_mega_moe_layers", default=None)


@contextmanager
def collect_channel_layers():
    layers = {}
    token = _loading_layers.set(layers)
    try:
        yield layers
    finally:
        _loading_layers.reset(token)


def register_channel_layer(runner, get_method):
    layers = _loading_layers.get()
    if layers is not None:
        cfg = runner.moe_config
        comm = get_method(MoECommType.FUSED_MC2, cfg)
        if comm is None:
            return
        key = (
            runner.quant_type,
            cfg.num_experts,
            cfg.num_local_experts,
            cfg.experts_per_token,
            cfg.hidden_dim,
            cfg.intermediate_size_per_partition,
            comm.token_dispatcher.global_bs,
            comm.token_dispatcher.max_num_tokens_per_rank,
        )
        if key not in layers:
            layers[key] = ([], {kind: get_method(kind, cfg) for kind in MoECommType})
        layers[key][0].append(runner)


class ChannelWindow:
    def __init__(self, factory, group, params):
        self.factory = factory
        self.params = params
        self.owner = factory(group, **params)
        self.views = []

    def fits(self, params):
        owner = self.owner
        missing = object()
        ep_size = getattr(owner, "ep_world_size", 0)
        if not isinstance(ep_size, int) or ep_size <= 0:
            return False
        for field in ("num_experts", "num_topk", "num_max_tokens_per_rank"):
            if params[field] > self.params[field]:
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
        view = SimpleNamespace(**vars(self.owner))
        vars(view).update(params)
        self.views.append(view)
        return view

    def restore(self, group):
        owner = self.owner
        if group is owner.group:
            return
        with torch.inference_mode():
            replacement = self.factory(group, **self.params)
            for name in ("ep_world_size", "topo_type", "rank_num_per_server"):
                if getattr(replacement, name, None) != getattr(owner, name, None):
                    raise RuntimeError(f"MegaMoE {name} changed when restoring the channel window")
            if replacement.ccl_buffer_size < owner.ccl_buffer_size:
                raise RuntimeError("Restored MegaMoE channel window is smaller than its captured allocation")
            owner.context.copy_(replacement.context)
            for buffer in (owner, *self.views):
                for name in ("group", "rank_id", "group_name", "_ctx_manager", "ccl_buffer_size"):
                    setattr(buffer, name, getattr(replacement, name))
        torch.distributed.barrier(group=group, device_ids=[torch.npu.current_device()])


@torch.inference_mode(False)
def _agree(scope, local):
    decisions = [None] * scope.world_size
    torch.distributed.all_gather_object(decisions, local, group=scope.cpu_group)
    return decisions


@torch.inference_mode()
def bind_channel_layers(layers, target, scope, allow_sharing):
    # Use module membership after both loads. A draft can be instantiated
    # under the target VllmConfig and have exactly the same expert counts.
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, retain_channel_methods
    from vllm_ascend.ops.fused_moe.moe_utils import _get_cann_mega_moe_quant_settings

    target_modules = set(target.modules())
    entries = {}
    params_by_key = {}
    for shape, (runners, methods) in layers.items():
        for runner in runners:
            key = (runner in target_modules, *shape)
            entries.setdefault(key, []).append((runner, methods))
            if key not in params_by_key and runner.quant_type in A5_SUPPORT_MEGA_MOE_QUANT_TYPES:
                mode, dtype, _ = _get_cann_mega_moe_quant_settings(runner.quant_type)
                params_by_key[key] = methods[MoECommType.FUSED_MC2].mega_moe_buffer_params(mode, dtype)

    layout = [(key, params_by_key.get(key)) for key in entries]
    layout = (layout, allow_sharing)
    if any(peer != layout for peer in _agree(scope, layout)):
        raise RuntimeError("MegaMoE target/draft layouts differ across expert-parallel ranks")
    if not any(not key[0] for key in entries):
        return
    owner_key = next((key for key in params_by_key if key[0]), None)
    if owner_key is None:
        # A dense/non-MegaMoE target leaves the tag free for the draft.
        owner_key = next(iter(params_by_key), None)
    window = None
    if owner_key is not None:
        runner, methods = entries[owner_key][0]
        factory = methods[MoECommType.FUSED_MC2].get_symm_buffer_for_mega_moe
        window = getattr(scope, "mega_moe_channel_window", None)
        if window is None:
            window = ChannelWindow(factory, scope.device_group, params_by_key[owner_key])
            scope.mega_moe_channel_window = window
        elif window.params != params_by_key[owner_key] or window.owner.group is not scope.device_group:
            raise RuntimeError("The live MC2 group already owns a different MegaMoE channel window")
        # The registry is what the sleep backend enumerates. Keep a single
        # owner there even if a later layer replaced the original comm object.
        get_moe_comm_method(MoECommType.FUSED_MC2, runner.moe_config)._channel_window = window

    for key, matching_layers in entries.items():
        buffer = None
        if key == owner_key:
            buffer = window.owner
        elif allow_sharing and window is not None and key in params_by_key:
            params = params_by_key[key]
            try:
                fits = window.fits(params)
            except (ImportError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Could not verify the draft MegaMoE window size: %s", exc)
                fits = False
            if all(_agree(scope, fits)):
                buffer = window.view(params)
        if buffer is None:
            logger.info("Using ordinary MoE communication for %s layout %s", "target" if key[0] else "draft", key[1:])
        runner, methods = matching_layers[0]
        retain_channel_methods(runner.moe_config, (key[1].value, *key[2:]), methods)
        for runner, methods in matching_layers:
            runner.channel_mega_moe_fallback = buffer is None
            runner.routed_experts.channel_mega_moe_buffer = buffer
            runner.routed_experts.channel_moe_methods = methods
