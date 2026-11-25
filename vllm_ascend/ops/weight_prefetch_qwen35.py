# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 decode prefetch. State belongs to one loaded model on one device."""

import importlib
import re

import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType, decode_concurrency
from vllm_ascend.ops.weight_prefetch_config import NUM_HIDDEN_LAYERS

# 0 is NCHW, layout-equivalent to ND for 2-D weights; 2 is ND.
ND_FORMATS = (0, 2)
GDN_OUT_PROJ_SHAPE = (2048, 2048)
GDN_IN_PROJ_QKVZ_SHAPE = (6144, 2048)
FULL_ATTN_QKV_PROJ_SHAPE = (4608, 2048)
ROUTED_W2_SHAPE = (256, 256, 2048)
GDN_LAYER_KEY_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.linear_attn$")


def gdn_layer_index(layer_name):
    if not isinstance(layer_name, str):
        return None
    match = GDN_LAYER_KEY_RE.fullmatch(layer_name)
    return int(match.group(1)) if match else None


def current_controller():
    try:
        context = get_forward_context()
    except AssertionError:
        return None
    return getattr(getattr(context, "model_instance", None), "_qwen35_weight_prefetch", None)


def supported_weight(weight, shape):
    return (
        isinstance(weight, torch.Tensor)
        and weight.device.type == "npu"
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == shape
        and weight.is_contiguous()
        and weight.storage_offset() == 0
        and torch_npu.get_npu_format(weight) in ND_FORMATS
    )


class Qwen35WeightPrefetch:
    def __init__(self, config, device):
        self.config = config
        self.stream = torch.npu.Stream(device=device)
        self.next_inputs = {}
        self.pending_next_input = set()
        self.inline_next_input_join = False
        self.gdn_out_weights = {}
        self.next_gdn_out = {}
        self.pending_gdn_out = set()
        self.routed_weights = {}
        self.done = torch.empty((config.vector_cores, 4), dtype=torch.int64, device=device)
        self.extension = None
        if "routed_w2" in config.targets:
            self.extension = importlib.import_module("vllm_ascend.ops._qwen35_routed_prefetch")

    def can_run(self):
        return (
            _EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER
            and not _EXTRA_CTX.flash_comm_v1_enabled
            and decode_concurrency() >= self.config.min_decode_concurrency
        )

    def begin_next(self, layer_name, states):
        entry = self.next_inputs.get(layer_name)
        if entry is None or not self.can_run():
            return False
        projection, identity, address, shape, size, immediate = entry
        weight = projection.weight
        # Reloaded/replaced weights must be rebound before a new capture.
        if weight is not identity or weight.data_ptr() != address or tuple(weight.shape) != shape:
            return False
        main = torch.npu.current_stream()
        self.stream.wait_stream(main)
        with torch.npu.stream(self.stream):
            if self.config.mode == "full":
                torch_npu.npu_prefetch(weight, states, size, 0)
        if immediate:
            self.inline_next_input_join = True
        else:
            self.pending_next_input.add(layer_name)
        return True

    def settle_next_input(self):
        if self.inline_next_input_join:
            self.inline_next_input_join = False
            self.join()

    def join_next_input(self, layer_name):
        if layer_name in self.pending_next_input:
            self.pending_next_input.remove(layer_name)
            self.join()

    def join(self):
        torch.npu.current_stream().wait_stream(self.stream)

    def prepare_next_gdn_out(self, source_layer_name, dependency):
        target_layer_name = self.next_gdn_out.get(source_layer_name)
        if target_layer_name is None or not self.can_run():
            return None
        entry = self.gdn_out_weights.get(target_layer_name)
        if entry is None:
            return None
        projection, identity, address, shape, size = entry
        weight = projection.weight
        if weight is not identity or weight.data_ptr() != address or tuple(weight.shape) != shape:
            return None
        # Snapshot only the pre-collective dependency; submit CMO after the collective is enqueued.
        self.stream.wait_stream(torch.npu.current_stream())
        return target_layer_name, weight, dependency, size

    def commit_next_gdn_out(self, prepared):
        target_layer_name, weight, dependency, size = prepared
        with torch.npu.stream(self.stream):
            if self.config.mode == "full":
                torch_npu.npu_prefetch(weight, dependency, size, 0)
        self.pending_gdn_out.add(target_layer_name)

    def begin_gdn_out(self, layer_name, dependency):
        entry = self.gdn_out_weights.get(layer_name)
        if entry is None or not self.can_run() or layer_name in self.pending_gdn_out:
            return False
        projection, identity, address, shape, size = entry
        weight = projection.weight
        if weight is not identity or weight.data_ptr() != address or tuple(weight.shape) != shape:
            return False
        main = torch.npu.current_stream()
        self.stream.wait_stream(main)
        with torch.npu.stream(self.stream):
            if self.config.mode == "full":
                torch_npu.npu_prefetch(weight, dependency, size, 0)
        self.pending_gdn_out.add(layer_name)
        return True

    def join_gdn_out(self, layer_name):
        if layer_name in self.pending_gdn_out:
            self.pending_gdn_out.remove(layer_name)
            self.join()

    def begin_routed(self, weight, groups, dependency, position, group_list_type):
        if position != self.config.routed_start or group_list_type != 1 or not self.can_run():
            return False
        if not isinstance(weight, torch.Tensor) or self.routed_weights.get(id(weight)) is not weight:
            return False
        if groups.dtype != torch.int64 or groups.ndim != 1 or groups.numel() != weight.shape[0]:
            return False
        if not groups.is_contiguous() or groups.device != weight.device:
            return False
        main = torch.npu.current_stream()
        self.stream.wait_stream(main)
        with torch.npu.stream(self.stream):
            self.extension.warm(
                weight,
                groups,
                self.done,
                self.config.routed_w2_bytes,
                self.config.vector_cores,
                self.config.mode == "control",
            )
        return True


def _unbind(model, reason):
    logger.warning("Qwen3.5 prefetch disabled at bind: %s; using ordinary path", reason)
    if hasattr(model, "_qwen35_weight_prefetch"):
        delattr(model, "_qwen35_weight_prefetch")


def bind_qwen35_weight_prefetch(model, vllm_config, config):
    """Bind post-loader TP storage before any graph is captured."""
    if not config.enabled:
        return
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    parallel = vllm_config.parallel_config
    hf = vllm_config.model_config.hf_text_config
    if (
        get_ascend_device_type() != AscendDeviceType.A2
        or getattr(hf, "model_type", None) != "qwen3_5_moe_text"
        or getattr(hf, "num_hidden_layers", None) != NUM_HIDDEN_LAYERS
        or parallel.tensor_parallel_size != 2
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_expert_parallel
        or vllm_config.quant_config is not None
        or vllm_config.lora_config is not None
        or vllm_config.speculative_config is not None
    ):
        logger.warning("Qwen3.5 prefetch unsupported device/model/parallel/quantization; using ordinary path")
        return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    layers = sorted((m for m in model.modules() if isinstance(m, Qwen3_5DecoderLayer)), key=lambda m: m.layer_idx)
    if [layer.layer_idx for layer in layers] != list(range(NUM_HIDDEN_LAYERS)):
        logger.warning("Qwen3.5 prefetch requires all %d local layers; using ordinary path", NUM_HIDDEN_LAYERS)
        return
    device = next(model.parameters()).device
    controller = Qwen35WeightPrefetch(config, device)
    # Plain model-owned object: no module cycles or persistent checkpoint changes.
    model._qwen35_weight_prefetch = controller
    for index, layer in enumerate(layers):
        if config.layers is not None and index not in config.layers:
            continue
        if "gdn_out" in config.targets and layer.layer_type == "linear_attention":
            attention = layer.linear_attn
            layer_name = getattr(attention, "prefix", "")
            if gdn_layer_index(layer_name) == index:
                projection = attention.out_proj
                weight = projection.weight
                if supported_weight(weight, GDN_OUT_PROJ_SHAPE):
                    if config.gdn_out_bytes > weight.numel() * weight.element_size():
                        raise ValueError(f"gdn_out_bytes exceeds weight storage at layer {index}")
                    controller.gdn_out_weights[layer_name] = (
                        projection,
                        weight,
                        weight.data_ptr(),
                        GDN_OUT_PROJ_SHAPE,
                        config.gdn_out_bytes,
                    )
                    attention._qwen35_gdn_out_prefetch_layer_name = layer_name
                    logger.info(
                        "Qwen3.5 GDN out_proj prefetch storage layer=%d name=%s shape=%s format=%d bytes=%d",
                        index,
                        layer_name,
                        GDN_OUT_PROJ_SHAPE,
                        torch_npu.get_npu_format(weight),
                        config.gdn_out_bytes,
                    )
                else:
                    logger.warning("Layer %d GDN out_proj is not supported BF16 ND TP2 storage", index)
            else:
                logger.warning("Layer %d GDN prefix %r is not a main-model layer name", index, layer_name)

        if not ("next_input" in config.targets or "routed_w2" in config.targets):
            continue
        owners = [m for m in layer.mlp.modules() if isinstance(m, AscendMoERunner)]
        if len(owners) != 1:
            _unbind(model, f"expected one AscendMoERunner at layer {index}, found {len(owners)}")
            return
        owner = owners[0]
        if "routed_w2" in config.targets:
            weight = owner.routed_experts.w2_weight
            if supported_weight(weight, ROUTED_W2_SHAPE):
                if config.routed_w2_bytes > weight.numel() * weight.element_size():
                    raise ValueError("routed_w2_bytes exceeds bound weight storage")
                controller.routed_weights[id(weight)] = weight
                logger.info(
                    "Qwen3.5 routed prefetch storage layer=%d shape=%s format=%d expert_bytes=%d",
                    index,
                    tuple(weight.shape),
                    torch_npu.get_npu_format(weight),
                    weight[0].numel() * weight.element_size(),
                )
            else:
                logger.warning("Layer %d routed w2 is not supported BF16 ND TP2 storage", index)
        if "next_input" not in config.targets or index == len(layers) - 1:
            continue
        target = layers[index + 1]
        if target.layer_type not in config.next_layer_types:
            continue
        if target.layer_type == "linear_attention":
            projection, shape = target.linear_attn.in_proj_qkvz, GDN_IN_PROJ_QKVZ_SHAPE
        else:
            projection, shape = target.self_attn.qkv_proj, FULL_ATTN_QKV_PROJ_SHAPE
        weight = projection.weight
        if not supported_weight(weight, shape):
            logger.warning("Layer %d next input is not supported BF16 ND TP2 storage", index)
            continue
        size = config.next_input_bytes
        if size == "full":
            size = weight.numel() * weight.element_size()
        if size > weight.numel() * weight.element_size():
            raise ValueError(f"next_input_bytes exceeds storage of layer {index + 1}")
        # Deferred join keeps the prefetch running through residual/norm until the
        # consumer layer joins right before its input projection; only linear
        # attention targets have a bind-known join point.
        deferred = config.next_input_join == "deferred" and target.layer_type == "linear_attention"
        controller.next_inputs[owner.layer_name] = (projection, weight, weight.data_ptr(), shape, size, not deferred)
        owner._qwen35_prefetch_layer_name = owner.layer_name
        if deferred:
            target.linear_attn._qwen35_next_input_prefetch_join_key = owner.layer_name
        logger.info(
            "Qwen3.5 next prefetch storage source=%d target=%d shape=%s format=%d bytes=%d join=%s",
            index,
            index + 1,
            shape,
            torch_npu.get_npu_format(weight),
            size,
            "deferred" if deferred else "immediate",
        )
    # When next_input is enabled, its existing immediate join shares this stream;
    # keep gdn_out on its same-layer path rather than joining it too early.
    if "gdn_out" in config.targets and "next_input" not in config.targets:
        for index, target_layer in enumerate(layers[1:], start=1):
            if target_layer.layer_type != "linear_attention":
                continue
            target_name = getattr(target_layer.linear_attn, "_qwen35_gdn_out_prefetch_layer_name", "")
            if not target_name:
                continue
            source_layer = layers[index - 1]
            owners = [m for m in source_layer.mlp.modules() if isinstance(m, AscendMoERunner)]
            if len(owners) != 1:
                _unbind(model, f"expected one AscendMoERunner at layer {index - 1}, found {len(owners)}")
                return
            owner = owners[0]
            controller.next_gdn_out[owner.layer_name] = target_name
            owner._qwen35_prefetch_layer_name = owner.layer_name
            logger.info("Qwen3.5 GDN out_proj prefetch submit source=%d target=%d", index - 1, index)
    logger.info(
        "Qwen3.5 prefetch bound: next_inputs=%d routed_w2=%d gdn_out=%d mode=%s join=%s budgets=%s/%s/%s",
        len(controller.next_inputs),
        len(controller.routed_weights),
        len(controller.gdn_out_weights),
        config.mode,
        config.next_input_join,
        config.next_input_bytes,
        config.routed_w2_bytes,
        config.gdn_out_bytes,
    )
