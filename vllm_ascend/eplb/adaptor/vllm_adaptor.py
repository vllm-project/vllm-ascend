#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this adaptor.
import hashlib
import json
import os
from inspect import getattr_static
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.tensor_ops import build_expert_replica_routing_table
from vllm_ascend.quantization.quant_type import QuantType

EPLB_EXPERT_WEIGHT_NAMES = {
    (QuantType.NONE, False): ("w13_weight", "w2_weight"),
    (QuantType.NONE, True): ("w13_weight_list", "w2_weight_list"),
    (QuantType.W8A8, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
    ),
    (QuantType.W8A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
        "fused_w1_scale_list",
        "fused_w2_scale_list",
    ),
    (QuantType.W4A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
        "w13_scale_bias_list",
        "w2_scale_bias_list",
    ),
    (QuantType.W4A8, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
        "w13_scale_bias_list",
        "w2_scale_bias_list",
    ),
    (QuantType.W4A8MXFP, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
    ),
    (QuantType.W4A8MXFP, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
    ),
    (QuantType.W4A4MXFP, False): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W4A4MXFP, True): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W8A8MXFP, False): ("w13_weight_list", "w2_weight_list", "w13_weight_scale_list", "w2_weight_scale_list"),
    (QuantType.W8A8MXFP, True): ("w13_weight_list", "w2_weight_list", "w13_weight_scale_list", "w2_weight_scale_list"),
}

_W4A8MXFP_NZ_WEIGHT_NAMES = {"w13_weight_list", "w2_weight_list"}


def _new_w4a8_mxfp_nz_buffer(expert_tensor: torch.Tensor) -> torch.Tensor:
    """Create an offset-0 NZ receive buffer matching an MXFP expert."""
    buffer_tensor = torch.empty(expert_tensor.shape, dtype=expert_tensor.dtype, device=expert_tensor.device)
    input_dtype = (
        torch.float4_e2m1fn_x2 if expert_tensor.dtype == torch.float4_e2m1fn_x2 else torch_npu.float4_e2m1fn_x2
    )
    torch_npu.npu_format_cast_(
        buffer_tensor,
        29,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=input_dtype,
    )
    return buffer_tensor


class VllmEplbAdaptor:
    _registered_moe_layers: list["torch.nn.Module"] = []

    @staticmethod
    def register_layer(layer: "torch.nn.Module") -> None:
        """Register a MoE layer for EPLB. Called during layer initialization.

        Only real layers call this; PPMissingLayer won't, so the registry
        naturally contains only layers on this PP rank.
        """
        VllmEplbAdaptor._registered_moe_layers.append(layer)

    def __init__(self, model, **args):
        super().__init__(**args)
        if hasattr(model, "language_model"):
            self.model = model.language_model
            self.config = model.config.get_text_config()
        else:
            self.model = model
            self.config = model.config
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.config, "first_k_dense_replace", 0)

        self.moe_layers = VllmEplbAdaptor._registered_moe_layers
        self.num_moe_layers = len(self.moe_layers)

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        # Get num_local_experts from first real MoE layer
        first_layer = self.moe_layers[0]
        self.num_local_experts = first_layer.local_num_experts
        self.ep_rank = first_layer.ep_rank
        # Identical for every layer (placement depends only on the EPLB config);
        # None when EPLB expert maps are not used (no EP). Goes stale after the
        # first rebalance; verify dumps use per-layer phys_to_logical_fresh.
        self.phys_to_logical = getattr(first_layer, "phys_to_logical", None)

        self.expert_param_per_layer = dict()
        self.expert_weight_key_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = self.num_local_experts
        self.buffer_tensor_list: dict[Any, list[list[Any]]] = dict()
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        self.log2phy_rank_independent_map_per_layer = dict()
        for local_idx, layer in enumerate(self.moe_layers):
            self.log2phy_map_per_layer[local_idx] = layer.get_log2phy_map()
            self.log2phy_rank_independent_map_per_layer[local_idx] = layer.get_log2phy_rank_independent_map()

        # Experimental correctness-check hook; a no-op unless EPLB_VERIFY_DIR
        # is set in the environment.
        self.maybe_verify_dump(tag="init")

    def maybe_verify_dump(self, tag: str | None = None):
        """Dump expert placement and weight hashes for offline verification.

        This is an opt-in diagnostics hook for validating dynamic EPLB, not a
        serving feature. It does nothing unless ``EPLB_VERIFY_DIR`` is set. At
        ``tag="init"`` (right after model load) and at the end of each cycle
        listed in ``EPLB_VERIFY_AT`` it writes one JSON per rank holding every
        layer's physical-id -> local-slot map, the logical -> physical route
        map, and SHA-256 hashes of the expert tensors of the layers listed in
        ``EPLB_VERIFY_LAYERS`` (comma separated). The offline checker uses it
        to prove that each physical expert keeps exactly its own weights
        across rebalances and that the route map still resolves every logical
        expert to those weights.
        """
        outdir = os.environ.get("EPLB_VERIFY_DIR")
        if not outdir:
            return
        try:
            if tag is None:
                cycle = getattr(self, "_verify_cycle", 0) + 1
                self._verify_cycle = cycle
                at = {int(x) for x in os.environ.get("EPLB_VERIFY_AT", "1").split(",") if x.strip()}
                if cycle not in at:
                    return
                tag = f"cycle{cycle:03d}"

            hash_layers = {int(x) for x in os.environ.get("EPLB_VERIFY_LAYERS", "0").split(",") if x.strip()}
            record = {
                "tag": tag,
                "rank": self.rank_id,
                "ep_rank": self.ep_rank,
                "num_moe_layers": self.num_moe_layers,
                "num_local_experts": self.num_local_experts,
                "phys_to_logical": self.phys_to_logical.cpu().tolist() if self.phys_to_logical is not None else None,
            }
            layers = {}
            for local_idx, layer in enumerate(self.moe_layers):
                entry = {}
                ascend_map = layer.ascend_expert_map
                if ascend_map is not None:
                    entry["ascend_expert_map"] = ascend_map.detach().cpu().tolist()
                local_phys_ids = layer.local_phys_expert_ids
                if local_phys_ids is not None:
                    entry["local_phys_expert_ids"] = local_phys_ids.detach().cpu().tolist()
                fresh_p2l = getattr(layer, "phys_to_logical_fresh", None)
                if fresh_p2l is not None:
                    entry["phys_to_logical_fresh"] = fresh_p2l.tolist()
                log2phy = layer.get_log2phy_map()
                if log2phy is not None:
                    entry["log2phy"] = log2phy.detach().cpu().tolist()
                route_table = getattr(layer, "allgather_replica_routing_table", None)
                if route_table is not None:
                    cpu_table = route_table.detach().cpu()
                    entry["allgather_routing_table_sha256"] = hashlib.sha256(cpu_table.numpy().tobytes()).hexdigest()
                    entry["allgather_routing_table_rows0to7"] = cpu_table[:8].tolist()
                    # Per-slot token counts: with token-row routing both copies
                    # of a duplicated expert must show non-zero load.
                    moe_load = getattr(layer, "moe_load", None)
                    if moe_load is not None:
                        entry["moe_load"] = moe_load.detach().cpu().tolist()
                if local_idx in hash_layers:
                    entry["hashes"] = []
                    for slot in range(self.num_local_experts):
                        tensors = []
                        for tensor in self.expert_param_per_layer[local_idx][slot]:
                            cpu_tensor = tensor.detach().cpu().contiguous()
                            try:
                                raw = cpu_tensor.numpy().tobytes()
                            except TypeError:
                                # bfloat16 has no numpy dtype; hash the raw bytes.
                                raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
                            tensors.append(
                                {
                                    "shape": list(cpu_tensor.shape),
                                    "dtype": str(cpu_tensor.dtype),
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                }
                            )
                        entry["hashes"].append(tensors)
                layers[str(local_idx)] = entry
            record["layers"] = layers

            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"rank{self.rank_id}_{tag}.json")
            with open(path, "w") as f:
                json.dump(record, f)
            logger.info("[eplb/verify] dumped %s", path)
        except Exception:
            logger.exception("[eplb/verify] dump failed")

    def init_buffer_tensor(self, num_buffer_tensor):
        buffer_tensor_shapes: dict[Any, list[torch.Size]] = dict()
        for local_idx, _ in enumerate(self.moe_layers):
            expert_weight_key = self.expert_weight_key_per_layer[local_idx]
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            expert_tensors = [self.param_dict[f"{local_idx}.{name}"][0] for name in expert_weight_names]
            expert_tensor_shapes = [tensor.shape for tensor in expert_tensors]
            if expert_weight_key in self.buffer_tensor_list:
                assert expert_tensor_shapes == buffer_tensor_shapes[expert_weight_key], (
                    f"EPLB expert weight shapes mismatch for {expert_weight_key}: "
                    f"expected {buffer_tensor_shapes[expert_weight_key]}, got {expert_tensor_shapes}"
                )
                continue
            buffer_tensor_shapes[expert_weight_key] = expert_tensor_shapes
            self.buffer_tensor_list[expert_weight_key] = [[] for _ in range(num_buffer_tensor)]
            for buffer_id in range(num_buffer_tensor):
                for name, expert_tensor in zip(expert_weight_names, expert_tensors):
                    if expert_weight_key[0] == QuantType.W4A8MXFP and name in _W4A8MXFP_NZ_WEIGHT_NAMES:
                        buffer_tensor = _new_w4a8_mxfp_nz_buffer(expert_tensor)
                    else:
                        buffer_tensor = torch.empty_like(expert_tensor)
                    self.buffer_tensor_list[expert_weight_key][buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        self.param_dict = dict()

        for local_idx, layer in enumerate(self.moe_layers):
            quant_type = QuantType.NONE if self.model.quant_config is None else layer.quant_type
            expert_weight_key = (quant_type, get_ascend_config().enable_fused_mc2 == 1)
            if expert_weight_key not in EPLB_EXPERT_WEIGHT_NAMES:
                raise ValueError(f"EPLB not support {quant_type} with fused MC2 {expert_weight_key[1]}")
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            self.expert_weight_key_per_layer[local_idx] = expert_weight_key
            self.expert_param_per_layer[local_idx] = list()
            for name in expert_weight_names:
                param_key = f"{local_idx}.{name}"
                # Inspect the attribute without invoking __getattr__. Besides
                # avoiding side effects from dynamic proxies, this prevents an
                # unspecced MagicMock from fabricating the optional accessor.
                # The refactored MoERunner declares the accessor because its
                # RoutedExperts child owns weights; legacy layers expose them
                # directly.
                get_parameter = (
                    layer.get_eplb_parameter if getattr_static(layer, "get_eplb_parameter", None) is not None else None
                )
                self.param_dict[param_key] = get_parameter(name) if get_parameter is not None else getattr(layer, name)
            for local_expert_id in range(self.num_local_experts):
                per_expert_param = list()
                for name in expert_weight_names:
                    per_expert_param.append(self.param_dict[f"{local_idx}.{name}"][local_expert_id])
                self.expert_param_per_layer[local_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        loads = [layer.moe_load for layer in self.moe_layers]
        self.moe_load = torch.stack(loads, dim=0) if loads else torch.empty(0)
        return self.moe_load

    def clear_all_moe_loads(self):
        for layer in self.moe_layers:
            layer.clear_moe_load()

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        if self.rank_id == 0:
            num_local_experts = expert_maps.max() + 1
            phys_to_logical = self.phys_to_logical

            expert_maps_list = expert_maps.tolist()
            record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}

            for layer_idx, layer_data in enumerate(expert_maps_list):
                layer_record: dict[str, Any] = {
                    "layer_id": layer_idx,
                    "device_count": len(layer_data),
                    "device_list": [],
                }

                for device_idx, experts in enumerate(layer_data):
                    # expert maps index physical expert IDs; the record file
                    # format stores logical expert IDs per device.
                    phys_ids = [experts.index(i) for i in range(num_local_experts)]
                    if phys_to_logical is None:
                        placement = phys_ids
                    else:
                        placement = [int(phys_to_logical[phys_id].item()) for phys_id in phys_ids]
                    device_record = {"device_id": device_idx, "device_expert": placement}
                    layer_record["device_list"].append(device_record)

                record["layer_list"].append(layer_record)

            with open(expert_map_record_path, "w") as f:
                json.dump(record, f, indent=4)

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)
        # Keep the layer's runtime maps in sync with the worker's authoritative
        # map after each rebalance: the AllGather dispatcher masks topk_ids
        # against ascend_expert_map and the load collector reads the layer's
        # slot->physical IDs.
        layer = self.moe_layers[layer_id]
        layer.update_ascend_eplb_maps(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        expert_weight_key = self.expert_weight_key_per_layer[layer_id]
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace],
            self.buffer_tensor_list[expert_weight_key][buffer_tensor_id],
        ):
            expert_tensor.copy_(buffer_tensor)
            logger.debug("Expert tensor shape is :%s", expert_tensor.shape)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map, updated_log2phy_rank_independent_map=None):
        log2phy = self.log2phy_map_per_layer[layer_id]
        if log2phy is not None:
            if updated_log2phy_map.device.type == "cpu":
                updated_log2phy_map = updated_log2phy_map.to(dtype=log2phy.dtype).pin_memory()
            log2phy.copy_(updated_log2phy_map, non_blocking=True)
        rank_independent = self.log2phy_rank_independent_map_per_layer.get(layer_id)
        if rank_independent is not None and updated_log2phy_rank_independent_map is not None:
            rank_independent.copy_(updated_log2phy_rank_independent_map)

    def do_update_replica_routing_table(self, layer_id, logical_to_physical_map, logical_replica_count):
        """Refresh the ALLGATHER token-row replica routing table after a rebalance.

        The worker ships the compact group-common ``(logical_to_physical_map,
        logical_replica_count)`` payload; expand it into the [1024, N] routing
        table on the execution device and copy it in place so the tensor
        identity seen by captured ACL graphs is preserved.

        The table MUST be built with ``ep_rank=0``: every rank resolves the
        same (token row, logical expert) to the same flat physical id, which
        is exactly what keeps ALLGATHER execution exactly-once. Building it
        with a rank-dependent rotation silently double-counts one replica and
        drops the other (same failure as a rank-dependent log2phy).
        """
        layer = self.moe_layers[layer_id]
        table = getattr(layer, "allgather_replica_routing_table", None)
        if table is None:
            # Token-row routing was disabled at layer init; nothing to refresh.
            return
        device = table.device
        new_table = build_expert_replica_routing_table(
            logical_to_physical_map.to(device=device),
            logical_replica_count.to(device=device),
            ep_rank=0,
        )
        table.copy_(new_table, non_blocking=True)
        # Stash the fresh placement on the layer for the opt-in verify dump.
        # Placements are per-layer (each layer rebalances independently), so a
        # global stash would mix another layer's placement; layer 0's table and
        # layer 0's stash are refreshed in this same call, keeping them atomic.
        l2p = logical_to_physical_map.cpu()
        counts = logical_replica_count.cpu()
        width = int(counts.max().item())
        used = l2p[:, :width]
        fresh = torch.full((int(used.max().item()) + 1,), -1, dtype=torch.int64)
        fresh[used.reshape(-1)] = torch.arange(l2p.shape[0], dtype=torch.int64).repeat_interleave(width)
        layer.phys_to_logical_fresh = fresh

    def get_global_expert_map(self):
        all_layer_global_expert_map = []
        for local_idx, layer in enumerate(self.moe_layers):
            map_cpu = layer.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[local_idx] = map_cpu[self.ep_rank]

        return torch.stack(all_layer_global_expert_map)
