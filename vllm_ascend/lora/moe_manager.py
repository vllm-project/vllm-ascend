# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in MoE adapter packing and DP coordination on the upstream LoRA manager."""

import torch.distributed as dist
from vllm.distributed import get_dp_group
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


class AscendMoELoRAModelManager(LRUCacheLoRAModelManager):
    def _create_merged_loras_inplace(self, lora_model):
        # The upstream loader removes nonlocal experts before packing. A valid
        # subset adapter can consequently have no matrices on this EP rank.
        if not lora_model.loras and any(
            isinstance(module, FusedMoEWithLoRA) and module.use_ep for module in self.modules.values()
        ):
            return
        # The paired 2D packer requires complete expert triplets. Missing whole
        # targets represent zero deltas; malformed A/B pairs are not missing targets.
        if not self._enable_moe_shared_loras:
            for name, children in self.packed_modules.items():
                module = self.modules.get(name)
                if not isinstance(module, FusedMoEWithLoRA) or isinstance(module, FusedMoE3DWithLoRA):
                    continue
                children = self._restrict_to_local_experts(name, children)
                weights = [self._get_lora_layer_weights(lora_model, child) for child in children]
                present = [weight for weight in weights if weight is not None]
                if not present:
                    continue
                if any(weight.lora_a is None or weight.lora_b is None for weight in present):
                    raise ValueError(f"Incomplete LoRA A/B pair for Ascend MoE experts {name}.")
                ranks = {
                    child.rsplit(".", 1)[-1]: weight.rank
                    for child, weight in zip(children, weights, strict=True)
                    if weight is not None
                }
                intermediate = module.intermediate_size_per_partition * module.tp_size
                template = present[0].lora_a
                for child, weight in zip(children, weights, strict=True):
                    if weight is not None:
                        continue
                    projection = child.rsplit(".", 1)[-1]
                    rank = ranks.get(projection, lora_model.rank)
                    inputs, outputs = module.hidden_size, intermediate
                    if projection == "w2":
                        inputs, outputs = outputs, inputs
                    lora_model.loras[child] = LoRALayerWeights(
                        child, rank, rank, template.new_zeros((rank, inputs)), template.new_zeros((outputs, rank))
                    )
        super()._create_merged_loras_inplace(lora_model)


class AscendMoEWorkerLoRAManager(LRUCacheWorkerLoRAManager):
    _manager_cls = AscendMoELoRAModelManager

    def create_lora_manager(self, model, vllm_config):
        result = super().create_lora_manager(model, vllm_config)
        self.requires_dp_sync = vllm_config.parallel_config.data_parallel_size > 1 and any(
            isinstance(module, FusedMoEWithLoRA) for module in self._adapter_manager.modules.values()
        )
        return result

    def pin_adapter(self, adapter_id):
        if self.requires_dp_sync:
            raise ValueError("Ascend routed-expert LoRA does not support pinning across DP ranks.")
        return super().pin_adapter(adapter_id)

    def synchronize_requests(self, requests):
        # MoE receivers execute tokens from other DP engines. Their GPU slots
        # and local expert weights must represent the same adapters.
        group = get_dp_group()
        gathered = [None] * group.world_size
        dist.all_gather_object(gathered, list(requests), group=group.cpu_group)
        by_id = {}
        for rank_requests in gathered:
            for request in rank_requests:
                previous = by_id.get(request.lora_int_id)
                if previous is not None and (
                    previous.lora_path != request.lora_path
                    or previous.lora_name != request.lora_name
                    or previous.is_3d_lora_weight != request.is_3d_lora_weight
                ):
                    raise ValueError(f"Conflicting Ascend MoE LoRA identity {request.lora_int_id} across DP ranks.")
                if previous is None or request.load_inplace:
                    by_id[request.lora_int_id] = request
        manager = self._adapter_manager
        if len(by_id) > manager.lora_slots:
            raise ValueError(
                f"Ascend MoE LoRA requires {len(by_id)} adapters across DP ranks, "
                f"exceeding max_loras={manager.lora_slots}."
            )
        if not by_id:
            return
        identities = sorted(by_id)
        layout = identities + [None] * (manager.lora_slots - len(identities))
        if manager.lora_index_to_id != layout:
            for identity in tuple(manager.lora_index_to_id):
                if identity is not None:
                    manager.deactivate_adapter(identity)
        for identity in identities:
            self.add_adapter(by_id[identity])
