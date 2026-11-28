# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Read-only full parameter and adapter fingerprints for checkpoint reloads."""

import hashlib

import torch
import torch.distributed as dist
from bf16_scope import ScopeWorker
from vllm.distributed import get_dp_group, get_tp_group


def fingerprint(value):
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "meta": value.is_meta}
    if not value.is_meta:
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        result["sha256"] = hashlib.sha256(memoryview(raw)).hexdigest()
    return result


class ReloadWorker(ScopeWorker):
    def m31_reload_snapshot(self):
        model = self.model_runner.model
        params = {name.replace(".base_layer.", "."): fingerprint(tensor) for name, tensor in model.named_parameters()}
        manager = getattr(self.model_runner, "lora_manager", None)
        adapters = {}
        ids = []
        if manager is not None:
            manager = manager._adapter_manager
            ids = list(manager.lora_index_to_id)
            for name, layer in manager.modules.items():
                fields = (
                    ("lora_a_packed", "lora_b_packed")
                    if hasattr(layer, "lora_a_packed")
                    else (
                        "lora_a_stacked",
                        "lora_b_stacked",
                        "w13_lora_a_stacked",
                        "w13_lora_b_stacked",
                        "w2_lora_a_stacked",
                        "w2_lora_b_stacked",
                    )
                )
                for field in fields:
                    value = getattr(layer, field, None)
                    if value is None:
                        continue
                    values = (value,) if isinstance(value, torch.Tensor) else value
                    adapters[name + ":" + field] = [fingerprint(v) for v in values]
        return {"parameters": params, "adapter_buffers": adapters, "slot_ids": ids}

    def m31_reload_snapshot_all_dp(self):
        # The paired HTTP collective_rpc returns only the first DP engine's result.
        snapshot = self.m31_reload_snapshot()
        group = get_dp_group()
        snapshot["rank"] = [group.rank_in_group, get_tp_group().rank_in_group]
        gathered = [None] * group.world_size
        dist.all_gather_object(gathered, snapshot, group=group.cpu_group)
        return gathered
