# SPDX-License-Identifier: Apache-2.0
"""Adapt the upstream connectors that also inspect scalar specs on the worker."""

from functools import cache, wraps
from sys import modules

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_ascend.core.layerwise_cache_config import get_scheduler_spec, scheduler_geometry_config


def _patch_decode_bench(connector):
    module = modules[connector.__module__]
    original = module._get_startup_fill_group_ids

    @wraps(original)
    def fill_ids(kv_cache_config):
        ids = original(kv_cache_config)
        if not kv_cache_config.needs_kv_cache_zeroing:
            return ids
        groups = kv_cache_config.kv_cache_groups
        return {i for i in ids if not isinstance(get_scheduler_spec(groups[i].kv_cache_spec), AttentionSpec)}

    module._get_startup_fill_group_ids = fill_ids


def _patch_hidden_states(connector):
    original = connector._find_cache_kv_group_id.__func__

    @classmethod
    def find_group(cls, kv_cache_config):
        return original(cls, scheduler_geometry_config(kv_cache_config) if kv_cache_config is not None else None)

    connector._find_cache_kv_group_id = find_group


def _patch_mooncake_store(connector):
    original = connector._validate_kv_cache_config

    @staticmethod
    def validate(vllm_config, kv_cache_config):
        return original(vllm_config, scheduler_geometry_config(kv_cache_config))

    connector._validate_kv_cache_config = validate


def _patch_mooncake(connector):
    worker = modules[connector.__module__].MooncakeConnectorWorker

    def kernel_block_ids(self, block_ids):
        ratio = self._physical_blocks_per_logical_kv_block
        if ratio == 1:
            return block_ids
        offsets = np.arange(ratio)
        groups = self.kv_cache_config.transfer_groups
        return [
            ids
            if isinstance(get_scheduler_spec(groups[index].kv_cache_spec), MambaSpec)
            else (np.asarray(ids)[:, None] * ratio + offsets).reshape(-1).tolist()
            for index, ids in enumerate(block_ids)
        ]

    worker._logical_to_kernel_block_ids = kernel_block_ids


def _patched_loader(load, patch):
    @cache
    def load_once():
        connector = load()
        patch(connector)
        return connector

    return load_once


def register_layerwise_connector_patches():
    # Keep optional transfer libraries lazy. MultiConnector resolves its
    # children through these same loaders in each participating process.
    for name, patch in (
        ("DecodeBenchConnector", _patch_decode_bench),
        ("ExampleHiddenStatesConnector", _patch_hidden_states),
        ("MooncakeConnector", _patch_mooncake),
        ("MooncakeStoreConnector", _patch_mooncake_store),
    ):
        if name in KVConnectorFactory._registry:
            KVConnectorFactory._registry[name] = _patched_loader(KVConnectorFactory._registry[name], patch)
