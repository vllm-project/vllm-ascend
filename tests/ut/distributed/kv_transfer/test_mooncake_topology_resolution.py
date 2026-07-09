# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for Mooncake connector remote-topology auto-discovery.

Covers the fixes that make ``kv_connector_extra_config`` optional for
PD separation with heterogeneous TP (e.g. DeepSeek-V4 P(tp4)/D(tp1)):

1. ``_resolve_parallel_sizes`` must run after ``self.kv_role`` is assigned
   (regression: previously called as the first line of ``__init__``).
2. D-side discovers P's tp_size from per-request ``meta.remote_ptp_size``,
   not from the ZMQ metadata cache (which is empty before the first recv).
3. ``check_kv_extra_config`` raises on local-role config mismatch but
   tolerates a missing config (auto-discovery path).
"""

import types
import unittest
from unittest.mock import MagicMock

import torch

if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace(Event=type("Event", (), {}))  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorWorker,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    MooncakeConnectorWorker as MooncakeHybridConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    ReqMeta as HybridReqMeta,
)
from vllm_ascend.utils import check_kv_extra_config


def _make_vllm_config(kv_role: str, tp_size: int, dp_size: int, extra_config: dict):
    """Build a minimal mock VllmConfig for _resolve_parallel_sizes."""
    cfg = MagicMock()
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.kv_transfer_config.kv_role = kv_role
    cfg.kv_transfer_config.is_kv_producer = kv_role == "kv_producer"
    cfg.kv_transfer_config.is_kv_consumer = kv_role == "kv_consumer"
    cfg.kv_transfer_config.kv_connector_extra_config = extra_config
    cfg.kv_transfer_config.get_from_extra_config = lambda key, default=None: extra_config.get(
        key, default if default is not None else {}
    )
    return cfg


class TestResolveParallelSizesOrder(unittest.TestCase):
    """_resolve_parallel_sizes branches on self.kv_role, so it must be
    called only after self.kv_role is assigned in __init__.

    Previously it was the first line of __init__ and crashed with
    AttributeError. We verify the method itself no longer references
    attributes that __init__ hasn't set yet, by invoking it directly with
    kv_role pre-set on the instance.
    """

    def test_producer_resolves_own_prefill_tp(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_producer"
        cfg = _make_vllm_config("kv_producer", tp_size=4, dp_size=4, extra_config={})
        worker._resolve_parallel_sizes(cfg)
        self.assertEqual(worker._prefill_tp_size, 4)
        self.assertEqual(worker._prefill_dp_size, 4)
        # Remote (decode) unknown yet -> default 1, not resolved.
        self.assertEqual(worker._decode_tp_size, 1)
        self.assertFalse(worker._remote_sizes_resolved)

    def test_consumer_resolves_own_decode_tp(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        cfg = _make_vllm_config("kv_consumer", tp_size=1, dp_size=16, extra_config={})
        worker._resolve_parallel_sizes(cfg)
        self.assertEqual(worker._decode_tp_size, 1)
        self.assertEqual(worker._decode_dp_size, 16)
        self.assertEqual(worker._prefill_tp_size, 1)
        self.assertFalse(worker._remote_sizes_resolved)


class TestDSideDiscoversPrefillTpFromMeta(unittest.TestCase):
    """D-side must read P's tp_size from meta.remote_ptp_size so that
    heterogeneous TP works without kv_connector_extra_config and without
    waiting for the ZMQ metadata cache to be populated."""

    def test_d_side_uses_remote_ptp_size(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker._prefill_tp_size = 1
        worker._prefill_dp_size = 1
        worker._remote_sizes_resolved = False

        meta = ReqMeta(
            local_block_ids=[],
            num_external_tokens=0,
            num_computed_tokens=0,
            remote_block_ids=[],
            remote_host="127.0.0.1",
            remote_port=30100,
            remote_engine_id="1",
            remote_request_id="req-1",
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_ptp_size=4,
            remote_multi_nodes_meta_mapping={},
            num_prompt_blocks=0,
        )
        worker._update_remote_sizes_from_metadata(meta)
        self.assertEqual(worker._prefill_tp_size, 4)
        self.assertTrue(worker._remote_sizes_resolved)

    def test_d_side_ignores_meta_when_already_resolved(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker._prefill_tp_size = 8  # already set via extra_config
        worker._prefill_dp_size = 8
        worker._remote_sizes_resolved = True

        meta = ReqMeta(
            local_block_ids=[],
            num_external_tokens=0,
            num_computed_tokens=0,
            remote_block_ids=[],
            remote_host="127.0.0.1",
            remote_port=30100,
            remote_engine_id="1",
            remote_request_id="req-1",
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_ptp_size=4,
            remote_multi_nodes_meta_mapping={},
            num_prompt_blocks=0,
        )
        worker._update_remote_sizes_from_metadata(meta)
        # Unchanged: extra_config value wins.
        self.assertEqual(worker._prefill_tp_size, 8)
        self.assertTrue(worker._remote_sizes_resolved)

    def test_d_side_no_meta_stays_unresolved(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker._prefill_tp_size = 1
        worker._prefill_dp_size = 1
        worker._remote_sizes_resolved = False
        worker._update_remote_sizes_from_metadata(None)
        self.assertFalse(worker._remote_sizes_resolved)
        self.assertEqual(worker._prefill_tp_size, 1)


class TestPSideFallbackToExtraConfig(unittest.TestCase):
    """When extra_config provides decode topology, P-side uses it directly
    and marks _remote_sizes_resolved (no need for DONE_RECVING_MSG)."""

    def test_p_side_uses_decode_extra_config(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_producer"
        cfg = _make_vllm_config(
            "kv_producer",
            tp_size=4,
            dp_size=4,
            extra_config={"decode": {"tp_size": 1, "dp_size": 16}},
        )
        worker._resolve_parallel_sizes(cfg)
        self.assertEqual(worker._decode_tp_size, 1)
        self.assertEqual(worker._decode_dp_size, 16)
        self.assertTrue(worker._remote_sizes_resolved)

    def test_p_side_rejects_prefill_smaller_than_decode(self):
        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_producer"
        # P tp=2 but config claims decode tp=4 -> invalid.
        cfg = _make_vllm_config(
            "kv_producer",
            tp_size=2,
            dp_size=4,
            extra_config={"prefill": {"tp_size": 2}, "decode": {"tp_size": 4}},
        )
        with self.assertRaises(ValueError):
            worker._resolve_parallel_sizes(cfg)


class TestHybridConnectorDiscovery(unittest.TestCase):
    """Same fixes applied to MooncakeHybridConnector."""

    def test_hybrid_d_side_uses_remote_ptp_size(self):
        worker = MooncakeHybridConnectorWorker.__new__(MooncakeHybridConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker._prefill_tp_size = 1
        worker._prefill_dp_size = 1
        worker._remote_sizes_resolved = False

        meta = HybridReqMeta(
            local_block_ids=[],
            num_external_tokens=0,
            remote_block_ids=[],
            remote_host="127.0.0.1",
            remote_port=30100,
            remote_engine_id="1",
            remote_request_id="req-1",
            remote_ptp_size=4,
            remote_multi_nodes_meta_mapping={},
            num_prompt_blocks=0,
        )
        worker._update_remote_sizes_from_metadata(meta)
        self.assertEqual(worker._prefill_tp_size, 4)
        self.assertTrue(worker._remote_sizes_resolved)


class TestCheckKvExtraConfig(unittest.TestCase):
    """check_kv_extra_config raises on local-role mismatch, tolerates
    missing config (auto-discovery), and does not validate the remote
    role's config values."""

    def _make(self, kv_role: str, tp_size: int, dp_size: int, extra_config: dict):
        cfg = MagicMock()
        cfg.parallel_config.tensor_parallel_size = tp_size
        cfg.parallel_config.data_parallel_size = dp_size
        cfg.kv_transfer_config.is_kv_producer = kv_role == "kv_producer"
        cfg.kv_transfer_config.is_kv_consumer = kv_role == "kv_consumer"
        cfg.kv_transfer_config.get_from_extra_config = lambda key, default=None: extra_config.get(
            key, default if default is not None else {}
        )
        return cfg

    def test_missing_config_does_not_raise(self):
        cfg = self._make("kv_producer", tp_size=4, dp_size=4, extra_config={})
        # Should not raise; auto-discovery path.
        check_kv_extra_config(cfg)

    def test_local_tp_mismatch_raises(self):
        # Producer's local prefill config tp must match local tp.
        cfg = self._make(
            "kv_producer",
            tp_size=4,
            dp_size=4,
            extra_config={"prefill": {"tp_size": 8}},
        )
        with self.assertRaises(ValueError):
            check_kv_extra_config(cfg)

    def test_local_dp_mismatch_raises(self):
        cfg = self._make(
            "kv_consumer",
            tp_size=1,
            dp_size=16,
            extra_config={"decode": {"dp_size": 8}},
        )
        with self.assertRaises(ValueError):
            check_kv_extra_config(cfg)

    def test_local_config_matches_does_not_raise(self):
        cfg = self._make(
            "kv_producer",
            tp_size=4,
            dp_size=4,
            extra_config={"prefill": {"tp_size": 4, "dp_size": 4}},
        )
        check_kv_extra_config(cfg)


if __name__ == "__main__":
    unittest.main()
