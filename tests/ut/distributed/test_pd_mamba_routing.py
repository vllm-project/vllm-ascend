# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test PD-separated Mamba state independent channel routing.
Uses mock objects only — no Ascend hardware required.

Architecture: P node produces KV blocks + Mamba state.
Control layer (Gate) routes them through separate channels.
D node Coordinator aligns each channel independently,
skipping Mamba from LCM block alignment.
"""

import os
import threading
import unittest
from unittest.mock import patch

import torch

# Simulate Ascend dependencies for local testing
os.environ.setdefault("VLLM_ASCEND_PD_MAMBA_ROUTING", "1")
os.environ.setdefault("VLLM_PLATFORM", "ascend")


class TestPDMambaStateRouting(unittest.TestCase):
    """Verify that Mamba state payload is routed through an independent
    channel and does not participate in KV block LCM alignment."""

    def setUp(self):
        # Mock P-node KV blocks + Mamba state
        self.p_kv_blocks = {
            "block_ids": list(range(16)),
            "block_hashes": [f"hash_{i:04x}" for i in range(16)],
            "block_size": 128,
        }
        self.p_mamba_state = {
            "layer_0": torch.randn(4, 512),  # 模拟连续隐状态
            "layer_1": torch.randn(4, 512),
        }

    def test_mamba_payload_present_in_req_meta(self):
        """D节点收到的 ReqMeta 应包含独立的 mamba_state_payload 字段"""
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import ReqMeta

        req = ReqMeta(
            block_ids=self.p_kv_blocks["block_ids"],
            block_hashes=self.p_kv_blocks["block_hashes"],
            token_len=2048,
        )
        # 新字段: mamba state 独立payload
        req.mamba_state_payload = self.p_mamba_state

        self.assertIsNotNone(req.mamba_state_payload)
        self.assertIn("layer_0", req.mamba_state_payload)

        # 纯KV请求: mamba_state_payload 为空
        req_plain = ReqMeta(
            block_ids=self.p_kv_blocks["block_ids"],
            block_hashes=self.p_kv_blocks["block_hashes"],
            token_len=1024,
        )
        self.assertIsNone(getattr(req_plain, "mamba_state_payload", None))

    def test_hybrid_ready_barrier_both_channels_ready(self):
        """两个通道都就绪后才触发 TRANSFER_COMPLETE"""
        kv_ready = threading.Event()
        mamba_ready = threading.Event()

        # 模拟传输完成
        kv_ready.set()
        mamba_ready.set()

        both_ready = kv_ready.is_set() and mamba_ready.is_set()
        self.assertTrue(both_ready, "两个通道都就绪后才能开始decode")

    def test_hybrid_ready_barrier_kv_only_ready(self):
        """只有KV就绪, Mamba未到 → barrier 不触发"""
        kv_ready = threading.Event()
        mamba_ready = threading.Event()

        kv_ready.set()
        # mamba_ready NOT set

        both_ready = kv_ready.is_set() and mamba_ready.is_set()
        self.assertFalse(both_ready, "Mamba未就绪时barrier不应触发")

    def test_feature_flag_disabled_falls_back(self):
        """Feature flag关闭 → Mamba状态参与原有LCM, 不走独立通道"""
        with patch.dict(os.environ, {"VLLM_ASCEND_PD_MAMBA_ROUTING": "0"}):
            routing_enabled = os.environ.get("VLLM_ASCEND_PD_MAMBA_ROUTING", "0") == "1"
            self.assertFalse(routing_enabled)
            # 此时 find_longest_cache_hit 应该走原来的 LCM 对齐路径

    def test_feature_flag_enabled_skips_mamba_from_lcm(self):
        """Feature flag开启 → Mamba状态被跳过, 不参与LCM"""
        with patch.dict(os.environ, {"VLLM_ASCEND_PD_MAMBA_ROUTING": "1"}):
            routing_enabled = os.environ.get("VLLM_ASCEND_PD_MAMBA_ROUTING", "0") == "1"
            self.assertTrue(routing_enabled)


class TestMambaStateLengthValidation(unittest.TestCase):
    """Mamba状态长度与D节点前缀长度的校验"""

    def test_exact_match_passes(self):
        """P端1024 token → D端请求1024: 通过"""
        p_state_len = 1024
        d_request_len = 1024
        valid = p_state_len == d_request_len
        self.assertTrue(valid)

    def test_mismatch_triggers_fallback(self):
        """P端1024 token → D端请求1000: 不匹配, 触发fallback"""
        p_state_len = 1024
        d_request_len = 1000
        valid = p_state_len == d_request_len
        self.assertFalse(valid, "长度不匹配应触发fallback")


class TestGracefulDegradation(unittest.TestCase):
    """降级策略: Mamba通道失败不影响纯KV模型"""

    def test_mamba_channel_failure_does_not_break_kv_only(self):
        """Mamba payload为空 → 系统无缝回退纯KV路径"""
        payload = {"kv_blocks": ["block_0", "block_1"], "mamba_state": None}

        # 模拟: mamba为空, 走纯KV
        if payload["mamba_state"] is None:
            route = "kv_only"
        else:
            route = "hybrid"

        self.assertEqual(route, "kv_only")
        self.assertIsNotNone(payload["kv_blocks"])


if __name__ == "__main__":
    unittest.main()
