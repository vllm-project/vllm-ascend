# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import pytest

from vllm_ascend.ops.fused_moe import zb_runtime


class TestResolveZbShmemUri:
    def setup_method(self) -> None:
        zb_runtime.set_zb_shmem_conf_store_uri(None)

    def teardown_method(self) -> None:
        zb_runtime.set_zb_shmem_conf_store_uri(None)

    def test_uses_reserved_conf_store_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)
        zb_runtime.set_zb_shmem_conf_store_uri("tcp://127.0.0.1:45289")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://127.0.0.1:45289"

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://10.0.0.1:29999")
        zb_runtime.set_zb_shmem_conf_store_uri("tcp://127.0.0.1:45289")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://10.0.0.1:29999"

    def test_missing_reservation_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)

        with pytest.raises(RuntimeError, match="conf-store URI is unavailable"):
            zb_runtime.resolve_zb_shmem_uri()


class TestParseTcpHostPort:
    def test_ipv4(self) -> None:
        assert zb_runtime._parse_tcp_host_port("tcp://127.0.0.1:29500") == ("127.0.0.1", 29500)

    def test_ipv6(self) -> None:
        assert zb_runtime._parse_tcp_host_port("tcp://[::1]:29500") == ("::1", 29500)


class TestValidateZbServingParallelConfig:
    def test_allows_dp_gt1_when_zb_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeConfig:
            enable_mc2_zb = True

        monkeypatch.setattr(zb_runtime, "get_ascend_config", lambda: FakeConfig())

        class ParallelConfig:
            data_parallel_size = 2

        zb_runtime.validate_zb_serving_parallel_config(ParallelConfig())

    def test_skips_when_zb_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeConfig:
            enable_mc2_zb = False

        monkeypatch.setattr(zb_runtime, "get_ascend_config", lambda: FakeConfig())

        class ParallelConfig:
            data_parallel_size = 2

        zb_runtime.validate_zb_serving_parallel_config(ParallelConfig())
