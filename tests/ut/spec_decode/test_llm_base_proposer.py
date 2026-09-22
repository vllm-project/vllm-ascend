#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode, KVTransferConfig, VllmConfig

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import check_kv_extra_config

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class TestCreateDraftVllmConfigPD:
    """Regression for the false "conflicting data parallel size" raised while
    reconstructing the drafter's draft config in a PD-disaggregated, internal-DP
    setup.

    For non-MoE internal DP, ``vllm/v1/engine/core.py`` sets
    ``data_parallel_size=1`` on each EngineCore's config, while the
    user-declared ``kv_transfer_config`` still carries ``prefill``/``decode``
    ``dp_size`` > 1. The drafter's ``replace()``-based draft config
    reconstruction re-runs ``check_kv_extra_config``, which compared those two
    and raised. The drafter never owns a KV connector (it is built from the
    main worker's ``vllm_config``), so
    ``AscendSpecDecodeBaseProposer._create_draft_vllm_config`` strips the
    ``prefill``/``decode`` sub-configs from the worker config around the
    ``super()`` call and restores it afterwards.
    """

    @staticmethod
    def _make_proposer(vllm_config, speculative_config) -> AscendSpecDecodeBaseProposer:
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.vllm_config = vllm_config
        proposer.speculative_config = speculative_config
        return proposer

    @staticmethod
    def _platform_check_only_kv(*args):
        # Reduce NPUPlatform.check_and_update_config to the subset this
        # regression is about: run check_kv_extra_config only, and only when a
        # kv_transfer_config is present (mirroring platform.py:483). Robust to
        # the (cfg) / (cls, cfg) calling convention of the patched classmethod.
        for cfg in args:
            if hasattr(cfg, "kv_transfer_config"):
                if cfg.kv_transfer_config is not None:
                    check_kv_extra_config(cfg)
                return

    def test_strips_prefill_decode_and_restores_worker_config(self):
        with patch(
            "vllm_ascend.platform.NPUPlatform.check_and_update_config",
            side_effect=self._platform_check_only_kv,
        ):
            cfg = VllmConfig()
            cfg.kv_transfer_config = KVTransferConfig(
                kv_connector="MooncakeConnectorV1",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "prefill": {"dp_size": 2, "tp_size": 1},
                    "decode": {"dp_size": 2, "tp_size": 1},
                    "keep_me": 1,
                },
            )
            # Simulate the non-field marker that platform init injects at
            # startup (platform.py sets _engine_id_patched=True). The drafter
            # sees this on the worker's kv_transfer_config; rebuilding it must
            # not choke on the injected attr.
            cfg.kv_transfer_config._engine_id_patched = True
            # Non-MoE per-rank EngineCore view (vllm/v1/engine/core.py sets
            # data_parallel_size=1). tp=1 keeps world_size=1 so the
            # reconstruction does not trip device-count selection.
            cfg.parallel_config.data_parallel_size = 1
            cfg.parallel_config.tensor_parallel_size = 1

            proposer = self._make_proposer(
                cfg,
                SimpleNamespace(moe_backend=None, attention_backend=None),
            )
            draft = proposer._create_draft_vllm_config()

        # The draft config no longer carries the PD sub-configs that triggered
        # the false positive; unrelated extra_config is preserved.
        extra = draft.kv_transfer_config.kv_connector_extra_config
        assert "prefill" not in extra
        assert "decode" not in extra
        assert extra.get("keep_me") == 1

        # The worker config is restored, so the main engine's connector config
        # is untouched.
        worker_extra = proposer.vllm_config.kv_transfer_config.kv_connector_extra_config
        assert "prefill" in worker_extra
        assert "decode" in worker_extra

    def test_handles_none_kv_connector_extra_config(self):
        """Defensive: ``kv_connector_extra_config`` defaults to ``{}`` but an
        override may leave it ``None``. The strip must not raise
        ``AttributeError`` on ``.items()`` in that case."""
        with patch(
            "vllm_ascend.platform.NPUPlatform.check_and_update_config",
            side_effect=self._platform_check_only_kv,
        ):
            cfg = VllmConfig()
            cfg.kv_transfer_config = KVTransferConfig(
                kv_connector="MooncakeConnectorV1",
                kv_role="kv_producer",
            )
            cfg.kv_transfer_config.kv_connector_extra_config = None
            cfg.kv_transfer_config._engine_id_patched = True
            cfg.parallel_config.data_parallel_size = 1
            cfg.parallel_config.tensor_parallel_size = 1

            proposer = self._make_proposer(
                cfg,
                SimpleNamespace(moe_backend=None, attention_backend=None),
            )
            draft = proposer._create_draft_vllm_config()

        # No prefill/decode in the (empty) result; worker config restored.
        assert draft.kv_transfer_config.kv_connector_extra_config == {}
        assert (
            proposer.vllm_config.kv_transfer_config.kv_connector_extra_config
            is None
        )
