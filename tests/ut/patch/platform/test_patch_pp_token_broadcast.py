# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the no-consumer PP token broadcast skip.

``patch_pp_token_broadcast`` patches ``compute_need_sampled_mask`` so that
on a pure disaggregated-prefill node the mask returns ``None`` whenever
every sample-producing request is a final prefill chunk. Because
``PPHandler.receive``/``broadcast``/``broadcast_drafts`` all no-op when
the mask is ``None``, this single patch point skips the whole collective.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import vllm_ascend.patch.platform.patch_pp_token_broadcast as pp_token_broadcast


def _make_batch(
    *,
    num_computed: list[int],
    num_scheduled: list[int],
    prefill_len: list[int],
    is_prefilling: list[bool],
) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        num_scheduled_tokens=np.asarray(num_scheduled, dtype=np.int32),
        prefill_len_np=np.asarray(prefill_len, dtype=np.int32),
        is_prefilling_np=np.asarray(is_prefilling, dtype=np.bool_),
    )


def _patched_mask_applied() -> bool:
    from vllm.v1.worker.gpu import pp_utils

    return getattr(pp_utils.compute_need_sampled_mask, pp_token_broadcast._PATCHED_ATTR, False)


@pytest.mark.skipif(
    not _patched_mask_applied(),
    reason="vLLM without the deferred PP token broadcast",
)
class TestPatchedMask:
    @pytest.fixture(autouse=True)
    def _use_live_mask(self):
        from vllm.v1.worker.gpu import pp_utils

        self.mask = pp_utils.compute_need_sampled_mask

    def test_skip_when_pd_node_final_prefill_chunk(self, monkeypatch):
        monkeypatch.setattr(pp_token_broadcast, "_is_pd_prefill_node", lambda: True)
        batch = _make_batch(
            num_computed=[4096],
            num_scheduled=[4096],
            prefill_len=[8192],
            is_prefilling=[True],
        )
        assert self.mask(batch) is None

    def test_keep_mask_when_decode_request_in_batch(self, monkeypatch):
        # Row 0 is a finishing prefill request; row 1 keeps decoding on this
        # engine and needs the broadcast tokens.
        monkeypatch.setattr(pp_token_broadcast, "_is_pd_prefill_node", lambda: True)
        batch = _make_batch(
            num_computed=[4096, 17],
            num_scheduled=[4096, 1],
            prefill_len=[8192, 16],
            is_prefilling=[True, False],
        )
        mask = self.mask(batch)
        assert mask is not None
        np.testing.assert_array_equal(mask, [True, True])

    def test_keep_mask_on_non_prefill_nodes(self, monkeypatch):
        monkeypatch.setattr(pp_token_broadcast, "_is_pd_prefill_node", lambda: False)
        batch = _make_batch(
            num_computed=[4096],
            num_scheduled=[4096],
            prefill_len=[8192],
            is_prefilling=[True],
        )
        mask = self.mask(batch)
        assert mask is not None
        np.testing.assert_array_equal(mask, [True])

    def test_non_final_chunk_has_no_sample(self, monkeypatch):
        monkeypatch.setattr(pp_token_broadcast, "_is_pd_prefill_node", lambda: True)
        batch = _make_batch(
            num_computed=[0],
            num_scheduled=[4096],
            prefill_len=[8192],
            is_prefilling=[True],
        )
        assert self.mask(batch) is None

    def test_patch_is_idempotent(self):
        from vllm.v1.worker.gpu import pp_utils

        pp_token_broadcast._apply_patch()
        assert getattr(pp_utils.compute_need_sampled_mask, pp_token_broadcast._PATCHED_ATTR, False)
        assert pp_utils.compute_need_sampled_mask.__wrapped__ is not pp_utils.compute_need_sampled_mask


def test_is_pd_prefill_node_reads_kv_role(monkeypatch):
    from vllm import config as config_mod

    def _config(kv_role, is_producer, is_consumer):
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_role=kv_role,
                is_kv_producer=is_producer,
                is_kv_consumer=is_consumer,
            )
        )

    cases = [
        (_config("producer", True, False), True),
        (_config("consumer", False, True), False),
        (_config(None, None, None), False),
        (SimpleNamespace(kv_transfer_config=None), False),
        (None, False),
    ]
    for vllm_config, expected in cases:
        monkeypatch.setattr(
            config_mod,
            "get_current_vllm_config_or_none",
            lambda vllm_config=vllm_config: vllm_config,
        )
        assert pp_token_broadcast._is_pd_prefill_node() is expected
