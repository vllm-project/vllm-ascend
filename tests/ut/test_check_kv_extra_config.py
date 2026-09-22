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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.utils import check_kv_extra_config


def _make_vllm_config(
    *,
    kv_role: str,
    prefill: dict | None = None,
    decode: dict | None = None,
    dp_size: int = 1,
    tp_size: int = 1,
):
    """Minimal stand-in for VllmConfig that ``check_kv_extra_config`` reads.

    ``check_kv_extra_config`` only touches ``parallel_config`` (dp/tp size) and
    ``kv_transfer_config`` (``is_kv_producer``/``is_kv_consumer`` and
    ``get_from_extra_config``), so a mock is sufficient and NPU-free.
    """
    kvtc = MagicMock()
    kvtc.is_kv_producer = kv_role in ("kv_producer", "kv_both")
    kvtc.is_kv_consumer = kv_role in ("kv_consumer", "kv_both")

    extra: dict = {}
    if prefill is not None:
        extra["prefill"] = prefill
    if decode is not None:
        extra["decode"] = decode
    kvtc.get_from_extra_config.side_effect = lambda key, default: extra.get(key, default)

    vllm_config = MagicMock()
    vllm_config.parallel_config = SimpleNamespace(
        data_parallel_size=dp_size, tensor_parallel_size=tp_size
    )
    vllm_config.kv_transfer_config = kvtc
    return vllm_config


class TestCheckKVExtraConfig:
    def test_producer_prefill_dp_mismatch_raises(self):
        """Producer with prefill.dp_size != launch dp must be caught."""
        cfg = _make_vllm_config(
            kv_role="kv_producer",
            prefill={"dp_size": 2, "tp_size": 2},
            dp_size=1,
        )
        with pytest.raises(ValueError, match="prefill.*conflicting data parallel size"):
            check_kv_extra_config(cfg)

    def test_producer_prefill_dp_match_passes(self):
        cfg = _make_vllm_config(
            kv_role="kv_producer",
            prefill={"dp_size": 2, "tp_size": 2},
            dp_size=2,
        )
        check_kv_extra_config(cfg)  # must not raise

    def test_producer_prefill_tp_mismatch_raises(self):
        cfg = _make_vllm_config(
            kv_role="kv_producer",
            prefill={"dp_size": 1, "tp_size": 2},
            dp_size=1,
            tp_size=1,
        )
        with pytest.raises(ValueError, match="conflicting tensor parallel size"):
            check_kv_extra_config(cfg)

    def test_consumer_decode_dp_mismatch_raises(self):
        """Consumer with decode.dp_size != launch dp must be caught."""
        cfg = _make_vllm_config(
            kv_role="kv_consumer",
            decode={"dp_size": 2, "tp_size": 2},
            dp_size=1,
        )
        with pytest.raises(ValueError, match="decode.*conflicting data parallel size"):
            check_kv_extra_config(cfg)

    def test_skips_when_prefill_decode_absent(self):
        """When the prefill/decode sub-configs are absent (e.g. stripped from
        the drafter's draft config), the check must skip. This is the contract
        the drafter fix in AscendSpecDecodeBaseProposer relies on: a dp_size
        that would otherwise mismatch is no longer compared."""
        cfg = _make_vllm_config(
            kv_role="kv_producer",
            prefill=None,
            decode=None,
            dp_size=1,  # would mismatch if a prefill dp_size were present
        )
        check_kv_extra_config(cfg)  # must not raise
