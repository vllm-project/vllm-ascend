# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.timing import (
    KVPD_TIMING_PREFIX,
    kvpd_timing_enabled,
    log_kvpd_timing,
    sum_transfer_bytes,
)


def _config(extra_config):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra_config)
    )


def test_timing_is_opt_in():
    assert not kvpd_timing_enabled(None)
    assert not kvpd_timing_enabled(SimpleNamespace(kv_transfer_config=None))
    assert not kvpd_timing_enabled(_config({}))
    assert not kvpd_timing_enabled(_config({"enable_timing": False}))
    assert kvpd_timing_enabled(_config({"enable_timing": True}))


def test_sum_transfer_bytes_handles_nested_sizes():
    assert sum_transfer_bytes([]) == 0
    assert sum_transfer_bytes([1, 2, 3]) == 6
    assert sum_transfer_bytes([1, [2, 3], (4, 5)]) == 15


def test_log_kvpd_timing_is_silent_when_disabled():
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.timing.logger"
    ) as logger:
        log_kvpd_timing(False, "p_prefill_done", req_id="req-0")
        logger.info.assert_not_called()


def test_log_kvpd_timing_emits_one_json_line_when_enabled():
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.timing.logger"
    ) as logger:
        log_kvpd_timing(True, "d_load_done", req_id="req-0", bytes=4096)

    logger.info.assert_called_once()
    line = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
    assert line.startswith(KVPD_TIMING_PREFIX)
    assert '"event": "d_load_done"' in line
    assert '"req": "req-0"' in line