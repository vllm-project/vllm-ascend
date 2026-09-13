# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setup for the kv_offload CPU unit tests."""

from unittest.mock import patch

import pytest

_MOONCAKE_CONNECTOR = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Re-arm the mooncake connector's module-level group mocks.

    ``tests/ut/test_utils.py`` calls ``mock.patch.stopall()`` mid-batch, which
    deactivates the ``patch(...).start()`` calls that ``test_mooncake_connector``
    installs at import. Its thread tests then call the real
    ``get_pp_group()``/``get_tp_group()`` and fail with "pipeline model parallel
    group is not initialized". Re-applying the same mocks immediately before each
    of that module's tests keeps the shared process order-independent.
    """
    module = getattr(item, "module", None)
    if module is None or not module.__name__.endswith("test_mooncake_connector"):
        return
    patch(f"{_MOONCAKE_CONNECTOR}.get_pp_group", return_value=module._mock_pp_group).start()
    patch(f"{_MOONCAKE_CONNECTOR}.get_tp_group", return_value=module._mock_tp_group).start()
    patch(f"{_MOONCAKE_CONNECTOR}.get_tensor_model_parallel_world_size", return_value=4).start()
    patch(f"{_MOONCAKE_CONNECTOR}.get_tensor_model_parallel_rank", return_value=0).start()
    patch(f"{_MOONCAKE_CONNECTOR}.get_pcp_group", return_value=module._mock_pcp_group).start()
    patch("vllm.distributed.parallel_state._DCP", module._mock_dcp_group).start()
