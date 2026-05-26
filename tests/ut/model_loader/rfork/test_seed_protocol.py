#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Unit tests for seed_protocol.py.

Covers the four behavioural points in this module:
  1. get_local_seed_key() logs error and raises RuntimeError when model_url
     or model_deploy_strategy_name is empty
  2. RForkSeedProtocol.get_seed() returns seed dict on success
  3. RForkSeedProtocol.release_seed() logs "Seed released: ..." on success
  4. RForkSeedProtocol.report_seed() logs setup exception when get_ip fails
"""

import io
import logging
import sys
import unittest
from unittest.mock import MagicMock as _MM
from unittest.mock import patch

import torch

# ---------------------------------------------------------------------------
# Module-level mocks for device-dependent / missing modules.
# These MUST be in place before any vllm-ascend import.
# ---------------------------------------------------------------------------

# torch_npu -----
_tnpu = _MM()

_tnpu.npu.current_device.return_value = 0
_tnpu.__spec__ = _MM()
sys.modules["torch_npu"] = _tnpu
sys.modules["torch_npu.npu"] = _tnpu.npu
sys.modules["torch_npu._inductor"] = _MM()

_tnb = _MM()
_tnb.current_device = _MM(return_value=0)
_tnb.is_available = _MM(return_value=False)
torch.npu = _tnb

# cbor2, gguf (vllm deps) -----
sys.modules["cbor2"] = _MM()
sys.modules["gguf"] = _MM()
sys.modules["gguf.constants"] = _MM()
sys.modules["gguf.quants"] = _MM()

# requests (seed_protocol dep) -----
_req = _MM()
_req.get = _MM()
_req.post = _MM()
sys.modules["requests"] = _req

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LogCapture:
    """Context manager that captures vllm logger output into a StringIO buf."""

    def __init__(self):
        self.buf = io.StringIO()
        self._handler = logging.StreamHandler(self.buf)
        self._handler.setLevel(logging.DEBUG)

    def __enter__(self):
        logging.getLogger("vllm.logger").addHandler(self._handler)
        return self.buf

    def __exit__(self, *args):
        logging.getLogger("vllm.logger").removeHandler(self._handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetLocalSeedKey(unittest.TestCase):
    """Tests for the module-level get_local_seed_key function."""

    def test_empty_model_url_logs_error_and_raises(self):
        """get_local_seed_key logs error and raises RuntimeError when
        model_url is empty, and err_msg contains the actual values."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            get_local_seed_key,
        )

        with _LogCapture() as buf, self.assertRaises(RuntimeError) as ctx:
            get_local_seed_key(
                disaggregation_mode="mode_a",
                node_rank=0,
                tp_rank=0,
                model_url="",
                model_deploy_strategy_name="deploy_x",
            )

        err_msg = str(ctx.exception)
        output = buf.getvalue()

        # err_msg must contain the actual parameter values.
        self.assertIn("model_url=", err_msg)
        self.assertIn("''", err_msg)
        self.assertIn("model_deploy_strategy_name=", err_msg)
        self.assertIn("'deploy_x'", err_msg)

        # logger.error must have been called with the same message.
        self.assertIn("model_url=''", output)
        self.assertIn("model_deploy_strategy_name='deploy_x'", output)

    def test_empty_deploy_strategy_name_logs_error_and_raises(self):
        """get_local_seed_key logs error and raises RuntimeError when
        model_deploy_strategy_name is empty."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            get_local_seed_key,
        )

        with _LogCapture() as buf, self.assertRaises(RuntimeError) as ctx:
            get_local_seed_key(
                disaggregation_mode="mode_a",
                node_rank=0,
                tp_rank=0,
                model_url="http://model:8000",
                model_deploy_strategy_name="",
            )

        err_msg = str(ctx.exception)
        output = buf.getvalue()

        self.assertIn("model_url='http://model:8000'", err_msg)
        self.assertIn("model_deploy_strategy_name=''", err_msg)
        self.assertIn("model_deploy_strategy_name=''", output)

    def test_success_returns_correct_key(self):
        """get_local_seed_key returns correct concatenated seed key."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            get_local_seed_key,
        )

        result = get_local_seed_key(
            disaggregation_mode="mode_a",
            node_rank=1,
            tp_rank=2,
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_x",
        )
        self.assertEqual(result, "http://model:8000$deploy_x$mode_a$1$2")

    def test_draft_worker_appends_draft_suffix(self):
        """get_local_seed_key appends 'draft' suffix when is_draft_worker
        is True."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            get_local_seed_key,
        )

        result = get_local_seed_key(
            disaggregation_mode="mode_a",
            node_rank=1,
            tp_rank=2,
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_x",
            is_draft_worker=True,
        )
        self.assertEqual(result, "http://model:8000$deploy_x$mode_a$1$2$draft")

    def test_custom_separator(self):
        """get_local_seed_key uses the supplied seed_key_separator."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            get_local_seed_key,
        )

        result = get_local_seed_key(
            disaggregation_mode="mode_a",
            node_rank=1,
            tp_rank=2,
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_x",
            seed_key_separator="|",
        )
        self.assertEqual(result, "http://model:8000|deploy_x|mode_a|1|2")


class TestRForkSeedProtocol(unittest.TestCase):
    """Tests for RForkSeedProtocol class methods."""

    def setUp(self):
        # Patch names imported into seed_protocol's module namespace.
        self._requests_get_patch = patch("vllm_ascend.model_loader.rfork.seed_protocol.requests.get")
        self._requests_post_patch = patch("vllm_ascend.model_loader.rfork.seed_protocol.requests.post")
        self._get_ip_patch = patch("vllm_ascend.model_loader.rfork.seed_protocol.get_ip")

        self.mock_requests_get = self._requests_get_patch.start()
        self.mock_requests_post = self._requests_post_patch.start()
        self.mock_get_ip = self._get_ip_patch.start()
        self.mock_get_ip.return_value = "192.168.1.100"

        # Import the target class (inside test code so module-level mocks
        # are already installed).
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            RForkSeedProtocol,
        )

        self.protocol = RForkSeedProtocol(
            disaggregation_mode="mode_a",
            node_rank=0,
            tp_rank=1,
            scheduler_url="http://scheduler:8000",
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_x",
        )

    def tearDown(self):
        self._requests_get_patch.stop()
        self._requests_post_patch.stop()
        self._get_ip_patch.stop()

    # -- get_seed -------------------------------------------------------------

    def test_get_seed_success_returns_seed_dict(self):
        """get_seed returns seed dict on 200 response."""
        mock_response = _MM()
        mock_response.status_code = 200
        mock_response.headers = {
            "SEED_IP": "10.0.0.1",
            "SEED_PORT": "4321",
            "USER_ID": "user_abc",
            "SEED_RANK": "2",
        }
        self.mock_requests_get.return_value = mock_response

        result = self.protocol.get_seed()

        self.assertEqual(
            result,
            {
                "seed_ip": "10.0.0.1",
                "seed_port": "4321",
                "user_id": "user_abc",
                "seed_rank": "2",
            },
        )
        self.mock_requests_get.assert_called_once_with(
            "http://scheduler:8000/get_seed",
            headers={"SEED_KEY": self.protocol.get_local_seed_key()},
            timeout=10.0,
        )

    def test_get_seed_non_200_returns_none(self):
        """get_seed returns None when scheduler returns non-200."""
        mock_response = _MM()
        mock_response.status_code = 500
        self.mock_requests_get.return_value = mock_response

        with _LogCapture() as buf:
            result = self.protocol.get_seed()

        self.assertIsNone(result)
        output = buf.getvalue()
        self.assertIn("get_seed from scheduler RuntimeError", output)

    def test_get_seed_empty_scheduler_url_returns_none(self):
        """get_seed returns None when scheduler_url is empty."""
        from vllm_ascend.model_loader.rfork.seed_protocol import (  # noqa: E402
            RForkSeedProtocol,
        )

        protocol_no_url = RForkSeedProtocol(
            disaggregation_mode="mode_a",
            node_rank=0,
            tp_rank=1,
            scheduler_url="",
            model_url="http://model:8000",
            model_deploy_strategy_name="deploy_x",
        )

        with _LogCapture() as buf:
            result = protocol_no_url.get_seed()

        self.assertIsNone(result)
        # Unsure URL → _ensure_scheduler_url_set raises RuntimeError →
        # caught by the except RuntimeError handler.
        output = buf.getvalue()
        self.assertIn("get_seed from scheduler RuntimeError", output)

    # -- release_seed ---------------------------------------------------------

    def test_release_seed_success_logs_info(self):
        """release_seed returns True on 200 success."""
        seed = {
            "seed_ip": "10.0.0.1",
            "seed_port": 4321,
            "user_id": "user_abc",
            "seed_rank": "2",
        }
        mock_response = _MM()
        mock_response.status_code = 200
        self.mock_requests_post.return_value = mock_response

        result = self.protocol.release_seed(seed)

        self.assertTrue(result)

    def test_release_seed_non_200_returns_false(self):
        """release_seed returns False when planner returns non-200."""
        seed = {
            "seed_ip": "10.0.0.1",
            "seed_port": 4321,
            "user_id": "user_abc",
            "seed_rank": "2",
        }
        mock_response = _MM()
        mock_response.status_code = 500
        self.mock_requests_post.return_value = mock_response

        with _LogCapture() as buf:
            result = self.protocol.release_seed(seed)

        self.assertFalse(result)
        output = buf.getvalue()
        self.assertIn("release_seed to planner", output)

    # -- report_seed ----------------------------------------------------------

    def test_report_seed_setup_exception_logs_error(self):
        """report_seed logs 'report_seed setup Exception: ...' when get_ip
        raises."""
        self.mock_get_ip.side_effect = RuntimeError("no network")

        with _LogCapture() as buf:
            self.protocol.report_seed(port=9999)

        output = buf.getvalue()
        self.assertIn("report_seed setup Exception", output)
        self.assertIn("no network", output)


if __name__ == "__main__":
    unittest.main()
