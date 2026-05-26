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

"""Unit tests for transfer_backend.py.

Covers the five logging-centric failure paths in
RForkTransferBackend and get_remote_instance_transfer_engine_info:
  1. init_transfer_engine -- ImportError importing TransferEngine.
  2. init_transfer_engine -- TransferEngine.initialize() returns error.
  3. recv_from_source -- batch_transfer_sync_read returns error.
  4. get_remote_instance_transfer_engine_info -- HTTP status != 200.
  5. get_remote_instance_transfer_engine_info -- request raises exception.
"""

# ---------------------------------------------------------------------------
# Module-level mocks -- MUST be in place before any vllm-ascend import.
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock as _MM

import torch

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

sys.modules["cbor2"] = _MM()
sys.modules["gguf"] = _MM()
sys.modules["gguf.constants"] = _MM()
sys.modules["gguf.quants"] = _MM()

_yr = _MM()
_yr.datasystem = _MM()
sys.modules["yr"] = _yr
sys.modules["yr.datasystem"] = _yr.datasystem

_req = _MM()
_req.get = _MM()
_req.post = _MM()
sys.modules["requests"] = _req

# vllm (not installed in this test environment) -----
import logging as _logging  # noqa: E402

_vllm_logger = _MM()
_vllm_logger.logger = _logging.getLogger("vllm.logger")
sys.modules["vllm"] = _MM()
sys.modules["vllm.logger"] = _vllm_logger
sys.modules["vllm.utils"] = _MM()
_vllm_net = _MM()
_vllm_net.get_ip = _MM(return_value="127.0.0.1")
_vllm_net.get_open_port = _MM(return_value=12345)
_vllm_net.join_host_port = _MM(return_value="127.0.0.1:12345")
sys.modules["vllm.utils.network_utils"] = _vllm_net

# ---------------------------------------------------------------------------
# Standard-library imports (safe after the module-level mocks above)
# ---------------------------------------------------------------------------
import io  # noqa: E402
import logging  # noqa: E402
from unittest import TestCase  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402


class TestRForkTransferBackendLogging(TestCase):
    """Verify logger.error is called at each expected failure path."""

    def setUp(self):
        self.log_capture = io.StringIO()
        self._handler = logging.StreamHandler(self.log_capture)
        self._handler.setLevel(logging.ERROR)
        logging.getLogger("vllm.logger").addHandler(self._handler)

    def tearDown(self):
        logging.getLogger("vllm.logger").removeHandler(self._handler)

    def _logged(self) -> str:
        return self.log_capture.getvalue()

    # ------------------------------------------------------------------
    # Test 1: ImportError in init_transfer_engine
    # ------------------------------------------------------------------
    def test_init_transfer_engine_import_error_logs(self):
        from vllm_ascend.model_loader.rfork.transfer_backend import (  # noqa: E402
            RForkTransferBackend,
        )

        # Remove yr / yr.datasystem from sys.modules so the import
        # triggers builtins.__import__ instead of using the cached mock.
        saved = {}
        for key in list(sys.modules.keys()):
            if key.startswith("yr"):
                saved[key] = sys.modules.pop(key)

        try:
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                mock_self = MagicMock()
                with self.assertRaises(ImportError):
                    RForkTransferBackend.init_transfer_engine(mock_self)

            self.assertIn(
                "Failed to import TransferEngine from yr.datasystem",
                self._logged(),
            )
        finally:
            sys.modules.update(saved)

    # ------------------------------------------------------------------
    # Test 2: TransferEngine.initialize() returns error
    # ------------------------------------------------------------------
    def test_init_transfer_engine_initialize_error_logs(self):
        from vllm_ascend.model_loader.rfork.transfer_backend import (  # noqa: E402
            RForkTransferBackend,
        )

        mock_engine = MagicMock()
        mock_engine.initialize.return_value.is_error.return_value = True
        mock_engine.initialize.return_value.to_string.return_value = "mock init fail"

        with patch("yr.datasystem.TransferEngine", return_value=mock_engine):
            mock_self = MagicMock()
            with self.assertRaises(RuntimeError):
                RForkTransferBackend.init_transfer_engine(mock_self)

        self.assertIn(
            "TransferEngine initialization failed",
            self._logged(),
        )

    # ------------------------------------------------------------------
    # Test 3: recv_from_source -- batch_transfer_sync_read returns error
    # ------------------------------------------------------------------
    def test_recv_from_source_transfer_error_logs_seed_info(self):
        from vllm_ascend.model_loader.rfork.transfer_backend import (  # noqa: E402
            RForkTransferBackend,
        )

        # Patch the helper so it returns valid session + weight info.
        with patch(
            "vllm_ascend.model_loader.rfork.transfer_backend.get_remote_instance_transfer_engine_info"
        ) as mock_get_info:
            mock_get_info.return_value = (
                "session_xyz",
                {"param_a": (1000, 64, 2)},
            )

            # Construct a real RForkTransferBackend instance but skip
            # init_transfer_engine so we can control the engine manually.
            with patch.object(RForkTransferBackend, "init_transfer_engine"):
                backend = RForkTransferBackend()
            backend.rfork_transfer_engine = MagicMock()
            backend.rfork_transfer_engine.batch_transfer_sync_read.return_value.is_error.return_value = True
            backend.rfork_transfer_engine.batch_transfer_sync_read.return_value.to_string.return_value = (
                "mock batch fail"
            )

            # Mock model with parameters matching seed_weight_info
            param_tensor = MagicMock()
            param_tensor.numel.return_value = 64
            param_tensor.element_size.return_value = 2
            param_tensor.data_ptr.return_value = 2000

            mock_model = MagicMock()
            mock_model.named_parameters.return_value = [
                ("param_a", param_tensor),
            ]

            result = backend.recv_from_source(
                mock_model,
                "10.0.0.1",
                "54321",
                "local_key",
            )

        self.assertFalse(result)
        logged = self._logged()
        self.assertIn("seed_ip=10.0.0.1", logged)
        self.assertIn("seed_port=54321", logged)

    # ------------------------------------------------------------------
    # Test 4: get_remote_instance_transfer_engine_info -- HTTP != 200
    # ------------------------------------------------------------------
    def test_http_non_200_logs_seed_url(self):
        from vllm_ascend.model_loader.rfork.transfer_backend import (  # noqa: E402
            get_remote_instance_transfer_engine_info,
        )

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            sid, info = get_remote_instance_transfer_engine_info("http://10.0.0.2:9999", "some_key")

        self.assertIsNone(sid)
        self.assertIsNone(info)
        logged = self._logged()
        self.assertIn("http://10.0.0.2:9999", logged)
        self.assertIn("404", logged)

    # ------------------------------------------------------------------
    # Test 5: get_remote_instance_transfer_engine_info -- exception
    # ------------------------------------------------------------------
    def test_http_exception_logs_seed_url(self):
        from vllm_ascend.model_loader.rfork.transfer_backend import (  # noqa: E402
            get_remote_instance_transfer_engine_info,
        )

        with patch("requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("connection refused")

            sid, info = get_remote_instance_transfer_engine_info("http://10.0.0.3:8888", "some_key")

        self.assertIsNone(sid)
        self.assertIsNone(info)
        logged = self._logged()
        self.assertIn("http://10.0.0.3:8888", logged)
