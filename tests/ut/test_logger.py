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

import logging

from tests.ut.base import TestBase


class TestLoggerModule(TestBase):
    def test_is_ascend_module_with_ascend_path(self):
        from vllm_ascend.logger import _is_ascend_module

        self.assertTrue(_is_ascend_module("/path/to/vllm_ascend/platform.py"))
        self.assertTrue(_is_ascend_module("\\path\\to\\vllm_ascend\\compilation\\acl_graph.py"))

    def test_is_ascend_module_with_vllm_path(self):
        from vllm_ascend.logger import _is_ascend_module

        self.assertFalse(_is_ascend_module("/path/to/vllm/model.py"))
        self.assertFalse(_is_ascend_module(""))

    def test_infer_module_name_root_file(self):
        from vllm_ascend.logger import _infer_module_name

        self.assertEqual(
            _infer_module_name("/vllm_ascend/platform.py"),
            "platform",
        )
        self.assertEqual(
            _infer_module_name("/vllm_ascend/utils.py"),
            "utils",
        )

    def test_infer_module_name_nested_file(self):
        from vllm_ascend.logger import _infer_module_name

        self.assertEqual(
            _infer_module_name("/vllm_ascend/compilation/acl_graph.py"),
            "compilation",
        )
        self.assertEqual(
            _infer_module_name("/vllm_ascend/worker/worker.py"),
            "worker",
        )

    def test_infer_module_name_edge_cases(self):
        from vllm_ascend.logger import _infer_module_name

        self.assertEqual(_infer_module_name(""), "core")
        self.assertEqual(
            _infer_module_name("/vllm/model.py"),
            "core",
        )

    def test_ascend_formatter_adds_prefix_root_file(self):
        from vllm_ascend.logger import _DATE_FORMAT, _FORMAT, AscendFormatter

        fmt = AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="vllm.logger",
            level=logging.INFO,
            pathname="/vllm_ascend/ascend_config.py",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        self.assertIn("[vllm-ascend]", result)
        self.assertNotIn("[vllm-ascend] [ascend_config]", result)
        self.assertIn("test message", result)

    def test_ascend_formatter_adds_prefix_nested_file(self):
        from vllm_ascend.logger import _DATE_FORMAT, _FORMAT, AscendFormatter

        fmt = AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="vllm.logger",
            level=logging.INFO,
            pathname="/vllm_ascend/compilation/acl_graph.py",
            lineno=42,
            msg="test message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        self.assertIn("[vllm-ascend] [compilation]", result)
        self.assertIn("test message", result)

    def test_ascend_formatter_pass_through_vllm_logs(self):
        from vllm_ascend.logger import _DATE_FORMAT, _FORMAT, AscendFormatter

        fmt = AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="vllm.logger",
            level=logging.INFO,
            pathname="/vllm/model.py",
            lineno=42,
            msg="vllm message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        self.assertNotIn("[vllm-ascend]", result)
        self.assertIn("vllm message", result)

    def test_ascend_colored_formatter_adds_prefix(self):
        from vllm_ascend.logger import _DATE_FORMAT, _FORMAT, AscendColoredFormatter

        fmt = AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="vllm.logger",
            level=logging.INFO,
            pathname="/vllm_ascend/compilation/acl_graph.py",
            lineno=42,
            msg="colored test",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        self.assertIn("[vllm-ascend] [compilation]", result)
        self.assertIn("colored test", result)

    def test_log_dir_constant(self):
        from vllm_ascend.logger import _LOG_DIR

        self.assertIn("ascend", _LOG_DIR)
        self.assertIn("vllm_ascend", _LOG_DIR)

    def test_setup_file_logging_creates_handler(self):
        import os
        import tempfile

        import regex as re

        import vllm_ascend.logger as logger_module
        from vllm_ascend.logger import RotatingAscendFileHandler, _setup_file_logging

        logger_module._file_logging_configured = False
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_dir = os.path.join(tmpdir, "ascend", "log", "vllm_ascend")
                vllm_logger = logging.getLogger("vllm")
                if not vllm_logger.handlers:
                    vllm_logger.addHandler(logging.StreamHandler())
                expected_level = vllm_logger.handlers[0].level
                handler_count_before = len(vllm_logger.handlers)

                _setup_file_logging(log_dir)

                handler_count_after = len(vllm_logger.handlers)
                self.assertEqual(handler_count_after, handler_count_before + 1)

                new_handler = vllm_logger.handlers[-1]
                self.assertIsInstance(new_handler, RotatingAscendFileHandler)
                self.assertEqual(new_handler.level, expected_level)
                self.assertTrue(os.path.exists(log_dir))

                base = os.path.basename(new_handler.baseFilename)  # type: ignore[attr-defined]
                pattern = r"^vllm_ascend_\d{8}_\d{6}_\d+\.log$"
                self.assertTrue(re.match(pattern, base), f"Filename '{base}' does not match pattern")

                vllm_logger.removeHandler(new_handler)
        finally:
            logger_module._file_logging_configured = False

    def test_rotating_handler_rotates_on_size(self):
        import os
        import tempfile

        from vllm_ascend.logger import RotatingAscendFileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = RotatingAscendFileHandler(tmpdir, max_bytes=100)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger = logging.getLogger("test_rotate")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

            for i in range(200):
                logger.info("line %04d padding to fill size", i)

            handler.close()
            logger.removeHandler(handler)

            files = sorted(os.listdir(tmpdir))
            self.assertGreaterEqual(len(files), 2)
            self.assertTrue(files[0].endswith(".log"))
            self.assertNotIn("_002", files[0])
            self.assertIn("_002", files[1])
