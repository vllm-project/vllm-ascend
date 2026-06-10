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
