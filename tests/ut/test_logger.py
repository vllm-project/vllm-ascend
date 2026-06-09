# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_ascend.logger module."""

import logging

from vllm_ascend.logger import (
    AscendColoredFormatter,
    AscendFormatter,
    _infer_module_name,
    init_logger,
)


class TestInferModuleName:
    """Tests for _infer_module_name function."""

    def test_vllm_ascend_module(self):
        """Test module name inference for vllm_ascend modules."""
        assert _infer_module_name("vllm_ascend.attention") == "attention"
        assert _infer_module_name("vllm_ascend.worker") == "worker"
        assert _infer_module_name("vllm_ascend.platform") == "platform"

    def test_vllm_ascend_submodule(self):
        """Test module name inference for vllm_ascend submodules."""
        assert _infer_module_name("vllm_ascend.attention.mla_v1") == "attention"
        assert _infer_module_name("vllm_ascend.worker.worker") == "worker"

    def test_vllm_ascend_init(self):
        """Test module name inference for __init__ modules."""
        assert _infer_module_name("vllm_ascend.attention.__init__") == "attention"
        assert _infer_module_name("vllm_ascend.worker.__init__") == "worker"

    def test_non_vllm_ascend_module(self):
        """Test module name inference for non-vllm_ascend modules."""
        assert _infer_module_name("vllm.attention") == "core"
        assert _infer_module_name("torch.utils") == "core"
        assert _infer_module_name("numpy") == "core"

    def test_empty_module(self):
        """Test module name inference for empty module."""
        assert _infer_module_name("vllm_ascend") == "core"


class TestAscendFormatter:
    """Tests for AscendFormatter class."""

    def test_format_with_module(self):
        """Test formatter with module name."""
        formatter = AscendFormatter(fmt="%(message)s", datefmt="%m-%d %H:%M:%S")
        record = logging.LogRecord(
            name="vllm_ascend.test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "[vllm-ascend] [test_module] - test message" in formatted

    def test_format_without_module(self):
        """Test formatter without module name."""
        formatter = AscendFormatter(fmt="%(message)s", datefmt="%m-%d %H:%M:%S")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "[vllm-ascend] [core] - test message" in formatted


class TestAscendColoredFormatter:
    """Tests for AscendColoredFormatter class."""

    def test_format_with_module(self):
        """Test colored formatter with module name."""
        formatter = AscendColoredFormatter(fmt="%(message)s", datefmt="%m-%d %H:%M:%S")
        record = logging.LogRecord(
            name="vllm_ascend.test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "[vllm-ascend] [test_module] - test message" in formatted


class TestInitLogger:
    """Tests for init_logger function."""

    def test_init_logger_returns_logger(self):
        """Test that init_logger returns a Logger instance."""
        logger = init_logger("vllm_ascend.test")
        assert isinstance(logger, logging.Logger)

    def test_init_logger_sets_module_attribute(self):
        """Test that init_logger sets vllm_ascend_module attribute."""
        logger = init_logger("vllm_ascend.attention")
        assert hasattr(logger, "vllm_ascend_module")
        assert logger.vllm_ascend_module == "attention"

    def test_init_logger_with_custom_module(self):
        """Test init_logger with custom module name."""
        logger = init_logger("vllm_ascend.test", module="custom_module")
        assert logger.vllm_ascend_module == "custom_module"

    def test_init_logger_adds_once_methods(self):
        """Test that init_logger adds debug_once, info_once, warning_once methods."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "debug_once")
        assert hasattr(logger, "info_once")
        assert hasattr(logger, "warning_once")

    def test_init_logger_standard_methods(self):
        """Test that logger has standard logging methods."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

    def test_init_logger_level_attribute(self):
        """Test that logger has level attribute."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "level")

    def test_init_logger_setLevel_method(self):
        """Test that logger has setLevel method."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "setLevel")

    def test_init_logger_addHandler_method(self):
        """Test that logger has addHandler method."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "addHandler")

    def test_init_logger_removeHandler_method(self):
        """Test that logger has removeHandler method."""
        logger = init_logger("vllm_ascend.test")
        assert hasattr(logger, "removeHandler")
