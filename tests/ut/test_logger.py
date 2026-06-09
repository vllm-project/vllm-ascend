# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Tests for vllm_ascend.logger module."""

import logging
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.logger import (
    AscendColoredFormatter,
    AscendFormatter,
    _infer_module_name,
    _use_color,
    configure_ascend_logging,
    setup_module_logger,
)


class TestLoggerUtils(TestBase):
    """Test logger utility functions."""

    def test_infer_module_name_empty(self):
        """Test _infer_module_name with empty string."""
        assert _infer_module_name("") == "core"

    def test_infer_module_name_none(self):
        """Test _infer_module_name with None."""
        assert _infer_module_name(None) == "core"

    def test_infer_module_name_vllm_ascend_attention(self):
        """Test _infer_module_name with vllm_ascend.attention."""
        assert _infer_module_name("vllm_ascend.attention.mla_v1") == "attention"

    def test_infer_module_name_vllm_ascend_worker(self):
        """Test _infer_module_name with vllm_ascend.worker."""
        assert _infer_module_name("vllm_ascend.worker") == "worker"

    def test_infer_module_name_vllm_ascend_ops(self):
        """Test _infer_module_name with vllm_ascend.ops."""
        assert _infer_module_name("vllm_ascend.ops.linear") == "ops"

    def test_infer_module_name_vllm_ascend_distributed(self):
        """Test _infer_module_name with vllm_ascend.distributed."""
        assert _infer_module_name("vllm_ascend.distributed.parallel_state") == "distributed"

    def test_infer_module_name_vllm_ascend_compilation(self):
        """Test _infer_module_name with vllm_ascend.compilation."""
        assert _infer_module_name("vllm_ascend.compilation.compiler_interface") == "compilation"

    def test_infer_module_name_vllm_ascend_quantization(self):
        """Test _infer_module_name with vllm_ascend.quantization."""
        assert _infer_module_name("vllm_ascend.quantization.utils") == "quantization"

    def test_infer_module_name_vllm_ascend_model_loader(self):
        """Test _infer_module_name with vllm_ascend.model_loader."""
        assert _infer_module_name("vllm_ascend.model_loader.loader") == "model_loader"

    def test_infer_module_name_vllm_ascend_eplb(self):
        """Test _infer_module_name with vllm_ascend.eplb."""
        assert _infer_module_name("vllm_ascend.eplb.worker") == "eplb"

    def test_infer_module_name_vllm_ascend_core(self):
        """Test _infer_module_name with vllm_ascend.core."""
        assert _infer_module_name("vllm_ascend.core.scheduler") == "core"

    def test_infer_module_name_vllm_ascend_other(self):
        """Test _infer_module_name with other vllm_ascend module."""
        assert _infer_module_name("vllm_ascend.some_module") == "some_module"

    def test_infer_module_name_vllm(self):
        """Test _infer_module_name with vllm module."""
        assert _infer_module_name("vllm.attention") == "core"

    def test_infer_module_name_torch(self):
        """Test _infer_module_name with torch module."""
        assert _infer_module_name("torch.nn") == "core"

    def test_use_color_no_color_env(self):
        """Test _use_color when NO_COLOR is set."""
        with patch("vllm_ascend.logger.envs") as mock_envs:
            mock_envs.NO_COLOR = True
            mock_envs.VLLM_LOGGING_COLOR = "0"
            mock_envs.VLLM_LOGGING_STREAM = "ext://sys.stderr"
            assert _use_color() is False

    def test_use_color_force_color(self):
        """Test _use_color when VLLM_LOGGING_COLOR is '1'."""
        with patch("vllm_ascend.logger.envs") as mock_envs:
            mock_envs.NO_COLOR = False
            mock_envs.VLLM_LOGGING_COLOR = "1"
            mock_envs.VLLM_LOGGING_STREAM = "ext://sys.stderr"
            assert _use_color() is True


class TestAscendFormatter(TestBase):
    """Test AscendFormatter class."""

    def test_format_basic(self):
        """Test basic formatting."""
        formatter = AscendFormatter()
        record = logging.LogRecord(
            name="vllm_ascend.attention",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "[vllm-ascend]" in result
        assert "[attention]" in result
        assert "Test message" in result

    def test_format_with_args(self):
        """Test formatting with message args."""
        formatter = AscendFormatter()
        record = logging.LogRecord(
            name="vllm_ascend.worker",
            level=logging.INFO,
            pathname="test.py",
            lineno=456,
            msg="Processing %d items",
            args=(10,),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "[vllm-ascend]" in result
        assert "[worker]" in result
        assert "Processing 10 items" in result


class TestAscendColoredFormatter(TestBase):
    """Test AscendColoredFormatter class."""

    def test_format_basic(self):
        """Test basic colored formatting."""
        formatter = AscendColoredFormatter()
        record = logging.LogRecord(
            name="vllm_ascend.ops",
            level=logging.INFO,
            pathname="test.py",
            lineno=789,
            msg="Test colored message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "[vllm-ascend]" in result
        assert "[ops]" in result
        assert "Test colored message" in result


class TestConfigureAscendLogging(TestBase):
    """Test configure_ascend_logging function."""

    def test_configure_ascend_logging(self):
        """Test that configure_ascend_logging runs without error."""
        # Clear any existing handlers
        ascend_logger = logging.getLogger("vllm_ascend")
        ascend_logger.handlers.clear()

        # Call configure
        configure_ascend_logging()

        # Verify handler was added
        assert len(ascend_logger.handlers) > 0

    def test_configure_ascend_logging_idempotent(self):
        """Test that configure_ascend_logging is idempotent."""
        ascend_logger = logging.getLogger("vllm_ascend")
        ascend_logger.handlers.clear()

        configure_ascend_logging()
        first_handler_count = len(ascend_logger.handlers)

        configure_ascend_logging()
        second_handler_count = len(ascend_logger.handlers)

        # Should not add duplicate handlers
        assert first_handler_count == second_handler_count


class TestSetupModuleLogger(TestBase):
    """Test setup_module_logger function."""

    def test_setup_module_logger(self):
        """Test that setup_module_logger returns a logger."""
        logger = setup_module_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_setup_module_logger_with_name(self):
        """Test setup_module_logger with specific name."""
        logger = setup_module_logger("vllm_ascend.test")
        assert logger.name == "vllm_ascend.test"
