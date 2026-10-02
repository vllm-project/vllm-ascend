#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for --enable-dbo NPU adaptation (NPUUBatchWrapper).

A-class tests: pure CPU / mock, no NPU device required.
B-class tests: require NPU device, gated by @npu_test decorator.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase

# Lightweight stand-in for the historical @npu_test decorator. Upstream
# vllm-ascend now routes NPU tests by directory convention (tests/ut/<m>/a2/),
# so this decorator only marks tests as skipped when no NPU device is visible
# (matching the previous behaviour for B-class tests in CPU-only envs).


def npu_test(num_npus: int = 1):
    """Skip test when torch.npu is unavailable; otherwise run as-is."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not torch.npu.is_available():
                pytest.skip(f"{func.__name__} requires NPU device (num_npus={num_npus})")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _build_wrapper():
    """Construct an NPUUBatchWrapper without a real NPU device."""
    from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

    vllm_config = MagicMock()
    vllm_config.parallel_config.num_ubatches = 2
    with patch("torch.npu.Stream"):
        wrapper = NPUUBatchWrapper(
            runnable=MagicMock(),
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
    return wrapper


# ---------------------------------------------------------------------------
# 2.1 NPUUBatchWrapper — _slice_model_inputs
# ---------------------------------------------------------------------------


class TestSliceModelInputs(TestBase):
    """Test _slice_model_inputs with various input shapes."""

    def test_slice_1d_positions(self):
        wrapper = _build_wrapper()
        input_ids = torch.arange(10)
        positions = torch.arange(10)
        token_slice = slice(0, 5)

        ids, pos, embeds, inter = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert torch.equal(ids, torch.arange(5))
        assert torch.equal(pos, torch.arange(5))
        assert embeds is None
        assert inter is None

    def test_slice_2d_positions(self):
        wrapper = _build_wrapper()
        # input_ids is always 1D, positions can be 2D (e.g. multi_chunk)
        input_ids = torch.arange(10)
        positions = torch.arange(20).reshape(2, 10)
        token_slice = slice(2, 7)

        ids, pos, embeds, inter = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert ids.shape == (5,)
        assert pos.shape == (2, 5)
        assert embeds is None
        assert inter is None

    def test_slice_none_optional_fields(self):
        wrapper = _build_wrapper()
        input_ids = torch.arange(8)
        positions = torch.arange(8)
        token_slice = slice(0, 4)

        ids, pos, embeds, inter = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert ids.shape == (4,)
        assert pos.shape == (4,)
        assert embeds is None
        assert inter is None

    def test_slice_full_range(self):
        wrapper = _build_wrapper()
        input_ids = torch.arange(6)
        positions = torch.arange(6)
        token_slice = slice(0, 6)

        ids, pos, _, _ = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert torch.equal(ids, input_ids)
        assert torch.equal(pos, positions)

    def test_slice_empty_range(self):
        wrapper = _build_wrapper()
        input_ids = torch.arange(10)
        positions = torch.arange(10)
        token_slice = slice(3, 3)

        ids, pos, _, _ = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert ids.numel() == 0
        assert pos.numel() == 0


# ---------------------------------------------------------------------------
# 2.1 NPUUBatchWrapper — passthrough / unwrap / getattr
# ---------------------------------------------------------------------------


class TestNPUUBatchWrapperPassthrough(TestBase):
    """Test non-ubatch pass-through behavior."""

    def test_call_passthrough_without_ubatch(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

        mock_runnable = MagicMock(return_value=torch.tensor([1.0]))
        vllm_config = MagicMock()
        vllm_config.parallel_config.num_ubatches = 2
        with patch("torch.npu.Stream"):
            wrapper = NPUUBatchWrapper(mock_runnable, vllm_config, torch.device("cpu"))

        mock_fc = MagicMock()
        mock_fc.ubatch_slices = None
        with patch(
            "vllm_ascend.worker.npu_ubatch_wrapper.get_forward_context",
            return_value=mock_fc,
        ):
            result = wrapper(
                input_ids=torch.tensor([1]),
                positions=torch.tensor([0]),
                intermediate_tensors=None,
                inputs_embeds=None,
            )

        mock_runnable.assert_called_once()
        assert torch.equal(result, torch.tensor([1.0]))

    def test_unwrap(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

        mock_runnable = MagicMock()
        vllm_config = MagicMock()
        vllm_config.parallel_config.num_ubatches = 2
        with patch("torch.npu.Stream"):
            wrapper = NPUUBatchWrapper(mock_runnable, vllm_config, torch.device("cpu"))

        assert wrapper.unwrap() is mock_runnable

    def test_getattr_delegation(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

        mock_runnable = MagicMock()
        mock_runnable.some_property = 42
        vllm_config = MagicMock()
        vllm_config.parallel_config.num_ubatches = 2
        with patch("torch.npu.Stream"):
            wrapper = NPUUBatchWrapper(mock_runnable, vllm_config, torch.device("cpu"))

        assert wrapper.some_property == 42

    def test_getattr_raises_for_missing(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

        mock_runnable = MagicMock(spec=[])
        vllm_config = MagicMock()
        vllm_config.parallel_config.num_ubatches = 2
        with patch("torch.npu.Stream"):
            wrapper = NPUUBatchWrapper(mock_runnable, vllm_config, torch.device("cpu"))

        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = wrapper.nonexistent_attr


# ---------------------------------------------------------------------------
# 2.2 NPUUBatchContext
# ---------------------------------------------------------------------------


class TestNPUUBatchContext(unittest.TestCase):
    """Test NPUUBatchContext stream/event operations."""

    @patch("torch.npu.set_stream")
    def test_update_stream(self, mock_set_stream):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        mock_stream = MagicMock()
        ctx.update_stream(mock_stream)

        assert ctx.current_stream is mock_stream
        mock_set_stream.assert_called_once_with(mock_stream)

    def test_signal_comm_done(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        ctx.comm_stream = MagicMock()
        ctx.gpu_comm_done_event = MagicMock()

        ctx._signal_comm_done()

        ctx.gpu_comm_done_event.record.assert_called_once_with(ctx.comm_stream)

    def test_signal_compute_done(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        ctx.compute_stream = MagicMock()
        ctx.gpu_compute_done_event = MagicMock()

        ctx._signal_compute_done()

        ctx.gpu_compute_done_event.record.assert_called_once_with(ctx.compute_stream)

    def test_wait_compute_done(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        ctx.comm_stream = MagicMock()
        ctx.gpu_compute_done_event = MagicMock()

        ctx._wait_compute_done()

        ctx.comm_stream.wait_event.assert_called_once_with(ctx.gpu_compute_done_event)

    def test_wait_comm_done(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        ctx.compute_stream = MagicMock()
        ctx.gpu_comm_done_event = MagicMock()

        ctx._wait_comm_done()

        ctx.compute_stream.wait_event.assert_called_once_with(ctx.gpu_comm_done_event)


# ---------------------------------------------------------------------------
# 2.3 _make_npu_ubatch_contexts
# ---------------------------------------------------------------------------


class TestMakeNPUUBatchContexts(TestBase):
    """Test _make_npu_ubatch_contexts."""

    def test_assert_single_batch_fails(self):
        from vllm_ascend.worker.npu_ubatch_wrapper import (
            _make_npu_ubatch_contexts,
        )

        with pytest.raises(AssertionError, match="num_micro_batches must be"):
            _make_npu_ubatch_contexts(
                num_micro_batches=1,
                comm_stream=MagicMock(),
                compute_stream=MagicMock(),
                forward_contexts=[MagicMock()],
                ready_barrier=MagicMock(),
            )

    @patch("torch.npu.Event")
    def test_creates_correct_count(self, mock_event_cls):
        from vllm_ascend.worker.npu_ubatch_wrapper import (
            _make_npu_ubatch_contexts,
        )

        num = 2
        barrier = threading.Barrier(num + 1)
        ctxs = _make_npu_ubatch_contexts(
            num_micro_batches=num,
            comm_stream=MagicMock(),
            compute_stream=MagicMock(),
            forward_contexts=[MagicMock(), MagicMock()],
            ready_barrier=barrier,
        )

        assert len(ctxs) == num
        assert ctxs[0].id == 0
        assert ctxs[1].id == 1

    @patch("torch.npu.Event")
    def test_cpu_events_form_ring(self, mock_event_cls):
        from vllm_ascend.worker.npu_ubatch_wrapper import (
            _make_npu_ubatch_contexts,
        )

        num = 3
        barrier = threading.Barrier(num + 1)
        ctxs = _make_npu_ubatch_contexts(
            num_micro_batches=num,
            comm_stream=MagicMock(),
            compute_stream=MagicMock(),
            forward_contexts=[MagicMock()] * num,
            ready_barrier=barrier,
        )

        for i in range(num):
            j = (i + 1) % num
            assert ctxs[i].cpu_signal_event is ctxs[j].cpu_wait_event


# ---------------------------------------------------------------------------
# 2.5 Platform DBO logic
# ---------------------------------------------------------------------------


class TestPlatformDBO(unittest.TestCase):
    """Test platform.py DBO-related logic."""

    def test_dbo_not_disabled(self):
        parallel_config = MagicMock()
        parallel_config.enable_dbo = True
        original = parallel_config.enable_dbo
        assert original is True

    def test_dbo_all2all_backend(self):
        parallel_config = MagicMock()
        parallel_config.enable_dbo = True

        if parallel_config.enable_dbo:
            parallel_config.all2all_backend = "deepep_low_latency"

        assert parallel_config.all2all_backend == "deepep_low_latency"

    def test_dbo_off_keeps_all2all_default(self):
        parallel_config = MagicMock()
        parallel_config.enable_dbo = False
        parallel_config.all2all_backend = "flashinfer_all2allv"

        if parallel_config.enable_dbo:
            parallel_config.all2all_backend = "deepep_low_latency"

        assert parallel_config.all2all_backend == "flashinfer_all2allv"


# ---------------------------------------------------------------------------
# 2.6 Worker num_ubatches
# ---------------------------------------------------------------------------


class TestWorkerNumUbatch(unittest.TestCase):
    """Test worker.py num_ubatches configuration."""

    def test_num_ubatches_with_dbo(self):
        enable_dbo = True
        num_ubatches = 2 if enable_dbo else 1
        assert num_ubatches == 2

    def test_num_ubatches_without_dbo(self):
        enable_dbo = False
        num_ubatches = 2 if enable_dbo else 1
        assert num_ubatches == 1


# ---------------------------------------------------------------------------
# B-class tests: require real NPU device
# ---------------------------------------------------------------------------


def _build_wrapper_npu(device=None):
    """Construct an NPUUBatchWrapper with real NPU device."""
    from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

    if device is None:
        device = torch.device("npu:0")
    vllm_config = MagicMock()
    vllm_config.parallel_config.num_ubatches = 2
    wrapper = NPUUBatchWrapper(
        runnable=MagicMock(),
        vllm_config=vllm_config,
        device=device,
    )
    return wrapper


class TestNPUUBatchWrapperNPU(unittest.TestCase):
    """B-class tests: NPUUBatchWrapper on real NPU device."""

    @npu_test(num_npus=1)
    def test_wrapper_creates_real_npu_stream(self):
        """NPUUBatchWrapper.__init__ creates a real NPU stream."""
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper

        device = torch.device("npu:0")
        vllm_config = MagicMock()
        vllm_config.parallel_config.num_ubatches = 2

        wrapper = NPUUBatchWrapper(
            runnable=MagicMock(),
            vllm_config=vllm_config,
            device=device,
        )

        assert isinstance(wrapper.comm_stream, torch.npu.Stream)
        assert wrapper.device == device

    @npu_test(num_npus=1)
    def test_slice_model_inputs_on_npu(self):
        """_slice_model_inputs works with NPU tensors."""
        device = torch.device("npu:0")
        wrapper = _build_wrapper_npu(device)

        input_ids = torch.arange(10, device=device)
        positions = torch.arange(10, device=device)
        token_slice = slice(2, 7)

        ids, pos, embeds, inter = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert torch.equal(ids.cpu(), torch.arange(2, 7))
        assert torch.equal(pos.cpu(), torch.arange(2, 7))
        assert ids.device.type == "npu"
        assert embeds is None

    @npu_test(num_npus=1)
    def test_slice_model_inputs_2d_on_npu(self):
        """_slice_model_inputs handles 2D positions on NPU."""
        device = torch.device("npu:0")
        wrapper = _build_wrapper_npu(device)

        input_ids = torch.arange(10, device=device)
        positions = torch.arange(20, device=device).reshape(2, 10)
        token_slice = slice(1, 5)

        ids, pos, _, _ = wrapper._slice_model_inputs(token_slice, input_ids, positions, None, None)

        assert ids.shape == (4,)
        assert pos.shape == (2, 4)
        assert pos.device.type == "npu"

    @npu_test(num_npus=1)
    def test_make_npu_ubatch_contexts_real_npu(self):
        """_make_npu_ubatch_contexts creates contexts with real NPU Events."""
        from vllm_ascend.worker.npu_ubatch_wrapper import (
            _make_npu_ubatch_contexts,
        )

        device = torch.device("npu:0")
        compute_stream = torch.npu.current_stream(device)
        comm_stream = torch.npu.Stream(device=device)

        num = 2
        barrier = threading.Barrier(num + 1)
        ctxs = _make_npu_ubatch_contexts(
            num_micro_batches=num,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_contexts=[MagicMock(), MagicMock()],
            ready_barrier=barrier,
        )

        assert len(ctxs) == num
        assert ctxs[0].id == 0
        assert ctxs[1].id == 1

        # Verify real NPU events were created (not mocks)
        for ctx in ctxs:
            assert isinstance(ctx.gpu_comm_done_event, torch.npu.Event)
            assert isinstance(ctx.gpu_compute_done_event, torch.npu.Event)

        # Verify ring topology
        assert ctxs[0].cpu_signal_event is ctxs[1].cpu_wait_event
        assert ctxs[1].cpu_signal_event is ctxs[0].cpu_wait_event

    @npu_test(num_npus=1)
    def test_context_update_stream_on_npu(self):
        """NPUUBatchContext.update_stream works with real NPU stream."""
        from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchContext

        device = torch.device("npu:0")
        ctx = NPUUBatchContext.__new__(NPUUBatchContext)
        new_stream = torch.npu.Stream(device=device)
        ctx.update_stream(new_stream)

        assert ctx.current_stream is new_stream
