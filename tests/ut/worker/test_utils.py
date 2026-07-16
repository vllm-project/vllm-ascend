"""Tests for the double-buffer rotation patch on CpuGpuBuffer.

The patch (vllm_ascend.patch.worker.patch_cpu_gpu_buffer) replaces
``copy_to_gpu`` with a version that copies through two rotating pinned
CPU shadow buffers, preventing async-scheduling races where the next
iteration's CPU writes corrupt an in-flight H2D transfer.
"""

import torch
import torch_npu  # noqa: F401  -- required for NPU device

import vllm_ascend.patch.worker  # noqa: F401  -- triggers patch registration
from vllm.v1.utils import CpuGpuBuffer


def _make_buffer(size, dtype, device):
    return CpuGpuBuffer(size, dtype=dtype, device=device, pin_memory=False)


class TestCpuGpuBufferDoubleBuffer:
    def test_init_creates_two_shadow_buffers(self):
        buf = _make_buffer(8, torch.int32, torch.device("npu:0"))
        assert hasattr(buf, "_shadow_buffers")
        assert len(buf._shadow_buffers) == 2
        for shadow in buf._shadow_buffers:
            assert shadow.shape == buf.cpu.shape
            assert shadow.dtype == buf.cpu.dtype
        assert buf._shadow_idx == 0

    def test_cpu_gpu_np_addresses_stay_fixed(self):
        """The patch must not change cpu/gpu/np addresses (graph + ref safety)."""
        buf = _make_buffer(8, torch.int32, torch.device("npu:0"))
        cpu_ptr = buf.cpu.data_ptr()
        gpu_ptr = buf.gpu.data_ptr()
        np_ptr = buf.np.__array_interface__["data"][0]
        buf.copy_to_gpu()
        buf.copy_to_gpu()
        assert buf.cpu.data_ptr() == cpu_ptr
        assert buf.gpu.data_ptr() == gpu_ptr
        assert buf.np.__array_interface__["data"][0] == np_ptr

    def test_shadow_idx_rotates(self):
        buf = _make_buffer(8, torch.int32, torch.device("npu:0"))
        assert buf._shadow_idx == 0
        buf.copy_to_gpu()
        assert buf._shadow_idx == 1
        buf.copy_to_gpu()
        assert buf._shadow_idx == 0

    def test_copy_to_gpu_uses_shadow_not_cpu_directly(self):
        """Verify H2D source is the shadow buffer, not buffer.cpu.

        After copy_to_gpu, mutate buffer.cpu. If GPU got the snapshot via
        the shadow (pre-copy), gpu should reflect the value at copy time,
        not the mutation. (On NPU the copy is async, so synchronize first.)
        """
        buf = _make_buffer(4, torch.int32, torch.device("npu:0"))
        buf.cpu.fill_(42)
        buf.copy_to_gpu()
        # Mutate CPU after the snapshot was taken.
        buf.cpu.fill_(99)
        buf.gpu.npu.stream_synchronize()
        assert buf.gpu.tolist() == [42, 42, 42, 42]

    def test_alternating_shadows_isolate_iterations(self):
        """Simulate async race: iteration N+1 must not overwrite N's H2D.

        With double-buffer, copy_to_gpu(N) uses shadow[0], then copy_to_gpu(N+1)
        uses shadow[1]. Mutating buffer.cpu before N's H2D completes is safe
        because shadow[0] already holds N's snapshot.
        """
        buf = _make_buffer(4, torch.int32, torch.device("npu:0"))
        # Iteration 0
        buf.cpu.fill_(10)
        buf.copy_to_gpu()  # shadow[0] <- 10, H2D shadow[0] -> gpu
        # Iteration 1 (before gpu sync, simulating async overlap)
        buf.cpu.fill_(20)
        buf.copy_to_gpu()  # shadow[1] <- 20, H2D shadow[1] -> gpu
        # GPU now should have iteration 1's data (20).
        buf.gpu.npu.stream_synchronize()
        assert buf.gpu.tolist() == [20, 20, 20, 20]
        # shadow[0] should still hold 10 (iteration 0's snapshot).
        assert buf._shadow_buffers[0].tolist() == [10, 10, 10, 10]
        # shadow[1] should hold 20 (iteration 1's snapshot).
        assert buf._shadow_buffers[1].tolist() == [20, 20, 20, 20]

    def test_partial_copy_to_gpu(self):
        buf = _make_buffer(8, torch.int32, torch.device("npu:0"))
        buf.cpu.fill_(7)
        buf.copy_to_gpu(n=3)
        buf.gpu.npu.stream_synchronize()
        assert buf.gpu[:3].tolist() == [7, 7, 7]
