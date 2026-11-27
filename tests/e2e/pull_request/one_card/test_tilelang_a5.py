# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.worker.workspace import WorkspaceManager

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.tilelang.ascend_jit_warmup import AscendTileLangJitKernel

pytestmark = pytest.mark.skipif(not is_950(), reason="TileLang requires Ascend 950")


class CopyKernel(AscendTileLangJitKernel[tuple[int, str]]):
    def get_warmup_keys(self, keys):
        return keys

    def get_prim_func(self, compile_key):
        import tilelang.ascend.language as T

        numel, dtype = compile_key
        block_size = 128

        @T.prim_func
        def copy(source: T.Tensor((numel,), dtype), destination: T.Tensor((numel,), dtype)):  # type: ignore[valid-type]
            with T.Kernel(T.ceildiv(numel, block_size)) as block, T.SimtVF(threads=128):
                for index in T.Parallel(block_size):
                    offset = block * block_size + index
                    if offset < numel:
                        destination[offset] = source[offset]

        return copy


class MatmulKernel(AscendTileLangJitKernel[int]):
    def get_warmup_keys(self, keys):
        return keys

    def get_prim_func(self, rows):
        import tilelang.ascend.language as T

        hidden_size = 64

        @T.prim_func
        def matmul(
            source: T.Tensor((rows, hidden_size), "bfloat16"),  # type: ignore[valid-type]
            weight: T.Tensor((hidden_size, hidden_size), "bfloat16"),  # type: ignore[valid-type]
            output: T.Tensor((rows, hidden_size), "float32"),  # type: ignore[valid-type]
        ):
            with T.Kernel(1):
                source_l1 = T.alloc_l1((rows, hidden_size), "bfloat16")
                weight_l1 = T.alloc_l1((hidden_size, hidden_size), "bfloat16")
                source_l0 = T.alloc_l0a((rows, hidden_size), "bfloat16")
                weight_l0 = T.alloc_l0b((hidden_size, hidden_size), "bfloat16")
                accumulator = T.alloc_l0c((rows, hidden_size), "float32")
                T.copy(source, source_l1)
                T.copy(weight, weight_l1)
                T.copy(source_l1, source_l0)
                T.copy(weight_l1, weight_l0)
                T.gemm(source_l0, weight_l0, accumulator, transpose_B=True, clear_accum=True)
                T.copy(accumulator, output)

        return matmul


def test_matmul_with_persistent_workspace_and_graph():
    torch.npu.set_device(0)
    kernel = MatmulKernel()
    sizes = (15, 32)
    hidden_size = 64
    registry = JitWarmupRegistry(None)
    with registry.activate():
        kernel.register_warmup(sizes)
    registry.warmup()
    workspace = WorkspaceManager(torch.device("npu", 0))
    buffers = {
        rows: workspace.get_persistent(("tilelang_matmul", rows), (rows + 2, hidden_size), torch.float32)
        for rows in sizes
    }
    workspace.lock()
    generator = torch.Generator().manual_seed(0)
    for rows in sizes:
        # Binary fractions give an exact FP32 reference for the Cube calculation.
        source_cpu = torch.randint(-4, 5, (rows, hidden_size), generator=generator).to(torch.bfloat16) / 8
        weight_cpu = torch.randint(-4, 5, (hidden_size, hidden_size), generator=generator).to(torch.bfloat16) / 8
        source, weight = source_cpu.to("npu"), weight_cpu.to("npu")
        storage = buffers[rows]
        storage.fill_(-1234)
        output = storage[1:-1]
        output_address = output.data_ptr()
        executable = kernel.get_kernel(rows)
        torch.npu.synchronize()
        expected = source_cpu.float() @ weight_cpu.float().T
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            executable(source, weight, output)
        for scale in (-1, 2):
            source.copy_(source_cpu * scale)
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected * scale, rtol=0, atol=0)
            assert output.data_ptr() == output_address
            assert workspace.get_persistent(("tilelang_matmul", rows), storage.shape, storage.dtype) is storage
            assert torch.all(storage[0] == -1234) and torch.all(storage[-1] == -1234)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_streams_and_graph_buckets(dtype):
    torch.npu.set_device(0)
    kernel = CopyKernel()
    dtype_name = str(dtype).removeprefix("torch.")
    registry = JitWarmupRegistry(None)
    with registry.activate():
        kernel.register_warmup([(size, dtype_name) for size in (17, 128, 257)])
    registry.warmup()
    work = torch.randn((1024, 1024), dtype=dtype, device="npu")
    scratch = torch.empty_like(work)
    for size in (17, 128, 257):
        source = torch.zeros(size, dtype=dtype, device="npu")
        observed = torch.empty_like(source)
        executable = kernel.get_kernel((size, dtype_name))
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            # Delay the producer so a launch on the default stream sees stale input.
            for _ in range(64):
                torch.mm(work, work, out=scratch)
            source.fill_(7)
            executable(source, observed)
        stream.synchronize()
        torch.testing.assert_close(observed, torch.full_like(source, 7), rtol=0, atol=0)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            executable(source, observed)
        for value in (3, 9):
            source.fill_(value)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(observed, torch.full_like(source, value), rtol=0, atol=0)


def test_custom_op_fake_tensor_and_compile():
    torch.npu.set_device(0)
    kernel = CopyKernel()
    kernel.compile((257, "bfloat16"))

    @torch.library.custom_op("vllm_ascend_test::tilelang_copy", mutates_args=())
    def copy(source: torch.Tensor) -> torch.Tensor:
        destination = torch.empty_like(source)
        kernel.get_kernel((source.numel(), "bfloat16"))(source, destination)
        return destination

    @copy.register_fake
    def fake_copy(source):
        return torch.empty_like(source)

    source = torch.randn(257, dtype=torch.bfloat16, device="npu")
    compiled = torch.compile(copy, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(source), source, rtol=0, atol=0)
