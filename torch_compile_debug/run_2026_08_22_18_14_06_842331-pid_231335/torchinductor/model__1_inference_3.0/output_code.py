# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/l2/cl2ofykjsenkifay42ogtpwkdhbpoatqzlxll2ai6bujunznbvtt.py
# Topologically Sorted Source Nodes: [b, b_1], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   b => slice_1
#   b_1 => clone
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 2][2, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %slice_1 : Tensor "f32[2, 1][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 1, 0, 1), kwargs = {})
#   %clone : Tensor "f32[2, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_unk_fused_clone_slice_0 = async_compile.triton('triton_unk_fused_clone_slice_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[2], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=32, cc='Ascend950DT_9582', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_slice_0', 'mutated_arg_names': [], 'backend_hash': '7313eea2689854806015bf38c0ac1c1414ead99c80f61a8d4c70dad251ba8fa1', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_slice_0(in_ptr0, out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (2*x0), x0_mask)
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/4z/c4zz72q7mvvns23pzg4j7cysiunjglfvoeoqjd7mage5rh6e5yhn.py
# Topologically Sorted Source Nodes: [a, a_1], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   a => slice_2
#   a_1 => clone_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 2][2, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %slice_2 : Tensor "f32[2, 1][2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg0_1, 1, 1, 9223372036854775807), kwargs = {})
#   %clone_1 : Tensor "f32[2, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_unk_fused_clone_slice_1 = async_compile.triton('triton_unk_fused_clone_slice_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[2], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=32, cc='Ascend950DT_9582', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_slice_1', 'mutated_arg_names': [], 'backend_hash': '7313eea2689854806015bf38c0ac1c1414ead99c80f61a8d4c70dad251ba8fa1', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_slice_1(in_ptr0, out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (1 + 2*x0), x0_mask)
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/3u/c3upehgkpjzdfckfo62ukyveomhrldqnkdsjla4xyud4e4ctnbij.py
# Topologically Sorted Source Nodes: [core_attn_out], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   core_attn_out => full_default
# Graph fragment:
#   %full_default : Tensor "f32[2, 1, 2][2, 2, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 1, 2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_unk_fused_zeros_2 = async_compile.triton('triton_unk_fused_zeros_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[4], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=32, cc='Ascend950DT_9582', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_zeros_2', 'mutated_arg_names': [], 'backend_hash': '7313eea2689854806015bf38c0ac1c1414ead99c80f61a8d4c70dad251ba8fa1', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_zeros_2(out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = 0.0
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        buf0 = empty_strided((2, 1), (1, 2), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [b, b_1], Original ATen: [aten.slice, aten.clone]
        stream0 = get_raw_stream(0)
        triton_unk_fused_clone_slice_0.run(arg0_1, buf0, 2, stream=stream0)
        buf1 = empty_strided((2, 1), (1, 2), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [a, a_1], Original ATen: [aten.slice, aten.clone]
        stream0 = get_raw_stream(0)
        triton_unk_fused_clone_slice_1.run(arg0_1, buf1, 2, stream=stream0)
        buf2 = empty_strided((2, 1, 2), (2, 2, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [core_attn_out], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_unk_fused_zeros_2.run(buf2, 4, stream=stream0)
        # Topologically Sorted Source Nodes: [b, b_1, a, a_1, core_attn_out, qwen_gdn_attention_core], Original ATen: [aten.slice, aten.clone, aten.zeros, vllm.qwen_gdn_attention_core]
        torch.ops.vllm.qwen_gdn_attention_core.default(arg0_1, buf0, buf1, buf2, 'layers.0.linear_attn')
        del arg0_1
        return (reinterpret_tensor(buf2, (2, 2), (2, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 2), (2, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
