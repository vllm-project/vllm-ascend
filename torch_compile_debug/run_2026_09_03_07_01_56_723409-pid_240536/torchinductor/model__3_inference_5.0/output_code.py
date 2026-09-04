# AOT ID: ['3_inference']
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


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/3d2578789fda9a14131fb2d9fc438aa033ebe74d524c191835c1b4f1a09cf10f/inductor_cache/dj/cdj7elfopzkqihefpf62ci2tm7erqju4rlbnzqyx72iwpkihnetm.py
# Topologically Sorted Source Nodes: [steps, getitem_1, getitem_2, out_of_range, survival_1], Original ATen: [aten.arange, aten.unsqueeze, aten.ge, aten.masked_fill]
# Source node to ATen node mapping:
#   getitem_1 => unsqueeze
#   getitem_2 => unsqueeze_1
#   out_of_range => ge_1
#   steps => iota
#   survival_1 => full_default, where
# Graph fragment:
#   %arg6_1 : Tensor "i32[s36][1]npu:1" = PlaceHolder[target=arg6_1]
#   %cumprod : Tensor "f32[s36, s17][s17, 1]npu:1" = PlaceHolder[target=cumprod]
#   %iota : Tensor "i64[s17][1]npu:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg5_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: npu:1, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, s17][s17, 1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "i32[s36, 1][1, 1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 1), kwargs = {})
#   %ge_1 : Tensor "b8[s36, s17][s17, 1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %full_default : Tensor "f32[][]npu:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:1, pin_memory: False})
#   %where : Tensor "f32[s36, s17][s17, 1]npu:1"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %full_default, %cumprod), kwargs = {})
#   return %where
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_unk_fused_arange_ge_masked_fill_unsqueeze_0 = async_compile.triton('triton_unk_fused_arange_ge_masked_fill_unsqueeze_0', '''
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
    size_hints=[111, 5], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'ks0': 'i64', 'x1_numel': 'i32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=1, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_ge_masked_fill_unsqueeze_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': 'd8092b556b269deb261626774fe820b8b1f3369d43457a4a9822e05d132cd4ed', 'split_axis': [0], 'tiling_axis': [0, 1], 'axis_names': ['x1', 'x0'], 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_ge_masked_fill_unsqueeze_0(in_out_ptr0, in_ptr0, ks0, x1_numel, x0_numel, X1BLOCK : tl.constexpr, X1BLOCK_SUB : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x1_offset = tl.program_id(0) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (x0_numel + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x1 in range(loops_x1):
        x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[:,None]
        x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
        for loop_x0 in range(loops_x0):
            x0 = (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < x0_numel
            tmp0 = tl.load(in_ptr0 + (x1), x1_mask)
            tmp4 = tl.load(in_out_ptr0 + (x0 + ks0*x1), x0_mask & x1_mask)
            tmp1 = tmp0.to(tl.int64)
            tmp2 = x0
            tmp3 = tmp2 >= tmp1
            tmp5 = float("-inf")
            tmp6 = tl.where(tmp3, tmp5, tmp4)
            tl.store(in_out_ptr0 + (x0 + ks0*x1), tmp6, x0_mask & x1_mask)
''', device_str='npu')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/3d2578789fda9a14131fb2d9fc438aa033ebe74d524c191835c1b4f1a09cf10f/inductor_cache/c4/cc4nny3wdohu33bqpmxfjsc7e42n3fupeaorvddu76q7urwb2qxj.py
# Topologically Sorted Source Nodes: [zeros_like], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   zeros_like => full
# Graph fragment:
#   %full : Tensor "b8[s17*s36][1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%sym_numel_default], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:1, pin_memory: False})
#   return %full
# SchedulerNodes: [SchedulerNode(name='op9')]

triton_unk_fused_zeros_like_1 = async_compile.triton('triton_unk_fused_zeros_like_1', '''
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
    size_hints=[555], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=1, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_zeros_like_1', 'mutated_arg_names': [], 'backend_hash': 'd8092b556b269deb261626774fe820b8b1f3369d43457a4a9822e05d132cd4ed', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_zeros_like_1(out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.full([1], False, tl.int1)
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/3d2578789fda9a14131fb2d9fc438aa033ebe74d524c191835c1b4f1a09cf10f/inductor_cache/vo/cvoyejnc37lcmbezzukam4n2lz2trj73zo6ywsr3ljfv7f35evkp.py
# Topologically Sorted Source Nodes: [admitted], Original ATen: [aten.index_fill]
# Source node to ATen node mapping:
#   admitted => scalar_tensor_1
# Graph fragment:
#   %scalar_tensor_1 : Tensor "b8[][]npu:1"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (True,), kwargs = {dtype: torch.bool, layout: torch.strided, device: npu:1, pin_memory: False})
#   return %scalar_tensor_1
# SchedulerNodes: [SchedulerNode(name='op10')]

triton_unk_fused_index_fill_2 = async_compile.triton('triton_unk_fused_index_fill_2', '''
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
    size_hints=[1, 1], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1'}, 'device': NPUDeviceProperties(type='npu', index=1, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_index_fill_2', 'mutated_arg_names': [], 'backend_hash': 'd8092b556b269deb261626774fe820b8b1f3369d43457a4a9822e05d132cd4ed', 'split_axis': [], 'tiling_axis': [], 'axis_names': [], 'low_dims': set(), 'numof_reduction_axis': 0, 'split_axis_dtype': None, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_index_fill_2(out_ptr0):
    tmp0 = tl.full([], True, tl.int1)
    tl.store(out_ptr0 + (tl.arange(0,1)), tmp0, None)
''', device_str='npu')


# kernel path: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/3d2578789fda9a14131fb2d9fc438aa033ebe74d524c191835c1b4f1a09cf10f/inductor_cache/iv/civccqck7tjle2633uujneugindjn7snonzrwh23pizgh3fesllk.py
# Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1, view_2
# Graph fragment:
#   %index_put : Tensor "b8[s17*s36][1]npu:1" = PlaceHolder[target=index_put]
#   %view_2 : Tensor "b8[s36, s17][s17, 1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_put, [%arg3_1, %arg5_1]), kwargs = {})
#   %sum_1 : Tensor "i32[s36][1]npu:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_2, [1]), kwargs = {dtype: torch.int32})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op13')]

triton_unk_fused_sum_view_3 = async_compile.triton('triton_unk_fused_sum_view_3', '''
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

@npu_triton_heuristics.reduction_npu_index(
    size_hints=[111, 5],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i32', 'ks0': 'i64', 'x0_numel': 'i32', 'r1_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=1, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_sum_view_3', 'mutated_arg_names': [], 'backend_hash': 'd8092b556b269deb261626774fe820b8b1f3369d43457a4a9822e05d132cd4ed', 'split_axis': [0], 'tiling_axis': [0, 1], 'axis_names': ['x0', 'r1'], 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.int32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused_sum_view_3(in_ptr0, out_ptr0, ks0, x0_numel, r1_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp3 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], 0, tl.int32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + ks0*x0), r1_mask & x0_mask)
            tmp1 = tmp0.to(tl.int32)
            tmp2 = tl.reshape(tmp1, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp4 = _tmp3 + tmp2
            _tmp3 = tl.where(r1_mask & x0_mask, tmp4, _tmp3)
        tmp3 = tl.sum(_tmp3, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp3, x0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        s92 = arg0_1
        s17 = arg1_1
        s36 = arg3_1
        s96 = arg5_1
        s30 = arg7_1
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.index]
        buf0 = torch.ops.aten.index.Tensor(arg2_1, [arg4_1])
        del arg2_1
        del arg4_1
        buf1 = buf0
        assert_size_stride(buf1, (s36, s17), (s17, 1), 'torch.ops.aten.index.Tensor')
        assert_alignment(buf1, 16, 'torch.ops.aten.index.Tensor')
        del buf0
        # Topologically Sorted Source Nodes: [survival], Original ATen: [aten.cumprod]
        buf2 = torch.ops.aten.cumprod.default(buf1, 1)
        del buf1
        buf3 = buf2
        assert_size_stride(buf3, (s36, s17), (s17, 1), 'torch.ops.aten.cumprod.default')
        assert_alignment(buf3, 16, 'torch.ops.aten.cumprod.default')
        del buf2
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [steps, getitem_1, getitem_2, out_of_range, survival_1], Original ATen: [aten.arange, aten.unsqueeze, aten.ge, aten.masked_fill]
        stream1 = get_raw_stream(1)
        triton_unk_fused_arange_ge_masked_fill_unsqueeze_0.run(buf4, arg6_1, s17, 111, 5, stream=stream1)
        # Topologically Sorted Source Nodes: [flat, topk], Original ATen: [aten.view, aten.topk]
        buf5 = torch.ops.aten.topk.default(reinterpret_tensor(buf4, (s17*s36, ), (1, ), 0), s30)
        buf7 = buf5[1]
        assert_size_stride(buf7, (s30, ), (1, ), 'torch.ops.aten.topk.default')
        assert_alignment(buf7, 16, 'torch.ops.aten.topk.default')
        del buf5
        buf9 = empty_strided((s17*s36, ), (1, ), device='npu', dtype=torch.bool)
        # Topologically Sorted Source Nodes: [zeros_like], Original ATen: [aten.zeros_like]
        stream1 = get_raw_stream(1)
        triton_unk_fused_zeros_like_1.run(buf9, 555, stream=stream1)
        buf10 = empty_strided((), (), device='npu', dtype=torch.bool)
        # Topologically Sorted Source Nodes: [admitted], Original ATen: [aten.index_fill]
        stream1 = get_raw_stream(1)
        triton_unk_fused_index_fill_2.run(buf10, stream=stream1)
        # Topologically Sorted Source Nodes: [zeros_like, admitted], Original ATen: [aten.zeros_like, aten.index_fill]
        buf11 = torch.ops.aten.index_put_.default(buf9, [buf7], reinterpret_tensor(buf10, (s30, ), (0, ), 0))
        del buf10
        del buf7
        buf12 = buf11
        assert_size_stride(buf12, (s17*s36, ), (1, ), 'torch.ops.aten.index_put_.default')
        assert_alignment(buf12, 16, 'torch.ops.aten.index_put_.default')
        del buf9
        buf13 = empty_strided((s36, ), (1, ), device='npu', dtype=torch.int32)
        # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.view, aten.sum]
        stream1 = get_raw_stream(1)
        triton_unk_fused_sum_view_3.run(buf12, buf13, s17, 111, 5, stream=stream1)
        del buf12
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.copy_]
        buf14 = torch.ops.aten.copy_.default(arg6_1, buf13)
        del arg6_1
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 256
    arg1_1 = 5
    arg2_1 = rand_strided((256, 5), (5, 1), device='npu:1', dtype=torch.float32)
    arg3_1 = 111
    arg4_1 = rand_strided((111, ), (1, ), device='npu:1', dtype=torch.int64)
    arg5_1 = 5
    arg6_1 = rand_strided((111, ), (1, ), device='npu:1', dtype=torch.int32)
    arg7_1 = 401
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
