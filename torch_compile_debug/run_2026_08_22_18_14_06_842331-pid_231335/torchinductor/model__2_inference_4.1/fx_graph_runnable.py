
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TRITON_CACHE_AUTOTUNING'] = '1'
os.environ['TORCHINDUCTOR_COMPREHENSIVE_PADDING'] = '0'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_root/triton/0'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.post_grad_fusion_options = {'fav3_partition': {}}
torch._inductor.config.fallback_random = True
torch._inductor.config.deterministic = False
torch._inductor.config.compile_threads = 32
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.autotune_at_compile_time = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cpu
# torch cuda version: None
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 1)
        slice_2 = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807)
        clone = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        clone_1 = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        full_default = torch.ops.aten.full.default([2, 1, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.vllm.qwen_gdn_attention_core.default, qkv_or_qkvz = arg0_1, b_or_ba = clone, layer_name = 'layers.0.linear_attn', use_aiter = False, _a_or_z_out_base_index = 0, _core_attn_out_base_index = 1, _all_bases = [clone_1, full_default]);  arg0_1 = clone = clone_1 = full_default = None
        getitem_2 = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        view_8 = torch.ops.aten.view.default(getitem_2, [-1, 2]);  getitem_2 = None
        return (view_8,)
        
def load_args(reader):
    buf0 = reader.storage(None, 16)
    reader.tensor(buf0, (2, 2), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)