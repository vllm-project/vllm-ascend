import torch.distributed as dist
import torch.multiprocessing as mp
import time
import torch

from typing import Callable, Optional, TypeVar, Union


from vllm.engine.arg_utils import EngineArgs


from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker

T = TypeVar("T", bound=Worker)

# from torch import nn
from vllm.distributed.parallel_state import (
                                             init_model_parallel_group
                                             )


from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
    _get_default_group,
    _update_default_pg,
)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from vllm.config import  VllmConfig


from vllm_ascend.quantization.quant_config import AscendLinearMethod
from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm.distributed import parallel_state as ps
from vllm.distributed.parallel_state import get_new_default_group,get_new_cpu_default_group,get_world_group


from contextlib import contextmanager
from vllm_ascend.distributed.afd_communicators import send_object,recv_object,FFNNeedForwardData

import os
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"

@contextmanager
def default_process_group_switcher(default_group, new_default_group):
    try:
        _update_default_pg(new_default_group)
        yield
    finally:
        _update_default_pg(default_group)

class DefaultProcessGroupSwitcher:
    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)

# @torch._dynamo.disable(recursive=False)
# @torch._dynamo.skip
@torch.compiler.disable(recursive=False)
def set_process_group(new_default_group):
    _update_default_pg(new_default_group)  # 切换进程组

# @torch._dynamo.disable(recursive=False)
# @torch._dynamo.skip
@torch.compiler.disable(recursive=False)
def reset_process_group(default_group):
    _update_default_pg(default_group)  # 恢复原进程组

def creat_hccl_process_group(rank, world_size):
    import torch
    import torch_npu
    import os
    torch.npu.set_device(rank)
    new_default_group = init_process_group(
        init_method='tcp://127.0.0.1:29500',
        backend='hccl', 
        rank=rank, 
        world_size=world_size, 
        group_name="new_hccl"
    )
    cpu_new_default_group = init_process_group(
        init_method='tcp://127.0.0.1:29500',
        backend='gloo', 
        rank=rank, 
        world_size=world_size, 
        group_name="new_gloo"
    )
    return new_default_group,cpu_new_default_group
    
def create_ffn_process_group(rank, world_size,attn_size, ffn_size):
    print(f"进程 {rank} 启动，参数: world_size={world_size}, "
          f"attn_size={attn_size}, ffn_size={ffn_size}")
    if rank == 2: time.sleep(1); print('======================================================================')
    import torch
    
    torch.npu.set_device(rank)
    #TODO:remove hard code
    init_method = 'tcp://127.0.0.1:29505'
    ffn_default_group = dist.init_process_group(
            init_method=init_method,
            backend='hccl', 
            rank=rank % attn_size, 
            world_size=ffn_size
        )
    return ffn_default_group

def run_ffn(rank, world_size,attn_size, ffn_size):
    config = create_config()
    attn_ranks = list(config.additional_config.get("attn_ranks"))
    ffn_ranks = list(config.additional_config.get("ffn_ranks"))
    role = config.additional_config.get("role")
    node_num = int(config.additional_config.get("node_num"))
    print(f'attn_ranks is {attn_ranks}')
    if node_num > 1:
        local_rank = rank % ffn_size
        ffn_default_group = create_ffn_process_group(local_rank, world_size,attn_size, ffn_size)
    else:
        ffn_default_group = create_ffn_process_group(rank, world_size,attn_size, ffn_size)
    
    
    # new_default_group
    # global _NEW_DEFAULT_GROUP
    ps._NEW_DEFAULT_GROUP ,ps._NEW_CPU_DEFAULT_GROUP= creat_hccl_process_group(rank, len(attn_ranks) + len(ffn_ranks))
    # switcher, update default group to new_default_group
    pre_default_group = _get_default_group()
    set_process_group(ps._NEW_DEFAULT_GROUP)
    sub_group_ranks = []
    for i in range(len(ffn_ranks)):
        ranks = list([attn_ranks[i],ffn_ranks[i]])
        sub_group_ranks.append(ranks)
    ps._AE_GROUP = init_model_parallel_group(sub_group_ranks,
                                rank,
                                backend='hccl', 
                                group_name="ae")
    # send/recv in sub_group       [[0, 2], [1, 3]]
    data = torch.tensor([rank]).npu()
    # with default_pg_switcher:
    print(f'Sub Group recv Before: rank={rank}, data={data}') # [0, 1, 2, 3]
    # dist.send(tensor=data, dst=rank + 2,group=_AE_GROUP.device_group)
    ps._AE_GROUP.recv(data.size(),data.dtype)
    print(f'Sub Group recv After: rank={rank}, data={data}')  # 

    reset_process_group(pre_default_group)
    # --------- test cpu ---#
    # send/recv in sub_group       [[0, 2], [1, 3]]
    data = torch.tensor([rank])
    print(f'Sub cpu Group recv Before: rank={rank}, data={data}') # [0, 1, 2, 3]
    dist.recv(tensor=data, src=rank-2,group=ps._NEW_CPU_DEFAULT_GROUP)
    print(f'Sub cpu Group recv After: rank={rank}, data={data}')  

    print(f'rank={rank},create global process group success')   
    print(f'rank={rank},start to run model') 
    
    """Initialize the worker for Ascend."""
    # register patch for vllm
    from vllm_ascend.utils import adapt_patch
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
    from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
    adapt_patch()
    # Register ops when worker init.
    from vllm_ascend import ops
    ops.register_dummy_fusion_op()
    _register_atb_extensions()
    # # init ascend config
    # init_ascend_config(vllm_config)

    
    
    ffn_worker = create_worker(
        FFNWorker,
        model_runner_cls=FFNModelRunner,
        engine_config = config,
        rank = rank,
        ffn_size = ffn_size
    )

    
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


from vllm_ascend.platform import NPUPlatform
import threading

class FFNWorker(NPUWorker):

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            # Additional parameters for compatibility with vllm
            **kwargs):
        
        """Initialize the worker for Ascend."""
        # register patch for vllm
        from vllm_ascend.utils import adapt_patch
        from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
        adapt_patch()
        # Register ops when worker init.
        from vllm_ascend import ops
        ops.register_dummy_fusion_op()
        _register_atb_extensions()
        # init ascend config
        init_ascend_config(vllm_config)

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

    def init_device(self):
        node_num = int(self.vllm_config.additional_config.get("node_num"))
        if node_num > 1:
            device = torch.device(f"npu:{self.local_rank}")
        else:
            device = torch.device(f"npu:{self.rank}")
        NPUPlatform.set_device(device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]

        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        NPUPlatform.seed_everything(self.model_config.seed)

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = FFNModelRunner(self.vllm_config, device)

    def execute_model(
        self,
    ):

        output = self.model_runner.execute_model()
        return output

def create_config() -> VllmConfig:
    #/mnt/nfs/DeepSeek-V2-Lite
    #/mnt/nfs/y00889327/y00889327_DeepSeek-V3.1_w8a8mix_mtp
    engine_args = EngineArgs(
        model="/mnt/nfs/DeepSeek-V2-Lite",
        enforce_eager=True,
        trust_remote_code=True,
        # quantization="ascend",
        tensor_parallel_size=2,
        max_model_len=2048,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.95,
        additional_config={
            # 关闭chunked_prefill ,调度器走vllm-ascend 重写的调度器，V0
            'ascend_scheduler_config':{
                'enabled': True,},
            "enable_afd":True,
            "enable_ms_afd":False,
            "attn_ranks": [0,1],
            "ffn_ranks": [2,3],
            "role":"ffn",
            "attn_num": 2,
            "ffn_num": 2,
            "node_num": 1,
            # "torchair_graph_config":{
            #          "enabled":True,
            #         #  "enable_kv_nz":False,
            #         #  "enable_multistream_mla":False,
            #         #  "enable_multistream_moe":False,
            #         #  "graph_batch_sizes":[28],
            #         #  "enable_super_kernel":False, 
            #         #  "use_cached_graph":False
            #         }
            }
    )
    engine_config = engine_args.create_engine_config()  
    return engine_config
    
def create_worker(cls: Callable[..., T],
                  model_runner_cls: Optional[ModelRunner] = None,
                  engine_config: VllmConfig = None,
                  **kargs) -> T:
    rank = kargs.get('rank')
    ffn_size = kargs.get('ffn_size')

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    print(f'create worker rank is ========= {rank}')
    worker = cls(
        vllm_config = engine_config,
        local_rank = rank % ffn_size ,
        rank= rank,
        distributed_init_method = distributed_init_method,
        is_driver_worker = False,
        model_runner_cls = model_runner_cls,
    )

    worker.init_device()
    worker.load_model()
    worker.execute_model()


from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm.forward_context import set_forward_context
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               AscendSocVersion, ProfileExecuteDuration,
                               get_ascend_soc_version, is_310p,
                               lmhead_tp_enable)
class FFNModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):

        super().__init__(vllm_config=vllm_config,
                         device=device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
    
    @torch.inference_mode()
    def execute_model(self):
        """Execute FFN computation for a single request batch"""
        print('ffn forward begain')

        # skip dummy_run
        
        # recv ffn_need_forward_data
        new_cpu_default_group = get_new_cpu_default_group()
        rank = get_world_group().rank_in_group
        # TODO define a comptue_dst_rank function
        src = rank 
        print(f'ffn local_rank is {rank}')
        print(f'ffn src is {src}')
        # moe_comm_method = "mc2"
        # recv dummy ffn_need_forward_data
        with_prefill = True
        ffn_need_forward_data = recv_object(src,new_cpu_default_group)
        with_prefill = ffn_need_forward_data.with_prefill
        is_dummy_run = ffn_need_forward_data.is_dummy_run
        
        if is_dummy_run:
            print(f'ffn_need_forward_data.moe_comm_method is {ffn_need_forward_data.moe_comm_method}')
            print(f'ffn_need_forward_data.num_input_tokens is {ffn_need_forward_data.num_input_tokens}')
            print(f'ffn_need_forward_data.with_prefill is {ffn_need_forward_data.with_prefill}')
            print(f'ffn_need_forward_data.total_num_scheduled_tokens is {ffn_need_forward_data.total_num_scheduled_tokens}')
            print(f'dummy run ffn_need_forward_data.is_dummy_run is {ffn_need_forward_data.is_dummy_run}')
           
            with ProfileExecuteDuration().capture_async("forward"):
                with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        num_tokens=ffn_need_forward_data.num_input_tokens,
                        with_prefill=ffn_need_forward_data.with_prefill,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        moe_comm_method=ffn_need_forward_data.moe_comm_method,
                        prefetch_stream=self.prefetch_stream,
                        num_actual_tokens=ffn_need_forward_data.total_num_scheduled_tokens,
                        model_instance=self.model):
                    
                    layers_num = len(self.model.model.layers)
                    for i in range(layers_num):
                        self.model.model.layers[i].ffn_forward()
                        print(f'dummy run layerid is {i}')

        while True:
            ffn_need_forward_data = recv_object(src,new_cpu_default_group)
            with_prefill = ffn_need_forward_data.with_prefill
            is_dummy_run = ffn_need_forward_data.is_dummy_run
            # if not ffn_need_forward_data.is_dummy_run:
            # ffn_need_forward_data = recv_object(src,new_cpu_default_group)
            moe_comm_method = ffn_need_forward_data.moe_comm_method
            num_input_tokens = ffn_need_forward_data.num_input_tokens
            total_num_scheduled_tokens = ffn_need_forward_data.total_num_scheduled_tokens
            print(f'ffn_need_forward_data.moe_comm_method is {moe_comm_method}')
            print(f'ffn_need_forward_data.num_input_tokens is {num_input_tokens}')
            print(f'ffn_need_forward_data.with_prefill is {with_prefill}')
            print(f'ffn_need_forward_data.total_num_scheduled_tokens is {total_num_scheduled_tokens}')
            print(f'ffn_need_forward_data.is_dummy_run is {ffn_need_forward_data.is_dummy_run}')

            # TODO dp
            # num_tokens_across_dp
            # TODO graph 
            # aclgraph_runtime_mode
            # batch_descriptor
            
            # Run forward pass
            with ProfileExecuteDuration().capture_async("forward"):
                with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        num_tokens=num_input_tokens,
                        with_prefill=with_prefill,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        moe_comm_method=moe_comm_method,
                        prefetch_stream=self.prefetch_stream,
                        num_actual_tokens=total_num_scheduled_tokens,
                        model_instance=self.model):
                    layers_num = len(self.model.model.layers)
                    for i in range(layers_num):
                        self.model.model.layers[i].ffn_forward()
                        print(f'layerid is {i}')   
            print('ffn forward finished')
     
if __name__ == '__main__':
    hccl_world_size = 4
    attn_size = 2
    ffn_size = 2

    hccl_processes = []
    for rank in range(ffn_size,hccl_world_size):
        p = mp.Process(target=run_ffn, args=(rank, hccl_world_size, attn_size, ffn_size))
        hccl_processes.append(p)
        p.start()


    for p in hccl_processes:
        p.join()

    print("All processes finished")
