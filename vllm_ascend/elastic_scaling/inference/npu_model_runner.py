import os
from copy import deepcopy
from multiprocessing import Manager

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_dcp_group,
                                             get_pcp_group)
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.kv_cache_interface import (KVCacheConfig,
                                        MambaSpec)
from vllm.v1.sample.logits_processor import build_logitsprocs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
# yapf: enable
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.utils import set_weight_prefetch_method
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    set_mc2_mask, set_mc2_tokens_capacity)


from worm_ipc.worm_client import WormNPUClient

from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.model_executor.model_loader.utils import initialize_model

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.model_runner_v1 import ExecuteModelState

def __init__(self, vllm_config: VllmConfig, device: torch.device):
    # TODO(qcs): These manual pad and unpad for GPUModelRunner are
    # used to expand some buffers, which need to be reverted after
    # the following PR is merged:
    # https://github.com/vllm-project/vllm/pull/28988
    max_pcp_pad_tokens = vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
    vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
    with _torch_cuda_wrapper():
        super(NPUModelRunner, self).__init__(vllm_config, device)
    vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
    self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
    self.max_num_reqs = self.scheduler_config.max_num_seqs
    self.dp_size = vllm_config.parallel_config.data_parallel_size
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank
    try:
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group(
        ).rank_in_group if self.pcp_size > 1 else 0
    except Exception:
        self.dcp_size = 1
        self.dcp_rank = 0
        self.pcp_size = 1
        self.pcp_rank = 0
    if self.pcp_size > 1:
        self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
    max_buffer_num_tokens = self.max_num_tokens
    if self.pcp_size * self.dcp_size > 1:
        max_buffer_num_tokens = (self.max_num_tokens +
                                    self.max_num_reqs * 2 * self.pcp_size)
        self.pcp_manager = PCPManager(
            self.pcp_size,
            self.pcp_rank,
            self.dcp_size,
            self.dcp_rank,
            max_buffer_num_tokens,
            self.max_num_reqs,
            self.device,
            self.vllm_config,
            self.use_async_scheduling,
            self.pin_memory,
        )
        # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
        self.input_ids = self._make_buffer(max_buffer_num_tokens,
                                            dtype=torch.int32)
        self.positions = self._make_buffer(max_buffer_num_tokens,
                                            dtype=torch.int64)
    self.sampler = AscendSampler()
    self.attn_state: AscendAttentionState | None = None

    # Ascend-specific configurations
    self.ascend_config = get_ascend_config()
    set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)
    # Dump / PrecisionDebugger configuration now comes from AscendConfig
    dump_cfg = self.ascend_config.dump_config_path
    self.debugger = None
    if dump_cfg is not None:
        if self.model_config.enforce_eager:
            from msprobe.pytorch import PrecisionDebugger
            self.debugger = PrecisionDebugger(dump_cfg)
        else:
            raise RuntimeError(
                "Dumping/debugging only works in eager mode.")
    # use_hybrid_blocks: if hybrid blocks is used.
    self.use_hybrid_blocks: bool = False
    self.need_accepted_tokens: bool = False

    self.is_multimodal_model = self.model_config.is_multimodal_model
    self.block_size = vllm_config.cache_config.block_size
    # Set up Attention
    self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config,
                                "index_topk")
    self.attn_backend = get_attn_backend(
        0,
        self.dtype,
        None,
        self.block_size,
        use_mla=self.model_config.use_mla,
        use_sparse=self.use_sparse,
        use_mm_prefix=self.model_config is not None
        and self.model_config.is_mm_prefix_lm)

    self._set_up_drafter()

    # kv role
    self.is_kv_producer = False
    self.is_kv_consumer = False
    if vllm_config.kv_transfer_config is not None:
        self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

    set_cos_and_sin(vllm_config, self.max_num_reqs,
                    self.uniform_decode_query_len, self.dtype, self.device)
    set_mc2_tokens_capacity(vllm_config, self.max_num_reqs,
                            self.uniform_decode_query_len)
    set_mc2_mask(vllm_config, self.device)
    self.decode_threshold = 1 + (
        self.speculative_config.num_speculative_tokens
        if self.speculative_config else 0)

    self.use_aclgraph = self._use_aclgraph()

    eplb_config = self.ascend_config.eplb_config
    self.dynamic_eplb = eplb_config.dynamic_eplb
    if self.dynamic_eplb:
        self.is_eplb_warmuped = False
        self.policy_type = eplb_config.eplb_policy_type
        self.eplb_loader = D2DExpertWeightLoader()
        self.manager = Manager()
        self.shared_dict = self.manager.dict({
            "expert_map": None,
            "moe_load": None,
            "expert_maps": None
        })
        self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                        policy_type=self.policy_type,
                                        enable_d2d=True)
        self.process = self.eplb_process._launch_process()
        self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader,
                                        self.eplb_process, self.process)
    # Input Batch
    # NOTE(Chen): Ideally, we should initialize the input batch inside
    # `initialize_kv_cache` based on the kv cache config. However, as in
    # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
    # reasons, we have to initialize the input batch before `load_model`,
    # quantization + weight offloading will fail otherwise. As a temporary
    # solution, we initialize the input batch here, and re-initialize it
    # in `initialize_kv_cache` if the block_sizes here is different from
    # the block_sizes in the kv cache config.
    self.input_batch = NPUInputBatch(
        max_num_reqs=self.max_num_reqs,
        max_model_len=max(self.model_config.max_model_len,
                            self.max_encoder_len),
        max_num_batched_tokens=self.max_num_tokens,
        device="npu",
        pin_memory=self.pin_memory,
        vocab_size=self.model_config.get_vocab_size(),
        block_sizes=[self.block_size],
        kernel_block_sizes=[[self.cache_config.block_size]],
        is_spec_decode=bool(self.vllm_config.speculative_config),
        logitsprocs=build_logitsprocs(
            self.vllm_config, self.device, self.pin_memory,
            self.is_pooling_model,
            self.vllm_config.model_config.logits_processors),
        is_pooling_model=self.is_pooling_model,
        num_speculative_tokens=(
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config else 0),
        cp_kv_cache_interleave_size=self.parallel_config.
        cp_kv_cache_interleave_size,
    )
    self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                dtype=torch.int32)
    # here we use int32
    self.sampled_token_ids_pinned_cpu = torch.empty(
        (self.max_num_reqs, 1),
        dtype=torch.int32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    # for cleancode , actually the three attrs is defined in gpu_model_runner
    self.execute_model_state: ExecuteModelState | None = None
    # None in the first PP rank. The rest are set after load_model.
    self.intermediate_tensors: IntermediateTensors | None = None
    self.reorder_batch_threshold: int | None = None
    self.long_seq_metadata = None

    ## Initialize WORM client

    self.ipc_role = ""

    ## get the parallel info to connect to the correct socket
    tp_rank = int(self.parallel_config.rank)
    tp_size = int(self.parallel_config.tensor_parallel_size)
    dp_rank = int(self.parallel_config.data_parallel_rank)
    dp_size = int(self.parallel_config.data_parallel_size)
    self.current_device = torch.npu.current_device()

    self.ipc_config = {
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "device_id": self.current_device
    }



def load_model(self) -> None:
    logger.info("Starting to load model %s...", self.model_config.model)

    with DeviceMemoryProfiler() as m:  # noqa: SIM117
        if int(os.getenv("ELASTIC", "0")) > 0:
            if int(os.getenv("INFER_STATUS", "0")) > 0:
                self.ipc_engine = WormNPUClient(self.ipc_config)
                self.vllm_config.device_config.device = torch.device("cpu")
                self.vllm_config.device_config.device_type = "cpu"
                
                target_device = self.vllm_config.device_config.device

                with set_default_torch_dtype(self.model_config.dtype):
                    with target_device:
                        self.cpu_model = initialize_model(vllm_config=self.vllm_config,
                            model_config=self.model_config)
                self.vllm_config.device_config.device = torch.device("npu")
                self.vllm_config.device_config.device_type = "npu"
                
                from vllm.model_executor.model_loader import get_model_loader
                self.loader = get_model_loader(self.vllm_config.load_config)

                self.model = self.loader.zero_copy_model(self.vllm_config,
                                self.ipc_engine, self.cpu_model, self.model_config)
            else:
                logger.info("skipping model loading for pre-initialized inference client")

        else: # not elastic (should never run here anyways with this patch)
            raise Exception("Should not be in --elastic mode")

    logger.info("Loading model weights took %.4f GB",
                m.consumed_memory / float(2**30))

    # wrap the model with full graph wrapper if needed.
    if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
        self.update_stream: torch.npu.Stream = torch.npu.Stream()
        self.model = ACLGraphWrapper(self.model,
                                        self.vllm_config,
                                        runtime_mode=CUDAGraphMode.FULL)



def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    """
    Initialize KV cache based on `kv_cache_config`.
    Args:
        kv_cache_config: Configuration for the KV cache, including the KV
        cache size of each layer
    """
    kv_cache_config = deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config
    if int(os.getenv("ELASTIC", "0")) > 0:
        if int(os.getenv("INFER_STATUS", "0")) > 0:

            
            self.may_add_encoder_only_layers_to_kv_cache_config()
            self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
            # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
            self.initialize_attn_backend(kv_cache_config)
            self.use_hybrid_blocks = (len(self.attn_groups) > 1)
            # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
            self.need_accepted_tokens = any([
                isinstance(attn_group[0].kv_cache_spec, MambaSpec)
                for attn_group in self.attn_groups
            ])
            self.may_reinitialize_input_batch(kv_cache_config)
            kv_cache_raw_tensors = self.ipc_engine.zero_copy_kv_caches()

            ## Here we need to reshape it to the expected shape
            kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                kv_cache_raw_tensors)

            # Set up cross-layer KV cache sharing
            for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
            ):
                logger.debug("%s reuses KV cache of %s", layer_name,
                            target_layer_name)
                kv_caches[layer_name] = kv_caches[target_layer_name]
        else: ## we are pre-initializing the client without loading weights or kv
            return
    else: ## for server side we don't touch anything
        raise Exception("Should not be in --elastic mode")

    from vllm.v1.worker.utils import bind_kv_cache
    num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
    bind_kv_cache(kv_caches,
                    self.compilation_config.static_forward_context,
                    self.kv_caches, num_attn_module)

    if has_kv_transfer_group():
        get_kv_transfer_group().register_kv_caches(kv_caches)
        