# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field, fields
import json
import threading
from vllm.logger import logger
import omni.adaptors.vllm.envs as envs
import os

@dataclass
class ModelParallelConfig:
    dense_mlp_tp_size: int = 1
    dp_size: int = 1
    o_proj_tp_size: int = 1
    attn_sp_size: int = 1
    
    redundancy_shared_expert_num: int = 0

 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_moe_all_to_all: bool = True
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    gmm_nz: bool = False
    unquant_bmm_nz: bool = False
    decode_moe_dispatch_combine: bool = True
    use_omni_placement: bool = False
    omni_placement_config_path:str = None
    enable_moe_expert_parallel: bool = True
    use_super_kernel: bool = False
    enable_prefill_micro_batch: bool = False
    use_mlaprolog: bool = False
    opt_w2_scale_cast: bool = False
    enable_mc2_v2: bool = False
    decode_gear_list: list[int] = field(default_factory=lambda: [1])
    control_accept_rate: float = -1 # <0 or >1 不控制, >=0 and <=1 控制MTP开启时接受率为该值，几乎必然导致输出结果异常，仅保证只投机1个token时满足这一数值

    use_prefetch: bool = True # 是否开启预取
    expert_gate_up_prefetch: int = 50 # 默认预取大小为 50Mb；如果是权重是BF16型，设置为 30Mb
    expert_down_prefetch: int = 28 # 当权重是w8a8且ep_size > 64 时，默认预取大小为 28Mb，否则为0
    attn_prefetch: int = 96 # 默认预取大小为 96Mb

    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    pd_seperate_prefill: bool = False
    prefill_enable_long_seq: bool = False
    prefill_enable_mla_alltoall: bool = False
    prefill_enable_mla_alltoall_local: bool = False
    fa_quant: bool = False
    enable_dsa: bool = False # 使能mla = Indexer + select FA
    use_omni_cache: bool = False
    
    max_split_token_ratio_threshold: float = 0.8 # Split hidden_states in prefill if token duplication ratio exceeds threshold, to avoid GMM OOM.
    max_split_token_count_threshold: int = 32768 # Split hidden_states in prefill if token duplication count exceeds threshold, to avoid GMM OOM.

    def __post_init__(self):
        # Check the dependencies of use_omni_placement and omni_placement_config_path
        if self.use_omni_placement and not self.omni_placement_config_path:
            raise ValueError(
                "When use_omni_placement=True, omni_placement_config_path must be provided!"
            )
        
        if os.getenv("ENABLE_OMNI_CACHE", "0") == "1":
            self.use_omni_cache = True

        # Check the dependencies of use_prefetch and prefetch_Mb
        if not self.use_prefetch:
            self.expert_gate_up_prefetch = 0
            self.expert_down_prefetch = 0
            self.attn_prefetch = 0

@dataclass      
class ModelExtraConfig:
    parall_config: ModelParallelConfig = field(default_factory=ModelParallelConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory=ModelOperatorOptConfig)
    model_extra_cfg_path: str = ""


def filter_dict_by_dataclass(dataclass_type, data_dict):
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data_dict.items() if k in valid_keys}


def init_model_extra_config() -> ModelExtraConfig:
    model_config = ModelExtraConfig()
    model_extra_cfg_path = envs.MODEL_EXTRA_CFG_PATH

    try:
        with open(model_extra_cfg_path, 'r') as f:
            config_data = json.load(f)
        # Recursively create nested objects
        parall_config = ModelParallelConfig(**config_data['model_parallel_config'])
        operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimizition_config']))
        model_config = ModelExtraConfig(
                parall_config=parall_config,
                operator_opt_config=operator_opt_config,
                model_extra_cfg_path=model_extra_cfg_path)
    except FileNotFoundError:
        logger.warning(f"[WARNING] Config file not found: {model_extra_cfg_path}, using default configuration.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Invalid JSON format in config file: {e}")
    except KeyError as e:
        raise RuntimeError(f"[ERROR] Missing required key in config data: {e}")
    except TypeError as e:
        raise RuntimeError(f"[ERROR] Config structure mismatch or incorrect field types: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error while loading model extra config: {e}")

    return model_config

_model_extra_config = None
_model_extra_config_lock = threading.Lock()

def get_model_extra_config():
    global _model_extra_config
    if _model_extra_config is None:
        with _model_extra_config_lock:
            if _model_extra_config is None:
                _model_extra_config = init_model_extra_config()
    return _model_extra_config

def update_model_extra_config(**kwargs):
    global model_extra_config
    model_extra_config = get_model_extra_config()
    operator_opt_config = getattr(model_extra_config, 'operator_opt_config', None)
    if operator_opt_config is not None and kwargs:
        for key, value in kwargs.items():
            if hasattr(operator_opt_config, key):
                setattr(operator_opt_config, key, value)
                logger.info(f"{key} loads from additional config: {value}")
        if 'enable_torchair_graph_mode' in kwargs and not kwargs['enable_torchair_graph_mode']:
            operator_opt_config.moe_multi_stream_tune = False
            operator_opt_config.use_super_kernel = False
            operator_opt_config.use_prefetch = False
            operator_opt_config.use_mlaprolog = False

model_extra_config = get_model_extra_config()

