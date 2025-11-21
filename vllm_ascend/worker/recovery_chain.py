import json
import os.path
import torch
import yaml


from abc import ABC, abstractmethod
from typing import List,Tuple,Dict,Any
from vllm.logger import logger
from vllm_ascend.worker.common import RecoveryStatus,FaultStatus,UCEType,FaultToleranceLevel
from vllm_ascend.worker.recovery_context import RecoveryContext
from torch_npu.npu.utils import _get_uce_addr
from collections.abc import Generator
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from safetensors.torch import safe_open
"""
all_err = [
    "UCE ERROR",
    "HBM MULTI BIT ECC ERROR",
    "FORCE STOP",
    "SUSPECT MEM ERROR",
    "HCCS LINK ERROR",
    "HCCL OP RETRY FAILED",
    "SUSPECT REMOTE ERROR"
]
"""

uce_error = [
    "UCE ERROR",
    "HBM MULTI BIT ECC ERROR"
]
force_stop_error = ["FORCE STOP"]
network_error = [
    "SUSPECT REMOTE ERROR",
    "HCCL OP RETRY FAILED"
]


class RecoveryHandler(ABC):
    """责任链处理器基类"""
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'RecoveryHandler') -> 'RecoveryHandler':
        """Set next handler"""
        self.next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, ctx:RecoveryContext) -> bool:
        pass

    @abstractmethod
    def recover(self, ctx:RecoveryContext) -> torch.Tensor:
        """Specific recovery function"""
        pass

    def handle(self, ctx:RecoveryContext) -> torch.Tensor:
        """ Entry point for RecoveryHandler """
        if self.can_handle(ctx):
            return self.recover(ctx)
        elif self.next_handler:
            return self.next_handler.handle(ctx)
        else:
            logger.warning("No handler can process the exception")
            return RecoveryStatus.FAILED_ABORT


class ForceStopHandler(RecoveryHandler):
    """处理Force Stop异常的处理器"""

    def can_handle(self, ctx:RecoveryContext) -> bool:
        """判断是否为Force Stop异常，如果是则入队"""
        error_str = str(ctx.exception).lower()
        for error in force_stop_error:
            if error in error_str:
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """处理Force Stop异常，仅返回状态"""
        return RecoveryStatus.SUCCESS_RECOMPUTE

class NetworkHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in network_error:
            if error in error_str:
                ctx.fault_queue.put_nowait(FaultStatus.NETWORK_ERR)
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """恢复Network Error,无特殊操作"""
        return RecoveryStatus.SUCCESS_RECOMPUTE

class UCEHandler(RecoveryHandler):
    """统一处理UCE异常的处理器"""
    def can_handle(self, ctx:RecoveryContext) -> bool:
        """判断是否为UCE异常，如果是则入队"""
        error_str = str(ctx.exception).lower()
        for error in uce_error:
            if error in error_str:
                ctx.fault_queue.put_nowait(FaultStatus.UCE_ERR)
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """处理UCE异常，内部判断具体类型并执行恢复"""
        #1.判断类型
        uce_result = self.classify_uce_type(ctx)
        recovery_statuses = []
        #2.根据类型执行恢复策略
        for uce_type,layer_names in uce_result:
            if uce_type == UCEType.KVCACHE_UCE.name:
                recovery_statuses.append(self._recover_kv_cache_uce(ctx,layer_names))
            elif uce_type == UCEType.WEIGHTS_UCE.name:
                recovery_statuses.append(self._recover_weight_uce(ctx,layer_names))
            elif uce_type == UCEType.ACTIVATION_UCE.name:
                recovery_statuses.append(self._recover_activation_uce(ctx))
            else:
                logger.error(f"UCEHandler: Unknown UCE type: {uce_type}")
                recovery_statuses.append(RecoveryStatus.FAILED_ABORT)
        if RecoveryStatus.FAILED_ABORT in recovery_statuses:
            return RecoveryStatus.FAILED_ABORT
        return RecoveryStatus.SUCCESS_RECOMPUTE

    def classify_uce_type(self,ctx:RecoveryContext) -> List[Tuple[UCEType,List[str]]]:
        #1.获取出现uce的地址信息
        try:
            memory_block_info = ctx.memory_block_info
            if not memory_block_info.initialized:
                memory_block_info.initialize()
            uce_ptrs = _get_uce_addr()
            if not uce_ptrs:
                logger.error(f"UCEHandler: No UCE addr found")
                return [(UCEType.UNKNOWN_UCE,[])]
            uce_results = []
            for uceptr in uce_ptrs:
                uce_type,layer_names = ctx.memory_block_info.category_address(uceptr)
                uce_results.append((uce_type,layer_names))
            return uce_results
        except Exception as e:
            logger.error(f"UCEHandler:Failed to classify UCE type，{e}")
            raise RuntimeError("Failed to classify UCE type")

    def _recover_weight_uce(self, ctx:RecoveryContext,layer_names:List[str]) -> RecoveryStatus:
        """恢复权重UCE错误"""
        # 1. 出错层名称检查
        if not layer_names:
            logger.error(f"UCEHandler:layer_names is empty")
            return RecoveryStatus.FAILED_ABORT

        logger.info(f"UCEHandler: Recovering weight UCE for layer: {layer_names}")
        # 2. 增量重加载权重
        original_weights_file_name = []
        for layer_name in layer_names:
            original_weights_file_name.extend(self.map_to_original_param(layer_name))
        try:
            weight_iterator = self.get_weight_iterator(ctx,original_weights_file_name)
            loaded_weights = ctx.model.load_model(weight_iterator)
            #TODO:这里可能要判断一下是否把需要加载的权重都加载成功了
            return RecoveryStatus.SUCCESS_RECOMPUTE
        except Exception as e:
            logger.error(f"UCEHandler: Weight reload failed: {e}")
            return RecoveryStatus.FAILED_ABORT

    def _recover_kv_cache_uce(self, ctx:RecoveryContext,layer_names:List[str]) -> RecoveryStatus:
        """恢复KV Cache UCE错误"""
        level = ctx.level

        if level == FaultToleranceLevel.BASIC:
            logger.warning("UCEHandler: KV Cache UCE in BASIC level, aborting recovery")
            return RecoveryStatus.FAILED_ABORT

        if level == FaultToleranceLevel.FULL:
            try:
                pass
            except Exception as e:
                logger.error(f"UCEHandler: KV Cache recovery failed: {e}")
                return RecoveryStatus.FAILED_ABORT

        logger.warning(f"UCEHandler: Unsupported fault tolerance level: {level}")
        return RecoveryStatus.FAILED_ABORT

    def _recover_activation_uce(self, ctx:RecoveryContext) -> RecoveryStatus:
        """恢复激活值UCE错误"""
        logger.info("UCEHandler: Activation UCE detected, no special recovery needed")
        # 激活值UCE无需特殊恢复，直接返回成功
        return RecoveryStatus.SUCCESS_RECOMPUTE


    def _load_mapping_config(self,config_path:str)->Dict[str,List[Tuple[str,Any]]]:
        with open(config_path,'r',encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f.read())
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("不支持该配置文件格式")

    def map_to_original_param(self,merged_name:str,mapping_config:Dict[str,List[Tuple[str,Any]]] = None) -> List[str]:
        default_mapping={
            "qkv_proj":[
                ("q_proj","q"),
                ("k_proj","k"),
                ("v_proj","v"),
            ],
            "gate_up_proj":[
                ("gate_proj",0),
                ("up_proj",1)
            ]
        }
        mapping = mapping_config if mapping_config is not None else default_mapping
        original_names = []
        for merged_param_name,mappings in mapping.items():
            if merged_param_name in merged_name:
                for original_param_name,_ in mappings:
                    original_name = merged_name.replace(merged_param_name,original_param_name)
                    original_names.append(original_name)
                break
        if not original_names:
            return [merged_name]
        return original_names
    def get_weight_iterator(self,ctx:RecoveryContext,original_names:List[str]) -> Generator[tuple[str,torch.Tensor],None,None]:
        index_file_name = os.path.join(ctx.model_or_path,SAFE_WEIGHTS_INDEX_NAME)
        with open(index_file_name) as f:
            weight_map = json.load(f)["weight_map"]
        weight_files_in_index = set()
        for original_name in original_names:
            weight_files_in_index.add(os.path.join(ctx.model_or_path,weight_map[original_name]))
        for st_file in weight_files_in_index:
            with safe_open(st_file,framework="pt") as f:
                for name in f.keys():
                    param = f.get_tensor(name)
                    yield name,param