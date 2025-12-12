from vllm_ascend.worker.common import UCEType
from typing import Tuple,List
class MemoryBlockInfo:
    def __init__(self,model):
        self.model = model
        self.weight_blocks = {}
        self.kvcache_blocks = {}
        self.initialized = False

    def initialize(self):
        self._get_weight_memory_info()
        self._get_kv_memory_info()
        self.initialized = True

    def _get_weight_memory_info(self):
        weights_blocks = {}
        state_dict = self.model.state_dict()
        for name,param in state_dict.items():
            start_address = param.data_ptr()
            size_bytes = param.numel() * param.element_size()
            end_address = start_address + max(0,size_bytes - 1)
            weights_blocks[name] = {
                'name':name,
                'start_address':start_address,
                'end_address':end_address,
            }
        self.weight_blocks = weights_blocks
    def _get_kv_memory_info(self):
        pass

    def category_address(self,ptr) -> Tuple[UCEType,List[str]]:
        weight_type,weight_layer = self.is_weight_uce(ptr)
        if weight_type != None:
            return weight_type,weight_layer

        kv_type,kv_layer = self.is_kv_uce(ptr)
        if kv_type != None:
            return kv_type,kv_layer

        return UCEType.ACTIVATION_UCE,[]

    def is_weight_uce(self,ptr) -> Tuple[UCEType,List[str]]:
        error_layer = []
        for name in self.weight_blocks:
            start_address = self.weight_blocks[name]['start_address']
            end_address = self.weight_blocks[name]['end_address']
            if start_address <= int(ptr) <= end_address:
                error_layer.append(name)
        if len(error_layer) > 0:
            return UCEType.WEIGHTS_UCE,error_layer
        return None,[]

    def is_kv_uce(self,ptr) -> Tuple[UCEType,List[str]]:
        pass