# Copyright (c) HuaWei Technologies Co., Ltd. 2025-2025. All rights reserved
from typing import List

import blake3
from ems import KvBufferWrapper
from vllm.logger import logger

from omni.adaptors.vllm.ems.ems_env import EmsEnv


class EmsKeyGenerator:
    _SPLITER = "@"

    def __init__(self, pp_size: int, tp_size: int, tp_rank: int):
        """tp_rank: 当前卡在tp group中的rank"""
        self.key_prefix = self._SPLITER.join(
            [f"{EmsEnv.llm_engine}", f"{EmsEnv.access_id}", f"pp{pp_size}", f"tp{tp_size}", f"{tp_rank}"])
        
        logger.info(f"[EMS] Init ems key generator, key_prefix: {self.key_prefix}")

    def gen_key(self, suffix):
        return self._SPLITER.join([f"{self.key_prefix}", f"{suffix}"])



class EmsKVCacheManager:
    """KV缓存管理器，负责KV缓存的初始化和访问"""

    def __init__(self):
        # KV缓存初始化参数
        self._kvcache_initialized = False
        self._key_size = 0
        self._value_size = 0
        self._block_num = 0
        self._block_addr_list = None

    def initialize_kvcache(self, kv_caches):
        """
        MOE模型: 只有k的缓存
        非MOE模型: 包含k/v两种缓存

        kvcache形状 [layers, k_v_index, GPU_blocks, Block_size, Attention heads Number, Head_size}]
        k_cache: k_v_index = 0
        v_cache: k_v_index = 1

        用于初始化kvcache缓存相关的全局变量
        并预计算各层kvcache的内存地址偏移起始点。
        该函数仅在首次调用时执行。
        """
        if not kv_caches:
            return
        
        # 首次调用时初始化缓存参数
        if not self._kvcache_initialized:
            self._block_num = len(kv_caches[0][1])
            self._block_addr_list = [[] for _ in range(self._block_num)]
            self._initialize_format_kvcache(kv_caches)
            self._kvcache_initialized = True
            logger.info(
                f"[EMS] Init kvcache finished - key_size: {self._key_size}, value_size: {self._value_size},"
                f"kvcache layer key shape: {kv_caches[0][0].shape}, kvcache layer value shape: {kv_caches[0][1].shape}")

    def _initialize_format_kvcache(self, kv_caches):
        """
        初始化非字典格式的KV缓存

        非字典格式时：
        kvcache形状：layers个 [k_v_index, GPU_blocks, Block_size, Attention heads Number, Head_size}]
        """
        # moe模型head_size = (512+64)，大EP方案使用k和v分别保存no rope和rope
        logger.debug(f"[KVCache] 开始初始化非字典格式缓存")

        first_layer_cache = kv_caches[0]
        # 提取参考块的元数据
        first_layer_k_cache = first_layer_cache[0]
        first_layer_v_cache = first_layer_cache[1]

        self._key_size = first_layer_k_cache[0].element_size() * first_layer_k_cache[0].numel()
        self._value_size = first_layer_v_cache[0].element_size() * first_layer_v_cache[0].numel()

        for layer_cache in kv_caches:
            layer_k = layer_cache[0]
            layer_v = layer_cache[1]
            for block_id in range(self._block_num):
                if self._value_size == 0:
                    self._block_addr_list[block_id].extend(
                        [KvBufferWrapper(layer_k[block_id].data_ptr(), self._key_size)])
                else:
                    self._block_addr_list[block_id].extend([
                        KvBufferWrapper(layer_k[block_id].data_ptr(), self._key_size),
                        KvBufferWrapper(layer_v[block_id].data_ptr(), self._value_size)])
        
        logger.info(
            f"[EMS][KVCache] first_layer_k_cache shape is {first_layer_k_cache.shape}, "
            f"first_layer_v_cache shape is {first_layer_v_cache.shape}, "
            f"block num: {len(self._block_addr_list)},slice num per block is {len(self._block_addr_list[0])}")
        
    def get_block_kv_buffer(self, block_id: int) -> List[KvBufferWrapper]:
        """
        根据kv_caches缓存信息通过偏移获取特定block的KV缓存

        参数:
        - block_id: 请求的block的ID，基于该ID进行缓存地址的计算
        """
        return self._block_addr_list[block_id]
    

def cal_hash_blocks(token_ids: List[int], block_size: int) -> List[int]:
    result = []
    num_blocks = len(token_ids) // block_size
    prev_block_hash = 0

    for block_id in range(num_blocks):
        block_hash = cal_block_hash(token_ids[block_id * block_size:(block_id + 1) * block_size], prev_block_hash)
        result.append(block_hash)
        prev_block_hash = block_hash

    return result

def cal_block_hash(block_token_ids: List[int], prev_block_hash: int) -> int:
    return hash((prev_block_hash, *block_token_ids))