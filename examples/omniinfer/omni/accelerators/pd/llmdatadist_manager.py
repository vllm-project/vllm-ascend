# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import time
from collections import defaultdict, namedtuple
from functools import cached_property

import llm_datadist
import torch
from llm_datadist import (BlocksCacheKey, CacheDesc, LLMConfig,
                          LLMDataDist, LLMRole, RegisterMemStatus, LLMException, LLMStatusCode)

from vllm.config import KVTransferConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from omni.accelerators.pd.ranktable.local_info import LocalInfo
from omni.accelerators.pd.ranktable.rank_table import GlobalRankTable, RankTableConfig
from omni.accelerators.pd.utils import get_p_start_rank, prepare_ranktables
from vllm.config import VllmConfig
import os

logger = init_logger(__name__)

_ROLE_STR_TO_ENUM = {
    "kv_producer": LLMRole.PROMPT,
    "kv_consumer": LLMRole.DECODER
}

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}

SCHEDULER_LINK_BATCH_SIZE = 32
SCHEDULER_LINK_INTERVAL = 0.5
KV_CACHE_RETRY_TIMES = 3
KV_CACHE_RETRY_WAIT_SECOND = 2

RETRYABLE_CODES = [
    LLMStatusCode.LLM_REPEAT_REQUEST,
    LLMStatusCode.LLM_CLUSTER_NUM_EXCEED_LIMIT,
    LLMStatusCode.LLM_PROCESSING_LINK,  # Building chain is in progress
    LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY,
    LLMStatusCode.LLM_TIMEOUT,
    LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT,
    LLMStatusCode.LLM_LINK_BUSY,
]


class LLMDataDistConfig:
    """
    Configuration for the separate deployment.
    """

    def __init__(self, vllm_config: VllmConfig, ignore_load_rank=False) -> None:
        additional_config = vllm_config.additional_config
        if additional_config:
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:
            self.multi_rank_pull_kv = False
        self.kv_transfer_config = vllm_config.kv_transfer_config
        ranktable_config = RankTableConfig.from_dict_or_env(vllm_config.kv_transfer_config.kv_connector_extra_config)
        self.global_rank_table = GlobalRankTable(ranktable_config)
        self.local_group = self.global_rank_table.find_group_by_local_info(LocalInfo(ranktable_config))

        kv_role_tmp = self.kv_transfer_config.kv_role
        server_role_tmp = self.global_rank_table.get_server_role(self.local_group.group_id)

        if (kv_role_tmp == "kv_producer" and server_role_tmp == "prefill"):
            logger.info(f"Engine {server_role_tmp} role: {server_role_tmp}")
        elif kv_role_tmp == "kv_consumer" and server_role_tmp == "decode":
            logger.info(f"Engine {server_role_tmp} role: {server_role_tmp}")
        else:
            raise ValueError

        self.cluster_id_start = self.local_group.cluster_id_start

        if ignore_load_rank:
            self.rank = -1
            self.local_rank = -1
            self.cluster_id = -1
        else:
            self.rank = get_world_group().rank_in_group
            self.local_rank = get_world_group().local_rank
            self.cluster_id = self.local_group.device_list[self.rank].cluster_id
        self.kv_parallel_size = self.kv_transfer_config.kv_parallel_size
        self.kv_producer_dp_size = self.kv_transfer_config.kv_connector_extra_config.get("kv_producer_dp_size", 1)

    @cached_property
    def role(self):
        return _ROLE_STR_TO_ENUM[self.kv_transfer_config.kv_role]

    @cached_property
    def is_prefill(self):
        return self.role == LLMRole.PROMPT


class LLMDataDistManager:
    def __init__(self, vllm_config: VllmConfig):
        additional_config = vllm_config.additional_config
        if additional_config:  # pragma: no cover
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:  # pragma: no cover
            self.multi_rank_pull_kv = False
        kv_transfer_config = vllm_config.kv_transfer_config
        self.data_dist_config = LLMDataDistConfig(vllm_config)
        self.rank = self.data_dist_config.rank
        self.local_rank = self.data_dist_config.local_rank

        self.data_dist_engine = self._init_llm_data_dist()

        self.registered_kv_caches = []
        self.rank_link_info_map = {}

        self.registered_link_infos = {}

    def get_real_remote_cluster_ids(self, meta: "ReqMeta"):
        remote_cluster_ids = self.registered_link_infos.get(
            (meta.remote_cluster_id, meta.remote_dp_rank, self.rank), None)
        if remote_cluster_ids is None:
            raise ValueError(f"Could not find remote cluster id from {meta.remote_cluster_id=}, {meta.remote_dp_rank=}.")
        return remote_cluster_ids

    def _init_llm_data_dist(self):
        data_dist = LLMDataDist(self.data_dist_config.role, self.data_dist_config.cluster_id)

        llm_config = LLMConfig()
        llm_config.device_id = self.local_rank
        llm_config.enable_switch_role = True
        llm_config.enable_cache_manager = True

        # RoCE timeout is 20s
        llm_config.sync_kv_timeout = 20000

        llm_config.enable_remote_cache_accessible = True
        options = llm_config.generate_options()
        data_dist.init(options)

        return data_dist

    def register_memory(self, kv_caches: dict[str, torch.Tensor]):
        if len(self.registered_kv_caches) > 0:
            raise ValueError("Attr `registered_kv_caches` must be empty before register kv_caches.")
        if isinstance(kv_caches, dict):
            flatten_kv_caches = unzip_kv_cache_dict(kv_caches)
        else:
            flatten_kv_caches = unzip_kv_cache_list(kv_caches)

        # dense model.
        flatten_kv_caches = maybe_merge_kv_caches(flatten_kv_caches)

        for model_id, sub_kv_caches in enumerate(flatten_kv_caches):
            cache_desc = CacheDesc(num_tensors=len(sub_kv_caches), shape=tuple(sub_kv_caches[0].shape),
                                   data_type=TORCH_DTYPE_TO_NPU_DTYPE[sub_kv_caches[0].dtype])

            cache_addrs = [int(item.data_ptr()) for item in sub_kv_caches]

            if self.data_dist_config.is_prefill:
                cache_key = BlocksCacheKey(self.data_dist_engine.cluster_id, model_id=model_id)
            else:
                cache_key = None

            cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
            self.registered_kv_caches.append(cache)
        logger.error(f" ***** registered_kv_caches num:{len(self.registered_kv_caches)}")

    def _pull_blocks(self, src_cache_key, dst_cache, src_blocks, dst_blocks):
        for _ in range(KV_CACHE_RETRY_TIMES):
            try:
                self.data_dist_engine.cache_manager.pull_blocks(src_cache_key, dst_cache, src_blocks,
                                                                          dst_blocks)
                return
            except LLMException as e:
                # Use the appropriate strategy depending on the type of anomaly
                code = e.status_code
                if code in RETRYABLE_CODES:
                    logger.info(f"kv cache pull blocks failed, need retry {e}")
                    time.sleep(KV_CACHE_RETRY_WAIT_SECOND)
                else:
                    logger.info(f"kv cache pull blocks failed, {e}")
                    raise e
            except (TypeError, ValueError) as e:
                logger.error(f"kv cache pull blocks input error {e}")
                raise e

        logger.error(f"kv cache pull blocks retry error {src_cache_key} {src_blocks} {dst_blocks}")
        raise RuntimeError("kv cache pull blocks failed")

    def pull_kv(self, src_blocks, tgt_blocks, prompt_cluster_id):
        # If this line is not added, the fx mode will report an error.
        # The preliminary reason is that the context is lost when multiple coroutines pull kv.
        torch.npu.set_device(f"npu:{self.local_rank}")
        for model_id, kv_cache in enumerate(self.registered_kv_caches):
            prompt_cache_key = BlocksCacheKey(
                prompt_cluster_id=prompt_cluster_id, model_id=model_id)
            self._pull_blocks(prompt_cache_key, kv_cache,
                              src_blocks, tgt_blocks)

    def register_link(self):

        if self.data_dist_config.is_prefill:
            prefill_server_groups = [self.data_dist_config.local_group]
            decode_server_groups = self.data_dist_config.global_rank_table.decode_group
        else:
            prefill_server_groups = self.data_dist_config.global_rank_table.prefill_group
            decode_server_groups = [self.data_dist_config.local_group]

        prefill_tp_size = len(prefill_server_groups[0].device_list) // self.data_dist_config.kv_producer_dp_size
        prefill_dp_size = self.data_dist_config.kv_producer_dp_size

        decode_tp_size = self.data_dist_config.kv_parallel_size
        decode_dp_size = len(decode_server_groups[0].device_list) // decode_tp_size
        decode_num = len(decode_server_groups)

        link_num = 0
        for prefill_server_group in prefill_server_groups:
            for decode_server_group in decode_server_groups:
                decode_id = 0
                for decode_device in decode_server_group.device_list:
                    d_rank = decode_device.rank_id
                    # compute p_rank with dp_size=1, and expand to dp_size>1.
                    p_rank_start = get_p_start_rank(prefill_tp_size, 1, decode_tp_size, decode_dp_size,
                                              decode_num, decode_id, d_rank)

                    pd_pairs = [(p_rank_start + dp_idx * prefill_tp_size, d_rank) for dp_idx in range(prefill_dp_size)]

                    for prefill_dp_rank, (p_rank, d_rank) in enumerate(pd_pairs):
                        prefill_cluster_id = prefill_server_group.device_list[p_rank].cluster_id
                        decode_cluster_id = decode_device.cluster_id

                        if self.multi_rank_pull_kv:
                            # first kv link
                            link_num = self._create_kv_link(prefill_server_group, decode_server_group, p_rank, d_rank,
                                                            prefill_cluster_id, decode_cluster_id, link_num)
                            # second kv link
                            second_p_rank = (p_rank + 1) % prefill_tp_size
                            second_prefill_cluster_id = prefill_server_group.device_list[second_p_rank].cluster_id

                            link_num = self._create_kv_link(prefill_server_group, decode_server_group, second_p_rank, d_rank,
                                                           second_prefill_cluster_id, decode_cluster_id, link_num)

                            prefill_cluster_id_list = [prefill_cluster_id, second_prefill_cluster_id]
                        else:
                            link_num = self._create_kv_link(prefill_server_group, decode_server_group, p_rank, d_rank,
                                                            prefill_cluster_id, decode_cluster_id, link_num)
                            prefill_cluster_id_list = [prefill_cluster_id]

                        if not self.data_dist_config.is_prefill:
                            self.registered_link_infos[(prefill_server_group.cluster_id_start, prefill_dp_rank, d_rank)] = prefill_cluster_id_list

        return self.check_register_status()

    def _create_kv_link(
        self, prefill_server_group, decode_server_group, p_rank, d_rank, prefill_cluster_id, decode_cluster_id,link_num
    ):
        p_ranktables, d_ranktables = prepare_ranktables(prefill_server_group, decode_server_group, p_rank, d_rank)

        # todo: get cluster id from server group.
        p_ser_ip = prefill_server_group.host_ip
        p_clu_id = prefill_cluster_id
        d_ser_ip = decode_server_group.host_ip
        d_clu_id = decode_cluster_id
        if self.data_dist_config.is_prefill:
            cluster_rank_infos = {
                p_rank: {prefill_cluster_id: 0, decode_cluster_id: 1}}
            comm_names = {p_rank:
                              f"{p_ser_ip}-{p_clu_id}-{d_ser_ip}-{d_clu_id}-p{p_rank}-d{d_rank}"}
            ranktables = p_ranktables
        else:
            cluster_rank_infos = {
                d_rank: {prefill_cluster_id: 0, decode_cluster_id: 1}}
            comm_names = {d_rank:
                              f"{p_ser_ip}-{p_clu_id}-{d_ser_ip}-{d_clu_id}-p{p_rank}-d{d_rank}"}

            ranktables = d_ranktables

        need_build = self._build_device_link(comm_names, cluster_rank_infos, ranktables)

        if need_build:
            link_num += 1
            if link_num >= SCHEDULER_LINK_BATCH_SIZE:
                link_num = 0
                time.sleep(SCHEDULER_LINK_INTERVAL)
        return link_num

    def _build_device_link(self, comm_names, cluster_rank_infos, rank_tables):
        if self.rank in comm_names and self.rank in cluster_rank_infos and self.rank in rank_tables:
            logger.warning(f"Current cluster id is {self.data_dist_config.cluster_id}, create link:{comm_names}")
            comm_name = comm_names[self.rank]

            cluster_rank_info = cluster_rank_infos[self.rank]
            cluster_rank_info = {int(key): value for key, value in cluster_rank_info.items()}
            rank_table = json.dumps(rank_tables[self.rank])
            comm_id = self.data_dist_engine.link(comm_name, cluster_rank_info, rank_table)
            logger.info(f"rank:{self.rank} linked {comm_name}:{comm_id}, cluster_rank_info:{cluster_rank_info}")
            # Save comm_name information
            self.rank_link_info_map[comm_name] = RankLinkInfo(comm_name, comm_id, cluster_rank_info)
            return True
        return False

    def check_register_status(self):
        status = {comm_name: False for comm_name in self.rank_link_info_map.keys()}
        ready_num = 0
        while ready_num < len(self.rank_link_info_map):
            for comm_name, rank_link_info in self.rank_link_info_map.items():
                if status[comm_name]:
                    continue
                ret = self.data_dist_engine.query_register_mem_status(rank_link_info.comm_id)
                if ret == RegisterMemStatus.OK:
                    status[comm_name] = True
                    ready_num += 1
                    logger.debug(f"rank:{self.rank} check link status success")
                    logger.debug(f"comm_id: {rank_link_info.comm_id} ret:{ret}")

                elif ret == RegisterMemStatus.FAILED:
                    logger.error(f"rank:{self.rank} check link status failed")
                    logger.debug(f"comm_id: {rank_link_info.comm_id} ret:{ret}")
                    raise RuntimeError("check kv link status failed")

                logger.warning(f"rank:{self.rank} check link status")
                logger.debug(f"comm_id: {rank_link_info.comm_id} ret:{ret}")

            if ready_num < len(self.rank_link_info_map):
                time.sleep(3)
        logger.debug(f"rank:{self.rank} build {ready_num} links successful.")
        return True


def unzip_kv_cache_dict(kv_caches: dict[str, torch.Tensor], ):
    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    _, first_kv_cache = next(iter(kv_caches.items()))
    if isinstance(first_kv_cache, tuple):
        cache_num = len(first_kv_cache)
    else:
        cache_num = 1

    flatten_kv_caches = [[] for _ in  range(cache_num)]

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        kv_cache = kv_caches[layer_name]
        if isinstance(kv_cache, tuple):
            for index, sub_cache in enumerate(kv_cache):
                flatten_kv_caches[index].append(sub_cache)
        else:
            flatten_kv_caches[0].append(kv_cache)
    return flatten_kv_caches


def unzip_kv_cache_list(kv_caches: list[torch.Tensor], ):
    first_kv_cache = kv_caches[0]
    if isinstance(first_kv_cache, tuple):
        cache_num = len(first_kv_cache)
    else:
        cache_num = 1

    flatten_kv_caches = [[] for _ in  range(cache_num)]

    for kv_cache in kv_caches:
        if isinstance(kv_cache, tuple):
            for index, sub_cache in enumerate(kv_cache):
                flatten_kv_caches[index].append(sub_cache)
        else:
            flatten_kv_caches[0].append(kv_cache)
    return flatten_kv_caches


def maybe_merge_kv_caches(flatten_kv_caches):
    # only 1 kvcache tensor with shape (2, b, s, n, d)
    if len(flatten_kv_caches) == 1 and len(flatten_kv_caches[0][0].shape) == 5 and flatten_kv_caches[0][0].shape[0] == 2:
        merged_kv_caches = [[]]
        for sub_kv_caches in flatten_kv_caches[0]:
            merged_kv_caches[0].append(sub_kv_caches[0])
            merged_kv_caches[1].append(sub_kv_caches[1])
        return merged_kv_caches
    return flatten_kv_caches

RankLinkInfo = namedtuple("RankLinkInfo", ["comm_name", "comm_id", "cluster_rank_info"])