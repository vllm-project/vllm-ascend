# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

import threading
import time
from typing import List, Tuple

import torch
from ems import (CcConfig, CcKvOption, ContextCaching, Ems, EmsConfig,
                 EmsException, KvBufferWrapper)
from omni.adaptors.vllm.ems.ems_env import EmsEnv
from omni.adaptors.vllm.ems.ems_metrics import MetricsType, collect_metric
from omni.adaptors.vllm.ems.ems_utils import (EmsKeyGenerator,
                                              EmsKVCacheManager,
                                              cal_hash_blocks)
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput

SLEEP_SECONDS = 30
MIN_BLOCK_LEN = 24


class EmsAdapter:

    def __init__(self, vllm_config: VllmConfig):
        rank = vllm_config.parallel_config.rank
        local_rank = get_tp_group().local_rank
        logger.info(
            f"[EMS] Init ems adapter, local_rank: {local_rank}, rank: {rank}, vllm_config: {vllm_config}"
        )
        self.vllm_config = vllm_config
        self.rank = rank

        self.is_mla = self._is_mla(vllm_config)
        tp_rank = rank if not self.is_mla else 0
        self.key_generator = EmsKeyGenerator(
            self.vllm_config.parallel_config.pipeline_parallel_size,
            self.vllm_config.parallel_config.tensor_parallel_size, tp_rank)
        self.kvcache_manager = EmsKVCacheManager()
        self.status_checker = EmsStatusChecker()
        self.context_caching = self._init_ems_cc(local_rank, rank)
        self.save_events = {}

    def bind_kvcaches(self, kv_caches: List[torch.Tensor]):
        self.kvcache_manager.initialize_kvcache(kv_caches)

    def _check_params(self, key_list, value_list) -> bool:
        if not self.context_caching:
            logger.error("[EMS] CC is not initialed.")
            return False

        if not self.status_checker.get_status():
            logger.error("[EMS] CC status is unhealthy")
            return False

        if len(key_list) == 0 or len(key_list) != len(value_list):
            logger.error(
                f"[EMS] Key/value list length is invalid, key len is {len(key_list)}, value len is {len(value_list)}"
            )
            return False

        return True

    @collect_metric(MetricsType.LOAD)
    def load(self, info_load_reqs) -> List[int]:
        self.sync_save_event()

        if not info_load_reqs:
            return []

        result = []
        load_events = {}

        for req_id, hash_blocks, block_ids in info_load_reqs:
            if not hash_blocks:
                load_events[req_id] = (None, 0)
                continue

            keys_load = self._cal_keys_by_hashes(hash_blocks)
            values_load = self._cal_values(block_ids)

            option = CcKvOption(write_rcache=EmsEnv.enable_write_rcache,
                                read_local_only=EmsEnv.enable_read_local_only)
            submit_time = time.perf_counter()
            if not self._check_params(keys_load, values_load):
                logger.error(
                    f"[EMS] req {req_id} async load check params fail.")
                load_events[req_id] = (None, submit_time)
                continue
            if len(keys_load) < MIN_BLOCK_LEN:
                logger.debug(
                    f"[EMS] req {req_id} load blocks num is less than {MIN_BLOCK_LEN}."
                )
                load_events[req_id] = (None, submit_time)
                continue
            try:
                load_future = self.context_caching.async_load(
                    option, keys_load, values_load)
                load_events[req_id] = (load_future, submit_time)
            except EmsException as e:
                self.status_checker.set_status(False)
                logger.error(f"[EMS] req {req_id} load failed, error: {e}.")
                load_events[req_id] = (None, submit_time)

        for req_id, (load_future, submit_time) in load_events.items():
            num_success_blocks = 0
            if load_future is None:
                result.append(num_success_blocks)
                continue

            try:
                load_result = load_future.result()
                logger.info(
                    f"[EMS] req {req_id} load result: {load_result}, "
                    f"cost time: {round(time.perf_counter() - submit_time, 6)} s."
                )
                num_success_blocks = load_result.success
            except EmsException as e:
                self.status_checker.set_status(False)
                logger.error(
                    f"[EMS] req {req_id} load result failed, error: {e}.")
            result.append(num_success_blocks)

        return result

    @collect_metric(MetricsType.ASYNC_SAVE)
    def async_save(self, scheduler_output: "SchedulerOutput") -> None:
        if not scheduler_output.scheduled_new_reqs:
            return

        block_size = self.vllm_config.cache_config.block_size

        for new_req in scheduler_output.scheduled_new_reqs:
            # v1 调度不区分prefill和decode，会把token_budget填满，这会导致请求被截断，一次step只计算部分token
            num_total_blocks = (
                scheduler_output.num_scheduled_tokens[new_req.req_id] +
                new_req.num_computed_tokens - 1) // block_size
            num_computed_blocks = new_req.num_computed_tokens // block_size

            logger.info(
                f"[EMS] Save req {new_req.req_id}, block_ids: {new_req.block_ids}, "
                f"prompt len: {len(new_req.prompt_token_ids)}, "
                f"num_scheduled_tokens: {scheduler_output.num_scheduled_tokens[new_req.req_id]}, "
                f"num_computed_tokens: {new_req.num_computed_tokens}, "
                f"num_total_blocks: {num_total_blocks}, num_computed_blocks: {num_computed_blocks}"
            )

            if num_computed_blocks >= num_total_blocks:
                continue

            keys_total = self._cal_keys(new_req.prompt_token_ids, block_size)
            keys_save_total = keys_total[num_computed_blocks:num_total_blocks]
            values_save_total = self._cal_values(
                new_req.block_ids[0][num_computed_blocks:num_total_blocks])

            keys_save, values_save = self._cal_save_kv(keys_save_total,
                                                       values_save_total)

            if not keys_save:
                return

            option = CcKvOption(write_rcache=EmsEnv.enable_write_rcache,
                                read_local_only=EmsEnv.enable_read_local_only)
            if not self._check_params(keys_save, values_save):
                logger.error(
                    f"[EMS] req {new_req.req_id} async save check params fail."
                )
                continue
            try:
                submit_time = time.perf_counter()
                self.save_events[new_req.req_id] = \
                    (self.context_caching.async_save(option, keys_save, values_save), submit_time)

            except EmsException as e:
                self.status_checker.set_status(False)
                logger.error(
                    f"[EMS] req {new_req.req_id} async save failed, error: {e}."
                )

    @collect_metric(MetricsType.SYNC_SAVE_EVENT)
    def sync_save_event(self) -> None:
        if not self.save_events:
            return

        for req_id, (save_future, submit_time) in self.save_events.items():
            try:
                save_result = save_future.result()
                logger.info(
                    f"[EMS] req {req_id} save result: {save_result},"
                    f"cost time: {round(time.perf_counter() - submit_time, 6)} s."
                )
            except EmsException as e:
                self.status_checker.set_status(False)
                logger.error(
                    f"[EMS] req {req_id} save result failed, error {e}.")

        self.save_events.clear()

    def _is_mla(self, vllm_config: VllmConfig):
        return "deepseek" in vllm_config.model_config.hf_config.model_type

    def _need_save(self):
        # mla, tp组内所有rank产生的kvcache一样，只保存rank 0的kvcache
        return not self.is_mla or self.rank == 0

    def _cal_save_kv(self, keys_save_total: List[str], values_save_total: List[List[KvBufferWrapper]]) \
            -> Tuple[List[str], List[List[KvBufferWrapper]]]:
        if not self.is_mla:
            return keys_save_total, values_save_total

        if EmsEnv.ems_store_local:
            n_parts = min(
                self.vllm_config.parallel_config.tensor_parallel_size, 16)
            cur_part = (self.rank -
                        n_parts) if self.rank >= n_parts else self.rank
        else:
            n_parts = self.vllm_config.parallel_config.tensor_parallel_size
            cur_part = self.rank
        return keys_save_total[cur_part::n_parts], values_save_total[
            cur_part::n_parts]

    def _init_ems_cc(self, local_rank: int, rank: int) -> "ContextCaching":
        cc = None

        cc_config = CcConfig(model_id=EmsEnv.model_id,
                             rank_id=rank,
                             device_id=local_rank)
        ems_config = EmsConfig(access_id=EmsEnv.access_id,
                               access_key=EmsEnv.access_key,
                               cc_config=cc_config)

        try:
            Ems.init(ems_config)
            cc = Ems.get_cc()
        except EmsException as e:
            logger.error(f"[EMS] Init ems failed, error: {e}.")

        return cc

    @collect_metric(MetricsType.CAL_KEYS)
    def _cal_keys(self, token_ids: List[int], block_size: int) -> List[str]:
        hash_blocks = cal_hash_blocks(token_ids, block_size)
        return self._cal_keys_by_hashes(hash_blocks)

    def _cal_keys_by_hashes(self, hash_blocks: List[int]) -> List[str]:
        return [
            self.key_generator.gen_key(block_hash)
            for block_hash in hash_blocks
        ]

    @collect_metric(MetricsType.CAL_VALUES)
    def _cal_values(self, block_ids: List[int]) -> List[List[KvBufferWrapper]]:
        return [
            self.kvcache_manager.get_block_kv_buffer(block_id)
            for block_id in block_ids
        ]


class EmsStatusChecker:

    def __init__(self):
        self._ems_ok = False
        self._start_cc_health_check()

    def get_status(self):
        return self._ems_ok

    def set_status(self, status: bool):
        self._ems_ok = status

    def _check_and_update(self):
        """检查EMS健康状态"""
        while True:
            time.sleep(SLEEP_SECONDS)
            try:
                self._ems_ok = Ems.check_health()
                if self._ems_ok:
                    logger.debug("EMS health status is ok.")
                else:
                    logger.info("EMS health status is abnormal.")
            except Exception as e:
                self._ems_ok = False
                logger.exception(f"EMS health status exception, {e}")

    def _start_cc_health_check(self):
        """启动一个新线程来执行定时任务"""
        health_check_thread = threading.Thread(target=self._check_and_update,
                                               name="cc-health")
        health_check_thread.daemon = True
        health_check_thread.start()
        logger.info("Start EMS health check")
