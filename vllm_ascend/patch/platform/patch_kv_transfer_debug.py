# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""
Debug patch: KV Cache Transfer diagnostic logging.

Adds KV_DEBUG-prefixed logging at every critical point in the
PD-disaggregation KV transfer path.

Import order must be AFTER ``patch_pp_handshake_metadata`` so that
``Worker.get_kv_connector_handshake_metadata`` is already the PP-fixed
implementation when we wrap it.
"""

import logging
import os

from vllm.logger import init_logger

logger = init_logger("vllm_kv_debug")


# ---------------------------------------------------------------------------
# 1. Worker handshake metadata  — called from every GPU worker
# ---------------------------------------------------------------------------
import vllm.v1.worker.gpu_worker as _gpu_worker

_orig_worker_meta = _gpu_worker.Worker.get_kv_connector_handshake_metadata


def _debug_worker_meta(self):
    result = _orig_worker_meta(self)
    if result is not None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        tp = get_tp_group()
        pp = get_pp_group()
        for key, meta in result.items():
            logger.info(
                "KV_DEBUG worker_meta: rank=%d local_rank=%d "
                "pp=%d/%d tp=%d/%d key=%s local_ip=%s engine_id=%s",
                self.rank, self.local_rank,
                pp.rank_in_group, pp.world_size,
                tp.rank_in_group, tp.world_size,
                key, getattr(meta, "local_ip", "?"),
                getattr(meta, "engine_id", "?"),
            )
    else:
        logger.info("KV_DEBUG worker_meta: rank=%d -> None", self.rank)
    return result


_gpu_worker.Worker.get_kv_connector_handshake_metadata = _debug_worker_meta


# ---------------------------------------------------------------------------
# 2. EngineCore init — merged mapping visible to the scheduler
# ---------------------------------------------------------------------------
import vllm.v1.engine.core as _core

_orig_engine_init = _core.EngineCore.__init__


def _debug_engine_init(self, vllm_config, executor_class, log_stats,
                       executor_fail_callback=None, include_finished_set=False):
    _orig_engine_init(self, vllm_config, executor_class, log_stats,
                      executor_fail_callback, include_finished_set)
    kv_connector = getattr(self.scheduler, 'connector', None)
    if kv_connector is not None:
        meta = getattr(kv_connector, 'multi_nodes_meta_mapping', None)
        if meta:
            logger.info("KV_DEBUG engine_core: multi_nodes_meta has %d entries",
                        len(meta))
            for key in sorted(meta, key=lambda k: int(k)):
                v = meta[key]
                logger.info(
                    "KV_DEBUG engine_core: mapping[%s] host=%s engine=%s",
                    key, v.get("host", "?"), v.get("engine_id", "?"))
        else:
            logger.info(
                "KV_DEBUG engine_core: no multi_nodes_meta_mapping "
                "(expected on D side without connector)")


_core.EngineCore.__init__ = _debug_engine_init


# ---------------------------------------------------------------------------
# 3. Mooncake TransferEngine — intercept read/write at RDMA boundary
# ---------------------------------------------------------------------------
try:
    from mooncake.engine import TransferEngine as _TE

    _orig_read = _TE.batch_transfer_sync_read

    def _debug_read(self, session_id, src_addrs, dst_addrs, lengths):
        n = min(len(src_addrs), 4)
        src_few = ", ".join(f"0x{x:x}[{l}]" for x, l in zip(src_addrs[:n], lengths[:n]))
        dst_few = ", ".join(f"0x{x:x}" for x in dst_addrs[:n])
        logger.info(
            "KV_DEBUG mooncake_read: session=%s desc=%d bytes=%d "
            "src=[%s] dst=[%s]",
            session_id, len(src_addrs), sum(lengths),
            src_few, dst_few,
        )
        ret = _orig_read(self, session_id, src_addrs, dst_addrs, lengths)
        if ret != 0:
            logger.error(
                "KV_DEBUG mooncake_read FAILED: ret=%d session=%s",
                ret, session_id)
        return ret

    _TE.batch_transfer_sync_read = _debug_read

    _orig_write = _TE.batch_transfer_sync_write

    def _debug_write(self, session_id, src_addrs, dst_addrs, lengths):
        logger.info(
            "KV_DEBUG mooncake_write: session=%s desc=%d bytes=%d",
            session_id, len(src_addrs), sum(lengths))
        ret = _orig_write(self, session_id, src_addrs, dst_addrs, lengths)
        if ret != 0:
            logger.error(
                "KV_DEBUG mooncake_write FAILED: ret=%d session=%s",
                ret, session_id)
        return ret

    _TE.batch_transfer_sync_write = _debug_write
    logger.info("KV_DEBUG: mooncake TransferEngine patched")

except Exception:
    logger.info("KV_DEBUG: mooncake not available, skipping TE patch")


# ---------------------------------------------------------------------------
# 4. global_te.register_buffer — memory registration
# ---------------------------------------------------------------------------
try:
    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import \
        global_te as _gte

    _orig_reg = type(_gte).register_buffer

    def _debug_reg(self, ptrs, sizes):
        logger.info("KV_DEBUG register_memory: %d regions", len(ptrs))
        for ptr, size in zip(ptrs, sizes):
            logger.info(
                "KV_DEBUG register_memory: ptr=0x%x size=%d (%.1f MB)",
                ptr, size, size / 1024 / 1024)
        _orig_reg(self, ptrs, sizes)

    type(_gte).register_buffer = _debug_reg
    logger.info("KV_DEBUG: global_te.register_buffer patched")
except Exception:
    pass


logger.info("KV_DEBUG: platform patch loaded (PID=%d)", os.getpid())
