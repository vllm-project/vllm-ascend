"""Capture per-request KV tensors for offline PCA/quantization calibration."""

from collections.abc import Mapping, Set
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.states import RequestState

KV_DUMP_IO_WORKERS = 4


class KVCacheDumper:
    """Snapshot live cache data on CPU before writing it on background threads."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.executor = ThreadPoolExecutor(max_workers=KV_DUMP_IO_WORKERS, thread_name_prefix="kvtc-dump")

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    @staticmethod
    def _save_snapshot(snapshot: torch.Tensor, path: Path) -> None:
        try:
            torch.save(snapshot, path)
        except Exception:
            logger.exception("[KVTC DUMP] Failed to write calibration sample %s", path)

    def dump_finished(
        self,
        request_ids: Set[str],
        req_states: "RequestState",
        block_tables: "BlockTables",
        kv_cache_config: "KVCacheConfig",
        attention_layers: Mapping[str, "AttentionLayerBase"],
    ) -> None:
        """Use live V2 state before the runner frees or overwrites request rows."""
        live = [
            (req_id, req_states.req_id_to_index[req_id])
            for req_id in request_ids
            if req_id in req_states.req_id_to_index
        ]
        if not live:
            return
        (group,) = kv_cache_config.kv_cache_groups
        names = sorted(group.layer_names, key=extract_layer_index)
        caches = [attention_layers[name].kv_cache for name in names]
        block_size = caches[0][0].shape[1]
        block_table = block_tables.block_tables[0].gpu
        # Device counts are exact after sampling, including speculative rejection.
        # Read them together only when the calibration dump is enabled.
        counts = req_states.num_computed_tokens.gpu[[slot for _, slot in live]].cpu().tolist()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pp_group().rank_in_group
        output_dir = self.output_dir / f"tp_{tp_rank}_pp_{pp_rank}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for (request_id, slot), num_tokens in zip(live, counts):
            num_blocks = cdiv(num_tokens, block_size)
            if num_tokens <= 0 or num_blocks > block_tables.num_blocks.np[0, slot]:
                continue
            block_ids = block_table[slot, :num_blocks].long()
            self._dump(request_id, block_ids, num_tokens, caches, output_dir)

    def _dump(self, request_id, block_ids, num_tokens, kv_caches, output_dir) -> None:
        # Use the same filesystem-safe sample ID on every rank.
        sample_id = sha256(request_id.encode("utf-8")).hexdigest()

        for layer_index, layer_cache in enumerate(kv_caches):
            for kind, cache in zip(("K", "V"), layer_cache):
                blocks = cache[block_ids]
                tokens = blocks.reshape(-1, blocks.shape[2], blocks.shape[3])[:num_tokens]
                # Complete the device-to-CPU copy before the runner can reuse
                # the live cache blocks; only file I/O runs in the background.
                snapshot = tokens.to(device="cpu", copy=True)
                self.executor.submit(
                    self._save_snapshot, snapshot, output_dir / f"{sample_id}-{kind}-layer_{layer_index}.bin"
                )

        logger.info(
            "[KVTC DUMP] Queued request %s: tokens=%d directory=%s",
            request_id,
            num_tokens,
            output_dir,
        )
