# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receive per-layer SFA reads into Core HiSparse's Host source and device Indexer."""

from dataclasses import dataclass
from itertools import islice

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_layout import (
    PageBuffer,
    check_page_range,
    indexer_spans,
    main_components,
    main_spans,
    page_offsets,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    HISPARSE_PROTOCOL_VERSION,
    hisparse_transfer_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import MembPullReadThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (
    CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS,
    SFAPDRD2HConsumerWorker,
    _layer_idx,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import global_memfabric_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    validate_register_region_count,
)
from vllm_ascend.hisparse.binding import hisparse_indexer_transfer_views

# A flat packed source may require two descriptors per token. Bound the Python
# working set even when one ready message covers a long cached prefix.
_READ_BATCH_SIZE = 1024


@dataclass(frozen=True)
class LayerDestination:
    host: PageBuffer
    indexer: tuple[PageBuffer, ...]


@dataclass
class HiSparseReadState:
    layer_metadata: dict[str, LayerDestination]
    dest_blocks_by_req: dict[str, tuple[list[int], list[int]]]
    block_size: int
    latent_bytes: int
    rope_bytes: int
    host_writer_rank: int
    host_writer_count: int
    dspark_context_receiver: None = None


class HiSparseLayerwiseReadThread(MembPullReadThread):
    protocol_version = HISPARSE_PROTOCOL_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finished_stages: dict[str, dict[int, frozenset[str]]] = {}

    def _check_final_layers(self, request_ids, producer_layers, pp_rank, pp_size) -> None:
        layers = frozenset(producer_layers)
        with self._lock:
            for request_id in request_ids:
                stages = self._finished_stages.setdefault(request_id, {})
                if stages.setdefault(pp_rank, layers) != layers:
                    raise ValueError("Prefill stage changed its KV layer set during a request")
                if len(stages) == pp_size:
                    del self._finished_stages[request_id]
                    if set().union(*stages.values()) != self._state.layer_metadata.keys():
                        raise ValueError("Prefill stages did not cover all HiSparse destination layers")

    def discard_requests(self, ext_ids: set[str]) -> None:
        super().discard_requests(ext_ids)
        with self._lock:
            for request_id in ext_ids:
                self._finished_stages.pop(request_id, None)

    def _decode_read_requests(self, entries):
        if any(len(entry) != 6 or type(entry[5]) is not int or entry[5] <= 0 for entry in entries):
            raise ValueError("HiSparse layerwise requires the producer's computed-token boundary")
        return entries

    def _read_spans(self, layer_name, requests, source):
        state = self._state
        destination = state.layer_metadata[layer_name]
        main_count = source["main_tensor_count"]
        bases = source["base_addrs"]
        scales = source["block_size_scale"]
        shapes = source["block_shapes"]
        capacities = source["num_logical_blocks"]
        widths = source["block_tokens"]
        if (
            main_count not in (1, 2)
            or len(bases) != main_count + len(destination.indexer)
            or bool(source["has_indexer"]) != bool(destination.indexer)
            or not all(len(part) == len(bases) for part in (source["block_len"], scales, shapes, capacities, widths))
            or any(width != state.block_size for width in widths)
            or any(base <= 0 for base in bases)
            or any(count <= 0 for count in capacities)
            or any(scale <= 0 for scale in scales)
        ):
            raise ValueError(f"Incompatible Prefill page metadata for {layer_name}")
        if main_count == 1 and len(shapes[0]) == 1 and scales[0] != 1:
            raise ValueError("Flat packed Prefill Main requires one kernel page per logical page")
        sizes = [length * scale for length, scale in zip(source["block_len"], scales, strict=True)]
        main_bases, main_sizes, main_strides = main_components(
            bases[:main_count], sizes[:main_count], shapes[:main_count], state.latent_bytes, state.rope_bytes
        )
        tokens = state.block_size
        for request_id, p_main, p_indexer, main_start, indexer_start, computed in requests:
            d_main, d_indexer = self._wait_for_dest_blocks(request_id, layer_name)
            check_page_range(p_main, main_start, len(d_main), min(capacities[:main_count]), tokens, computed)
            if p_indexer and not destination.indexer:
                raise ValueError(f"Unexpected Indexer pages for {layer_name}")
            if destination.indexer:
                check_page_range(
                    p_indexer, indexer_start, len(d_indexer), min(capacities[main_count:]), tokens, computed
                )
            for offset in page_offsets(main_start, len(p_main), state.host_writer_rank, state.host_writer_count):
                valid = min(tokens, computed - offset * tokens)
                yield from main_spans(
                    destination.host,
                    main_bases,
                    main_sizes,
                    tokens,
                    (d_main[offset], p_main[offset - main_start]),
                    valid,
                    source_strides=main_strides,
                )
            for offset, source_page in enumerate(p_indexer, indexer_start):
                valid = min(tokens, computed - offset * tokens)
                yield from indexer_spans(
                    destination.indexer,
                    bases[main_count:],
                    sizes[main_count:],
                    tokens,
                    (d_indexer[offset], source_page),
                    valid,
                )

    def _do_read_batch(self, layer_name, read_reqs, p_session=None, p_layer_meta=None, group_member_idx=0, ratio=1):
        if group_member_idx != 0 or ratio != 1 or not p_session or p_layer_meta is None:
            raise ValueError("Layerwise HiSparse requires one matching Prefill TP rank")
        spans = self._read_spans(layer_name, read_reqs, p_layer_meta[layer_name])
        while batch := list(islice(spans, _READ_BATCH_SIZE)):
            local, remote, lengths = map(list, zip(*batch, strict=True))
            if self.engine.batch_transfer_sync_read(p_session, local, remote, lengths) != 0:
                raise RuntimeError(f"HiSparse layerwise read failed for {layer_name}")


class HiSparseLayerwiseConsumerWorker(SFAPDRD2HConsumerWorker):
    def register_kv_caches(self, kv_caches):
        config = self.kv_cache_config
        main_id, indexer_id = hisparse_transfer_group_ids(config)
        main = config.kv_cache_groups[main_id]
        indexer = config.kv_cache_groups[indexer_id]
        context = self.vllm_config.compilation_config.static_forward_context
        tokens = main.kv_cache_spec.block_size
        if tokens != indexer.kv_cache_spec.block_size:
            raise ValueError("Layerwise HiSparse requires equal Main and Indexer page sizes")
        indexer_by_layer = {_layer_idx(name): name for name in indexer.layer_names}
        if len(indexer_by_layer) != len(indexer.layer_names):
            raise ValueError("Layerwise HiSparse requires at most one Indexer for each layer")
        destinations = {}
        transfer_views = {}

        def pages(name, views, capacity, device):
            if not views or any(view.device.type != device for view in views):
                raise ValueError(f"Layerwise HiSparse expects {name} on {device}")
            transfer_views[name] = views
            return tuple(PageBuffer.from_tensor(view, capacity) for view in views)

        for name in main.layer_names:
            cache = kv_caches[name]
            (host,) = pages(
                name, cache if isinstance(cache, (tuple, list)) else (cache,), config.hisparse_host_num_blocks, "cpu"
            )
            indexer_pages = ()
            index_name = indexer_by_layer.pop(_layer_idx(name), None)
            if index_name is not None:
                cache = kv_caches[index_name]
                parts = cache if isinstance(cache, (tuple, list)) else (cache,)
                views = hisparse_indexer_transfer_views(parts, context[index_name])
                indexer_pages = pages(index_name, views, config.num_blocks, "npu")
            destinations[name] = LayerDestination(host, indexer_pages)
        if indexer_by_layer:
            raise ValueError("Unmatched Indexer layers in the HiSparse transfer groups")

        # Every TP rank fills its private pool. If the backend actually allocated
        # one shared Host pool, assign each absolute Main page to exactly one rank.
        shared = config.hisparse_shared_host_pool
        model = self.vllm_config.model_config.hf_text_config
        group_spec = main.kv_cache_spec
        first_layer = main.layer_names[0]
        spec = getattr(group_spec, "kv_cache_specs", {}).get(first_layer, group_spec)
        element_size = spec.dtype.itemsize
        state = HiSparseReadState(
            destinations,
            self._dest_blocks_by_req,
            tokens,
            tokens * model.kv_lora_rank * element_size,
            tokens * model.qk_rope_head_dim * element_size,
            self.tp_rank if shared else 0,
            self.tp_size if shared else 1,
        )
        regions = collect_storage_merged_register_regions(transfer_views)
        validate_register_region_count(regions)
        engine = self._ensure_engine()
        global_memfabric_te.register_buffer(regions.ptrs, regions.lengths)
        thread = HiSparseLayerwiseReadThread(self.tp_rank, self.side_channel_port, engine, state)
        self._mf_read_thread = thread
        thread.start()
        if not thread.ready_event.wait(CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS) or thread.startup_error:
            thread.stop()
            raise RuntimeError("Layerwise HiSparse read thread failed to start") from thread.startup_error

    def shutdown(self) -> None:
        super().shutdown()
        thread = self._mf_read_thread
        if thread is not None and thread.is_alive():
            raise RuntimeError("Cannot release HiSparse Host pages while a layer read is in flight")
