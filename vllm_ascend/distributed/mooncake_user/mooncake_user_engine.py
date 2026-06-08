# Standard
import math
import threading
from typing import TYPE_CHECKING, Generator, Optional

import numpy as np
# Third Party
import torch
from vllm.config import VllmConfig
from vllm.transformers_utils.configs.hstu_config import InferenceHSTUConfig
from vllm.utils import get_kv_cache_torch_dtype, logger

from vllm_ascend.distributed.mooncake_user.backend import (MooncakeBackend,
                                                           backend_map)
from vllm_ascend.distributed.mooncake_user.user_config_data import (
    LasyerMultiBlockReqMeta, LayerMooncakeUserKey, MooncakeConnectorMetadata,
    MooncakeEngineMetadata, MooncakeUserKey)
from vllm_ascend.distributed.mooncake_user.user_kv_transfer import (
    KVCacheStoreLayerRecvingThread, KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread, KVCacheStoreSendingThread, KVTransferThread,
    get_start_end)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext


class MooncakeEngine:
    #The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        config: InferenceHSTUConfig = model_config.hf_config.hstu_config
        self.use_mla = False
        self.use_layerwise = use_layerwize
        self.tp_rank = parallel_config.rank
        self.tp_size = parallel_config.tensor_parallel_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", False)
        self.register_buffer = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "register_buffer", False)
        self.block_size = vllm_config.cache_config.block_size
        self.current_layer = 0
        # self.num_layers = model_config.get_num_layers(parallel_config)
        # num_kv_head = model_config.get_num_kv_heads(parallel_config)
        # num_kv_head = self.num_layers
        # head_size = model_config.get_head_size()
        self.num_layers = config.num_layers
        num_kv_head = config.num_heads
        head_size = config.head_dim
        kv_dtype = get_kv_cache_torch_dtype(
            vllm_config.cache_config.cache_dtype, model_config.dtype)
        self.hidden_dim_size = num_kv_head * head_size
        if self.use_mla:
            kv_shape = (self.num_layers, 1, self.block_size, 1, head_size)
        else:
            kv_shape = (self.num_layers, 2, self.block_size, num_kv_head,
                        head_size)
        self.metadata = MooncakeEngineMetadata(
            model_config.model,
            parallel_config.world_size,
            parallel_config.rank,
            kv_dtype,
            kv_shape,
            self.block_size,
            self.use_mla,
        )

        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "backend", "mooncake")
        backend_cls = backend_map.get(self.backend.lower())
        if backend_cls is None:
            raise ValueError(
                f"Unsupported backend: {self.backend}. Supported backends are: {list(backend_map.keys())}"
            )
        self.m_store = backend_cls(parallel_config)

        self.kv_send_thread: Optional[KVTransferThread] = None
        self.kv_recv_thread: Optional[KVTransferThread] = None
        if isinstance(
                self.m_store,
                MooncakeBackend) and self.m_store.config.use_ascend_direct:
            self.use_ascend_direct = True
        else:
            self.use_ascend_direct = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = first_kv_cache.shape[0]
        kv_elem_size = first_kv_cache.element_size()
        block_rank = 3  # [block_size, kv_heads, head_dim]
        block_shape = first_kv_cache.shape[-block_rank:]
        self.block_len = [kv_elem_size * math.prod(block_shape)]
        logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                    block_shape)

        logger.info("Registering KV_Caches. shape %s", first_kv_cache.shape)

        self.kv_caches = kv_caches
        self.kv_caches_base_addr = []
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            cache_list = cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                self.kv_caches_base_addr.append(base_addr)
                if self.register_buffer:
                    region_len = self.num_blocks * self.block_len[0]
                    self._register(base_addr, region_len)

        if self.use_layerwise:
            self.get_event = threading.Event()
            if self.kv_role in ['kv_producer', 'kv_both']:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.tp_rank, self.tp_size, self.m_store,
                    self.kv_caches_base_addr, self.metadata, self.block_len,
                    self.block_size, ready_event_sending, self.num_layers)
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.tp_rank, self.tp_size, self.m_store,
                self.kv_caches_base_addr, self.metadata, self.block_len,
                self.block_size, ready_event, self.get_event)
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ['kv_producer', 'kv_both']:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.tp_rank, self.tp_size, self.m_store,
                    self.kv_caches_base_addr, self.metadata, self.block_len,
                    self.block_size, ready_event_sending)
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.tp_rank, self.tp_size, self.m_store,
                    self.kv_caches_base_addr, self.metadata, self.block_len,
                    self.block_size, ready_event)
                self.kv_recv_thread.start()
                ready_event.wait()

    def _register(self, ptr, length):
        logger.debug(
            "Registering KV cache: ptr=0x%x, length=%d, num_blocks=%d, "
            "block_lens=%s", ptr, length, self.num_blocks, self.block_len)
        try:
            self.m_store.register_buffer(ptr, length)
        except Exception as e:
            raise RuntimeError(
                f"Mooncake memory registration failed. Error is: {e}")

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  #load =0
                continue
            tokens = request.token_ids
            req_id = request.req_id
            uid = request.uid

            tokens = tokens[:request.load_spec.mooncake_cached_tokens]

            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(
                    req_id,
                    uid,
                    tokens,
                    request.block_ids,
                )
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if self.load_async:
                    self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                        req_id,
                        uid,
                        tokens,
                        request.block_ids,
                    )
                else:
                    if self.use_ascend_direct:
                        addr_list = []
                        size_list = []
                        key_list = []
                        blockIds = []

                        starts, ends = get_start_end(len(tokens),
                                                     self.block_size)
                        key = MooncakeUserKey(
                            uid=uid,
                            model_name=self.metadata.model_name,
                            world_size=self.metadata.world_size,
                            worker_id=self.metadata.worker_id,
                            value_type="kv_cache")

                        key_list += [key.to_string()]
                        for start, end in zip(starts, ends):
                            addr, size, block_id = self.prepare_value(
                                start, end, request.block_ids)
                            addr_list += addr
                            size_list += size
                            blockIds.append(block_id)
                        self.m_store.get_batch(key_list, [addr_list],
                                               [size_list], blockIds)
                    else:
                        pass

    def prepare_value(self, start: int, end: int, block_ids: list[int]):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            block_len = (self.block_len[index % 2]
                         if self.use_mla else self.block_len[0])

            addr = base_addr + block_id * block_len
            length = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(length)
        return addr_list, size_list, block_id

    def wait_for_layer_load(self) -> None:
        """MooncakeConnector does not do layerwise saving."""
        # breakpoint()
        layerwise_retrievers = getattr(self, "layerwise_retrievers", None)
        if layerwise_retrievers is None:
            return

        for layerwise_retriever in layerwise_retrievers:
            next(layerwise_retriever)
            # ret_token_mask = next(layerwise_retriever)
            # if self.current_layer == self.num_layers - 1:
            #     assert ret_token_mask is not None
            #     num_retrieved_tokens = ret_token_mask.sum().item()
            #     logger.info(f"Retrieved {num_retrieved_tokens} tokens")

    def save_kv_layer(self,
                      connector_metadata: MooncakeConnectorMetadata) -> None:
        """MooncakeConnector does not save explicitly."""
        if self.current_layer == 0:
            self.layerwise_storers = []
            for request in connector_metadata.requests:
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                req_id = request.req_id
                uid = request.uid
                assert isinstance(token_ids, torch.Tensor)
                assert token_ids.is_cpu

                layerwise_storer = self.store_layer(
                    req_id,
                    uid,
                    token_ids,
                    block_ids=request.block_ids,
                )
                self.layerwise_storers.append(layerwise_storer)

        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception:
                raise
            self.current_layer = self.current_layer + 1

    def save_kv_cache_by_layer(self, layer_id: int, kv_cache: torch.Tensor,
                               attn_metadata: "AttentionMetadata",
                               connector_metadata: MooncakeConnectorMetadata,
                               **kwargs) -> None:
        uids = attn_metadata.additional_metadata["uids"]
        if uids is None or len(uids) == 0:
            logger.warning("uids in attn_metadata is None.")
            return

        for idx, uid in enumerate(uids):
            for request in connector_metadata.requests:
                if uid != request.uid:
                    continue

                # kv_cache_pinned = kv_cache.detach().to(device='cpu', non_blocking=True)
                layer_key = LayerMooncakeUserKey(
                    uid=uid,
                    model_name=self.metadata.model_name,
                    world_size=self.metadata.world_size,
                    worker_id=self.metadata.worker_id,
                    value_type="kv_cache",
                    layer_id=layer_id)
                addrs = []
                sizes = []
                starts, ends = get_start_end(len(request.token_ids),
                                             self.block_size)
                min_length = min(len(starts), len(request.block_ids))
                starts = starts[:min_length]
                request.block_ids = request.block_ids[:min_length]
                logger.info(f"block_size:{self.block_size}")
                assert self.kv_send_thread is not None
                for start, end in zip(starts, ends):
                    addr, size = self.kv_send_thread.prepare_value_layer(
                        start, end, request.block_ids, layer_id)
                    addrs += addr
                    sizes += size
                logger.info(
                    f"len of request_token_ids: {len(request.token_ids)}")
                self.m_store.put_batch([layer_key.to_string()], [addrs],
                                       [sizes], request.block_ids)

    def start_load_kv_cache_by_layer(
            self, forward_context: "ForwardContext", layer_name: str,
            layer_id: int, connector_metadata: MooncakeConnectorMetadata,
            **kwargs) -> None:

        attn_metadata = forward_context.attn_metadata[layer_name]
        if connector_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return
        uids = attn_metadata.additional_metadata["uids"]
        if uids is None or len(uids) == 0:
            logger.warning(
                "In connector.start_load_kv_by_layer, but user id is None.")
            return
        # Load the KV for each request each layer
        for idx, uid in enumerate(uids):
            for request in connector_metadata.requests:
                if uid != request.uid:
                    continue

                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue

                layer_key = LayerMooncakeUserKey(
                    uid=uid,
                    model_name=self.metadata.model_name,
                    world_size=self.metadata.world_size,
                    worker_id=self.metadata.worker_id,
                    value_type="kv_cache",
                    layer_id=layer_id)
                addrs = []
                sizes = []
                starts, ends = get_start_end(len(request.token_ids),
                                             self.block_size)
                assert self.kv_recv_thread is not None
                for start, end in zip(starts, ends):
                    addr, size = self.kv_recv_thread.prepare_value_layer(
                        start, end, request.block_ids, layer_id)
                    addrs += addr
                    sizes += size

                self.m_store.get_batch([layer_key.to_string()], [addrs],
                                       [sizes], request.block_ids)

    def wait_for_save(self, connector_metadata: MooncakeConnectorMetadata):
        """MooncakeConnector does not save explicitly."""
        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            req_id = request.req_id
            uid = request.uid
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu

            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                req_id,
                uid,
                token_ids,
                request.block_ids,
            )

    def retrieve_layer(
        self,
        req_id: str,
        uid: int,
        tokens: torch.Tensor,
        block_ids: list[int],
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. 
        """
        #breakpoint()
        num_tokens = len(tokens)
        starts, ends = get_start_end(num_tokens, self.block_size)
        keys = MooncakeUserKey(uid=uid,
                               model_name=self.metadata.model_name,
                               world_size=self.metadata.world_size,
                               worker_id=self.metadata.worker_id,
                               value_type="kv_cache").split_layers(
                                   self.num_layers)
        # starts = []
        # ends = []
        first_flag = True
        # for start, end, key in self.token_database.process_tokens(
        #         tokens, mask):
        #     keys_multi_layer = key.split_layers(self.num_layers)
        #     starts.append(start)
        #     ends.append(end)
        #     keys.append(keys_multi_layer)

        if keys:
            # Transpose the keys into layer major format
            # keys = [list(row) for row in zip(*keys)]  # [num_layer,block_num]
            for layer_id, layer_key in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  #try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LasyerMultiBlockReqMeta(req_id, layer_key, starts,
                                                   ends, block_ids, layer_id)
                self.kv_recv_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta)  # type: ignore[union-attr, call-arg, arg-type]
                first_flag = False
                yield None
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

    def store_layer(
        self,
        req_id: str,
        uid: int,
        tokens: torch.Tensor,
        block_ids: list[int],
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """

        num_stored_tokens = len(tokens)

        starts, ends = get_start_end(num_stored_tokens, self.block_size)
        keys = MooncakeUserKey(uid=uid,
                               model_name=self.metadata.model_name,
                               world_size=self.metadata.world_size,
                               worker_id=self.metadata.worker_id,
                               value_type="kv_cache").split_layers(
                                   self.num_layers)
        # breakpoint()
        # starts = []
        # ends = []
        # keys = []
        # for start, end, key in self.token_database.process_tokens(
        #         tokens, mask):
        #     keys_multi_layer = key.split_layers(self.num_layers)
        #     starts.append(start)
        #     ends.append(end)
        #     keys.append(keys_multi_layer)  #[block_num,layer_num]
        if keys:
            # keys = [list(row) for row in zip(*keys)]  #[layer_num,block_num]
            for layer_id, layer_key in enumerate(keys):
                req_meta = LasyerMultiBlockReqMeta(req_id, layer_key, starts,
                                                   ends, block_ids, layer_id)
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta)  # type: ignore[union-attr, call-arg, arg-type]
                yield
        else:
            for layer_id in range(self.num_layers):
                yield

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role in ['kv_producer', 'kv_both'] else set())

        done_recving = (
            self.kv_recv_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.load_async else set())

        # logger.debug(
        #     "Number of completed KV cache send requests: %d, receive "
        #     "requests: %d, tp_rank:%d", len(done_sending), len(done_recving),
        #     self.tp_rank)
        return done_sending, done_recving

    def wait_layer_transfer_finish(self):
        # time.sleep(0.5)
        pass

    def lookup_scheduler(
        self,
        uid: int,
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            key = MooncakeUserKey(
                uid=uid,
                model_name=self.metadata.model_name,
                world_size=self.metadata.world_size,
                worker_id=self.metadata.worker_id,
                value_type='token_id',
            )
            if not self.m_store.exists(key.to_string()):
                return 0
            buffer = self.m_store.get_buffer(key.to_string())
            history_token_ids = np.frombuffer(buffer, dtype=np.int64).tolist()
            return len(history_token_ids)
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return -1
        return -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        self.m_store.close()
