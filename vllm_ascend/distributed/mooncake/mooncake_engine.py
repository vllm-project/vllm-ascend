# Standard
import math
import threading
import time
from typing import Generator, List, Optional, Union

# Third Party
import torch
from vllm.config import VllmConfig
from vllm.utils import get_kv_cache_torch_dtype, logger

from vllm_ascend.distributed.mooncake.config_data import (
    ChunkedTokenDatabase, LasyerMultiBlockReqMeta, MooncakeConnectorMetadata,
    MooncakeEngineMetadata)
from vllm_ascend.distributed.mooncake.kv_transfer import (
    KVCacheStoreLayerRecvingThread, KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread, KVCacheStoreSendingThread, KVTransferThread)
from vllm_ascend.distributed.mooncake.mooncake_store import Mooncakestore


class MooncakeEngine:
    #The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.use_mla = False
        if (hasattr(model_config, "use_mla")
                and isinstance(model_config.use_mla, bool)
                and model_config.use_mla):
            self.use_mla = True
        self.use_layerwise = use_layerwize
        self.tp_rank = parallel_config.rank
        self.tp_size = parallel_config.tensor_parallel_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.block_size = vllm_config.cache_config.block_size
        self.current_layer = 0
        # self.use_mla = first_kv_cache_tuple[0].size(
        #     -1) != first_kv_cache_tuple[1].size(-1)
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.block_size = vllm_config.cache_config.block_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
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

        self.token_database = ChunkedTokenDatabase(self.metadata)

        self.m_store = Mooncakestore(parallel_config)

        self.kv_send_thread: Optional[KVTransferThread] = None
        self.kv_recv_thread: Optional[KVTransferThread] = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        if self.use_mla:
            # MLA case.[num_block, block_size, 1, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, latent_dim]
            block_shape_norm = first_kv_cache_tuple[0].shape[-block_rank:]
            block_shape_pe = first_kv_cache_tuple[1].shape[-block_rank:]
            self.block_len = [
                first_kv_cache[0].element_size() * math.prod(block_shape_norm),
                first_kv_cache[1].element_size() * math.prod(block_shape_pe)
            ]
            logger.info(
                "num_blocks: %s, block_shape_norm: %s, block_shape_pe: %s",
                self.num_blocks, block_shape_norm, block_shape_pe)
        else:
            # [num_block, block_size, num_head, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            kv_elem_size = first_kv_cache.element_size()
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            self.block_len = [kv_elem_size * math.prod(block_shape)]
            logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                        block_shape)

        logger.info("Registering KV_Caches. use_mla: %s, shape %s",
                    self.use_mla, first_kv_cache.shape)

        self.kv_caches = kv_caches
        self.m_store.set_kv_caches(kv_caches.values())
        self.kv_caches_base_addr = []
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            if self.use_mla:
                for i, cache in enumerate(cache_or_caches, 0):
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)
            else:
                cache_list = [cache_or_caches
                              ] if self.use_mla else cache_or_caches
                for cache in cache_list:
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)

        if self.use_layerwise:
            self.get_event = threading.Event()
            if self.kv_role == 'kv_producer':
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.tp_rank, self.tp_size, self.m_store,
                    self.kv_caches_base_addr, self.token_database,
                    self.block_len, self.block_size, ready_event_sending,
                    self.num_layers)
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.tp_rank, self.tp_size, self.m_store,
                self.kv_caches_base_addr, self.token_database, self.block_len,
                self.block_size, ready_event, self.get_event)
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role == 'kv_producer':
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.tp_rank, self.tp_size, self.m_store,
                    self.kv_caches_base_addr, self.token_database,
                    self.block_len, self.block_size, ready_event_sending)
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreRecvingThread(
                self.tp_rank, self.tp_size, self.m_store,
                self.kv_caches_base_addr, self.token_database, self.block_len,
                self.block_size, ready_event)
            self.kv_recv_thread.start()
            ready_event.wait()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  #load =0
                continue
            tokens = request.token_ids
            req_id = request.req_id
            if (load_spec.mooncake_cached_tokens % self.block_size
                    != 0) and (load_spec.mooncake_cached_tokens
                               == tokens.shape[0] - 1):
                tokens = tokens[:request.load_spec.mooncake_cached_tokens + 1]
            else:
                tokens = tokens[:request.load_spec.mooncake_cached_tokens]
            masked_token_count = (request.load_spec.vllm_cached_tokens //
                                  self.block_size * self.block_size)
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            token_mask[:masked_token_count] = False
            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(
                    req_id,
                    tokens,
                    request.block_ids,
                    token_mask,
                )
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    req_id,
                    tokens,
                    request.block_ids,
                    token_mask,
                )

    def wait_for_layer_load(self) -> None:
        """MooncakeConnector does not do layerwise saving."""
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

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
                assert isinstance(token_ids, torch.Tensor)
                assert token_ids.is_cpu

                # TODO: whether need to remov saveThread
                # no lookup, skipmask
                skip_leading_tokens = max(
                    self.lookup(token_ids, self.use_layerwise),
                    save_spec.skip_leading_tokens,
                )
                if skip_leading_tokens == len(token_ids):
                    if request.is_last_chunk:
                        self.kv_send_thread.set_finished_request(  # type: ignore[union-attr]
                            req_id)
                    continue  # skip this request

                skip_leading_tokens = (skip_leading_tokens // self.block_size *
                                       self.block_size)

                store_mask = torch.ones_like(token_ids, dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False
                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                layerwise_storer = self.store_layer(
                    req_id,
                    token_ids,
                    mask=store_mask,
                    block_ids=request.block_ids,
                )
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception:
                raise
            self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: MooncakeConnectorMetadata):
        """MooncakeConnector does not save explicitly."""
        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            req_id = request.req_id
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu

            skip_leading_tokens = max(
                self.lookup(token_ids, self.use_layerwise),
                save_spec.skip_leading_tokens,
            )
            if skip_leading_tokens == len(token_ids):
                if request.is_last_chunk:
                    self.kv_send_thread.set_finished_request(  # type: ignore[union-attr]
                        req_id)
                continue  # skip this request

            skip_leading_tokens = (skip_leading_tokens // self.block_size *
                                   self.block_size)

            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                req_id,
                token_ids,
                request.block_ids,
                store_mask,
                request.is_last_chunk,
            )

    def retrieve_layer(
        self,
        req_id: str,
        tokens: torch.Tensor,
        block_ids: list[int],
        mask: Optional[torch.Tensor] = None,
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

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        first_flag = True
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys = [list(row) for row in zip(*keys)]  # [num_layer,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  #try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LasyerMultiBlockReqMeta(req_id, keys_multi_chunk,
                                                   starts, ends, block_ids,
                                                   layer_id)
                self.kv_recv_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta)  # type: ignore[union-attr, call-arg, arg-type]
                first_flag = False
                yield None
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug(f"Retrieved {retrieved_tokens} "
                     f"out of {num_required_tokens} "
                     f"out of total {len(tokens)} tokens")

        yield ret_mask

    def store_layer(
        self,
        req_id: str,
        tokens: torch.Tensor,
        block_ids: list[int],
        mask: Optional[torch.Tensor] = None,
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

        if mask is not None:
            num_stored_tokens = torch.sum(mask).item()
        else:
            num_stored_tokens = len(tokens)

        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)  #[block_num,layer_num]

        if keys:
            keys = [list(row) for row in zip(*keys)]  #[layer_num,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta = LasyerMultiBlockReqMeta(req_id, keys_multi_chunk,
                                                   starts, ends, block_ids,
                                                   layer_id)
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta)  # type: ignore[union-attr, call-arg, arg-type]
                yield
        else:
            for layer_id in range(self.num_layers):
                yield
        logger.debug(
            f"Stored {num_stored_tokens} out of total {len(tokens)} tokens")

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.kv_role == 'kv_producer' else set())
        done_recving = self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
        )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive "
            "requests: %d, tp_rank:%d", len(done_sending), len(done_recving),
            self.tp_rank)
        return done_sending, done_recving

    def wait_layer_transfer_finish(self):
        time.sleep(10)
        pass

    def lookup(
        self,
        tokens: Union[torch.Tensor, List[int]],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]

        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0

        for start, end, key in self.token_database.process_tokens(tokens):
            try:
                if use_layerwise:
                    keys = []
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for key in keys_multi_layer:
                        keys.append(key.to_string())
                    # batch is_exists
                    ress = self.m_store.batch_exists(keys)
                    res = 1
                    for value in ress:
                        if value != 1:
                            res = 0
                            break
                else:
                    res = self.m_store.exists(key)
                if res == 1:
                    continue
                else:
                    return start
            except Exception as e:
                logger.warning(f"Remote connection failed in contains: {e}")
                return start

        # all tokens where found, return the maximal end
        return end

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        self.m_store.close()
