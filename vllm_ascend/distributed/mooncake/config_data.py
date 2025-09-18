import array
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorMetadata
from vllm.utils import cdiv, logger
from vllm.v1.core.sched.output import NewRequestData


@dataclass
class MooncakeEngineMetadata:
    """name of the LLM model"""

    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    """ (num_layer, 2, metadata.block_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    block_size: int = 128
    """ whether use MLA"""
    use_mla: bool = False


@dataclass(order=True)
class MooncakeEngineKey:
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: str

    def __hash__(self):
        return hash((
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
        ))

    def to_string(self):
        return (f"{self.model_name}@{self.world_size}"
                f"@{self.worker_id}@{self.chunk_hash}")

    def split_layers(self, num_layers: int) -> List["LayerMooncakeEngineKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerMooncakeEngineKey(
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.chunk_hash,
                    layer_id,
                ))
        return keys

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "CacheEngineKey",
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
        }

    @staticmethod
    def from_dict(d):
        return MooncakeEngineKey(
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
        )


@dataclass(order=True)
class LayerMooncakeEngineKey(MooncakeEngineKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash((
            self.model_name,
            self.world_size,
            self.worker_id,
            self.chunk_hash,
            self.layer_id,
        ))

    def to_string(self):
        return (f"{self.model_name}@{self.world_size}"
                f"@{self.worker_id}@{self.chunk_hash}@{self.layer_id}")


class ChunkedTokenDatabase():

    def __init__(
        self,
        metadata: MooncakeEngineMetadata,
    ):
        self.metadata = metadata

    def _make_key_by_hash(self,
                          chunk_hash: str,
                          layer_id: Optional[int] = None):
        assert self.metadata is not None
        return MooncakeEngineKey(
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            chunk_hash,
        )

    def _hash(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: str,
    ) -> str:
        # TODO: change it to a more efficient hash function
        if isinstance(tokens, torch.Tensor):
            tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
        elif isinstance(tokens, list):
            tokens_bytes = array.array("I", tokens).tobytes()
        return hashlib.sha256(prefix_hash.encode("ascii") +
                              tokens_bytes).hexdigest()

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.metadata.block_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [metadata.block_size]
        """
        for i in range(0, len(tokens), self.metadata.block_size):
            yield tokens[i:i + self.metadata.block_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[str]:
        prefix_hash = ''
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
    ) -> Iterable[Tuple[int, int, MooncakeEngineKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.metadata.block_size != 0:
            raise ValueError(
                "The number of Falses in the mask is not a multiple of the chunk size."
            )
        total_len = len(tokens)

        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)

        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.metadata.block_size
            end_idx = min(start_idx + self.metadata.block_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in mooncake
    mooncake_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].
            copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[tuple[list[int], ...], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(
                f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor

    block_ids: list[int]
    # # Slot mapping if exchange for block_id
    # slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    is_last_chunk: Optional[bool] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: Optional[LoadSpec] = None,
        skip_save: Optional[bool] = False,
        is_last_chunk: Optional[bool] = None,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (cdiv(tracker.num_saved_tokens + 1, block_size) *
                          block_size if discard_partial_chunks else 0)
        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = ((input_token_len // block_size * block_size)
                              if discard_partial_chunks else input_token_len)

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block ids
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]

        # # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.mooncake_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None
        logger.debug(
            f"request:{tracker.req_id}, meta save spec:{save_spec}, meta load spec:{load_spec}"
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            block_ids=tracker.allocated_block_ids,
            save_spec=save_spec,
            load_spec=load_spec,
            is_last_chunk=is_last_chunk,
        )


class MooncakeConnectorMetadata(KVConnectorMetadata):

    def __init__(self, unfinished_request_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass
class LasyerMultiBlockReqMeta:
    req_id: str
    keys: List[LayerMooncakeEngineKey]
    starts: List[int]
    ends: list[int]
    block_ids: list[int]
    layer_id: int


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"))

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)