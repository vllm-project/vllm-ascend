import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorMetadata
from vllm.utils import logger
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
class MooncakeUserKey:
    uid: int
    model_name: str
    world_size: int
    worker_id: int
    value_type: Literal["kv_cache", "token_id"]

    def __hash__(self):
        return hash((
            self.uid,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.value_type,
        ))

    def to_string(self):
        return (f"{self.uid}"
                f"@{self.model_name}@{self.world_size}"
                f"@{self.worker_id}@{self.value_type}")

    def split_layers(self, num_layers: int) -> List["LayerMooncakeUserKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerMooncakeUserKey(
                    self.uid,
                    self.model_name,
                    self.world_size,
                    self.worker_id,
                    self.value_type,
                    layer_id,
                ))
        return keys

    def to_dict(self):
        # Note(Kuntai): this is used for serializing CacheEngineKey via msgpack.
        return {
            "__type__": "MooncakeUserKey",
            "uid": self.uid,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "type": self.value_type,
        }

    @staticmethod
    def from_dict(d):
        return MooncakeUserKey(
            uid=d["uid"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            value_type=d["value_type"],
        )


@dataclass(order=True)
class LayerMooncakeUserKey(MooncakeUserKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash((
            self.uid,
            self.model_name,
            self.world_size,
            self.worker_id,
            self.value_type,
            self.layer_id,
        ))

    def to_string(self):
        return (f"{self.uid}"
                f"{self.model_name}@{self.world_size}"
                f"@{self.worker_id}@{self.value_type}@{self.layer_id}")


@dataclass
class LoadSpec:
    # Number of tokens that are cached in mooncake
    mooncake_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Original history length. This can differ from prompt length for decode
    # requests whose prompt only contains candidates.
    history_len: Optional[int] = None
    # History token ids (used for decode where prompt_token_ids only has candidates)
    history_token_ids: Optional[list[int]] = None


@dataclass
class SaveSpec:
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str
    # user id
    uid: int

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
        history_token_ids: list[int],
        candidate_ids: list[int],
    ) -> "RequestTracker":
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []
        # breakpoint()
        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        uid = new_request.sampling_params.extra_args.get("uid", None)
        if isinstance(uid, list):
            uid = uid[0] if uid else None

        # 当前没有增量推理，暂时直接取history token
        num_candidates = len(candidate_ids)
        token_ids = history_token_ids

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=token_ids,
            uid=uid,
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=len(history_token_ids)
            if num_candidates == 0 else 0,
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
    # User id
    uid: int
    # Request tokens
    token_ids: torch.Tensor

    block_ids: list[int]
    # # Slot mapping if exchange for block_id
    # slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: Optional[LoadSpec] = None,
    ) -> Optional["ReqMeta"]:
        # breakpoint()
        input_token_ids = tracker.token_ids
        num_tokens_to_save = 0
        if load_spec is None:
            num_tokens_to_save = len(input_token_ids)

        save_spec = SaveSpec(num_tokens_to_save > 0)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block ids
        if load_spec is None:
            token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        else:
            token_ids = torch.tensor(input_token_ids)

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
            uid=tracker.uid,
            token_ids=token_ids,
            block_ids=tracker.allocated_block_ids,
            save_spec=save_spec,
            load_spec=load_spec,
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
    key: LayerMooncakeUserKey
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
    use_ascend_direct: bool

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
            master_server_address=config.get("master_server_address"),
            use_ascend_direct=config.get("use_ascend_direct", False))

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)
