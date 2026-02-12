from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core import unpack_sharing_cache_salt

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash


def from_engine_core_request_v2(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    sharing_cache_salt = unpack_sharing_cache_salt(request.request_id)
    req = cls(
        request_id=request.request_id,
        client_index=request.client_index,
        prompt_token_ids=request.prompt_token_ids,
        prompt_embeds=request.prompt_embeds,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        eos_token_id=request.eos_token_id,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=block_hasher,
    )
    if sharing_cache_salt is not None:
        req.sharing_cache_salt = sharing_cache_salt
    return req


def request_get_sharing_cache_salt(request) -> str | None:
    return None if not hasattr(request, "sharing_cache_salt") else request.sharing_cache_salt


def register_request():
    Request.from_engine_core_request = from_engine_core_request_v2
