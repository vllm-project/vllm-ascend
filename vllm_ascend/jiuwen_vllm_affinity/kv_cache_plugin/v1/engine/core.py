import json
from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)


class EngineCoreEx(EngineCore):
    def release_kv_cache(self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]) -> int:
        released_blocks = 0
        for params, release_index in token_requests:
            request = decode_engine_core_request(params)
            logger.debug("request decode %s", request)
            req = Request.from_engine_core_request(request, self.request_block_hasher)
            if not req.all_token_ids:
                release_block_index = 0
            else:
                release_block_index = (release_index * len(req.block_hashes)) // len(req.all_token_ids)
            released_blocks += self.scheduler.release_kv_cache(session_id, req.block_hashes[release_block_index:])
        return released_blocks


def pack_request_sharing_cache_salt(request_id: str, sharing_cache_salt: str) -> str:
    try:
        req_and_salt = {
            "request_id": request_id,
            "sharing_cache_salt": sharing_cache_salt,
        }
        return json.dumps(req_and_salt)
    except Exception:
        return request_id


def unpack_sharing_cache_salt(request_id_and_salt: str) -> str | None:
    try:
        req_and_salt = json.loads(request_id_and_salt)
        return req_and_salt["sharing_cache_salt"]
    except json.decoder.JSONDecodeError:
        return None


def encode_engine_core_request(request: EngineCoreRequest) -> Sequence[bytestr]:
    encoder = MsgpackEncoder()
    return encoder.encode(request)


def decode_engine_core_request(frame: Sequence[bytestr]) -> EngineCoreRequest:
    decoder = MsgpackDecoder(EngineCoreRequest)
    return decoder.decode(frame)


def register_engine_core():
    EngineCore.release_kv_cache = EngineCoreEx.release_kv_cache
