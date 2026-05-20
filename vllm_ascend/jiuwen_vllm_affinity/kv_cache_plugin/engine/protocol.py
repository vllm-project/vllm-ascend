from collections.abc import Sequence

from vllm.engine.protocol import EngineClient
from vllm.v1.serial_utils import bytestr


class EngineClientEx(EngineClient):
    async def release_kv_cache(self, session_id: str, token_requests: list[tuple[Sequence[bytestr], int]]) -> int:
        return 0


def register_engine_protocol():
    EngineClient.release_kv_cache = EngineClientEx.release_kv_cache
