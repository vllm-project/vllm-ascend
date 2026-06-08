from typing import Dict, Type

from .backend import Backend
from .memcache_backend import MemcacheBackend
from .mooncake_backend import MooncakeBackend

backend_map: Dict[str, Type[Backend]] = {
    "mooncake": MooncakeBackend,
    "memcache": MemcacheBackend,
}
