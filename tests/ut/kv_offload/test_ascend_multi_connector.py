import builtins
import importlib
from types import ModuleType

from vllm_ascend.distributed.kv_transfer import ascend_multi_connector


def test_import_does_not_require_mooncake(monkeypatch):
    original_import = builtins.__import__

    def import_without_mooncake(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "mooncake" or name.startswith(
            ("mooncake.", "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector")
        ):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_mooncake)

    importlib.reload(ascend_multi_connector)
