import importlib
import os
import sys
import types

_pkg_name = "vllm.models.deepseek_v4"


def _setup_deepseek_v4_fake_package():
    if _pkg_name in sys.modules:
        return
    vllm_models = importlib.import_module("vllm.models")
    dsv4_dir = next(
        os.path.join(p, "deepseek_v4")
        for p in vllm_models.__path__
        if os.path.isdir(os.path.join(p, "deepseek_v4"))
    )
    pkg = types.ModuleType(_pkg_name)
    pkg.__path__ = [dsv4_dir]
    pkg.__package__ = _pkg_name
    sys.modules[_pkg_name] = pkg
    quant_config = importlib.import_module("vllm.models.deepseek_v4.quant_config")
    pkg.DeepseekV4FP8Config = quant_config.DeepseekV4FP8Config


def ensure_deepseek_v4_fake_package():
    _setup_deepseek_v4_fake_package()

    original_import = __builtins__["__import__"]

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == _pkg_name and _pkg_name in sys.modules:
            return sys.modules[name]
        return original_import(name, globals, locals, fromlist, level)

    if original_import.__name__ != "patched_import_deepseek_v4":
        patched_import.__name__ = "patched_import_deepseek_v4"
        __builtins__["__import__"] = patched_import