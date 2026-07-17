#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm_ascend

patch_path = Path(vllm_ascend.__file__).parent / "patch" / "worker" / "patch_lora_compile_wrapper.py"
patch_spec = spec_from_file_location("test_lora_compile_wrapper_patch", patch_path)
assert patch_spec is not None and patch_spec.loader is not None
patch_module = module_from_spec(patch_spec)
patch_spec.loader.exec_module(patch_module)


def test_vllm_wrapper_baseline_matches():
    patch_module._verify_vllm_wrapper_baseline()


def test_clone_forward_uses_independent_code_objects():
    class Model:
        def forward(self, value):
            return value

    model = Model()
    base = patch_module._clone_forward(model, "base")
    base_one = patch_module._clone_forward(model, "base_one")

    assert base.__func__.__code__ is not model.forward.__func__.__code__
    assert base_one.__func__.__code__ is not model.forward.__func__.__code__
    assert base.__func__.__code__ is not base_one.__func__.__code__
    assert base.__func__.__code__.co_name == "forward_base"
    assert base_one.__func__.__code__.co_name == "forward_base_one"
    assert base(1) == 1
    assert base_one(2) == 2


def test_specialized_init_creates_base_variants(monkeypatch):
    backend_prefixes = []

    class CompilationConfig:
        cudagraph_specialize_lora = True
        backend = "vllm-ascend"

        def init_backend(self, _vllm_config, prefix, is_encoder):
            backend_prefixes.append((prefix, is_encoder))
            return f"backend:{prefix}"

    config = SimpleNamespace(
        lora_config=object(),
        compilation_config=CompilationConfig(),
    )

    class Model:
        def forward(self, value):
            return value

    model = Model()

    def original_init(self, compile_prefix, is_encoder):
        assert patch_module.envs.VLLM_USE_BYTECODE_HOOK is False
        self._compiled_callable = "lora"
        self.first_compile = True
        self.evaluate_guards = False
        self.vllm_config = config

    def fake_compile(callable_fn, **kwargs):
        return SimpleNamespace(callable_fn=callable_fn, kwargs=kwargs)

    monkeypatch.setattr(patch_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch_module, "_verify_vllm_wrapper_baseline", lambda: None)
    monkeypatch.setattr(patch_module, "_ORIGINAL_INIT", original_init)
    monkeypatch.setattr(patch_module.torch, "compile", fake_compile)
    monkeypatch.setattr(patch_module.envs, "VLLM_USE_BYTECODE_HOOK", True)
    monkeypatch.setattr(patch_module.envs, "VLLM_USE_AOT_COMPILE", False)

    patch_module._patched_init(model)

    assert patch_module.envs.VLLM_USE_BYTECODE_HOOK is True
    assert model.specialize_lora is True
    assert model.use_bytecode_hook is False
    assert backend_prefixes == [("base", False), ("base_one", False)]
    assert model._base_compiled_callable.callable_fn.__func__.__code__.co_name == "forward_base"
    assert model._base_one_compiled_callable.callable_fn.__func__.__code__.co_name == "forward_base_one"


def test_specialized_init_rejects_aot(monkeypatch):
    config = SimpleNamespace(
        lora_config=object(),
        compilation_config=SimpleNamespace(cudagraph_specialize_lora=True),
    )
    monkeypatch.setattr(patch_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch_module.envs, "VLLM_USE_AOT_COMPILE", True)

    with pytest.raises(RuntimeError, match="VLLM_USE_AOT_COMPILE=0"):
        patch_module._patched_init(SimpleNamespace())


@pytest.mark.parametrize(
    ("has_lora", "num_tokens", "expected"),
    [
        (True, 8, "lora"),
        (False, 8, "base"),
        (False, 1, "base_one"),
    ],
)
def test_compiled_callable_routing(monkeypatch, has_lora, num_tokens, expected):
    calls = []

    def callable_for(name):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return run

    wrapper = SimpleNamespace(
        _compiled_callable=callable_for("lora"),
        _base_compiled_callable=callable_for("base"),
        _base_one_compiled_callable=callable_for("base_one"),
        _base_dynamic_inputs_marked=True,
        specialize_lora=True,
        use_bytecode_hook=False,
        first_compile=False,
        evaluate_guards=False,
        _punica_has_lora=lambda: has_lora,
        _call_with_optional_nvtx_range=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    descriptor = SimpleNamespace(has_lora=has_lora, num_tokens=num_tokens)
    monkeypatch.setattr(patch_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        patch_module,
        "get_forward_context",
        lambda: SimpleNamespace(batch_descriptor=descriptor),
    )
    monkeypatch.setattr(
        patch_module.wrapper_module,
        "_compilation_context",
        nullcontext,
    )

    assert patch_module._patched_call(wrapper, "input") == expected
    assert calls == [(expected, ("input",), {})]
