# SPDX-License-Identifier: Apache-2.0
"""Check the backport against the real upstream wrapper without NPU setup."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.v1.worker.worker_base import WorkerWrapperBase


def load_patch():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_worker_extension.py"
    spec = importlib.util.spec_from_file_location("_test_worker_extension_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wrapper(monkeypatch):
    # Test versions that need the backport, even when CI runs with native hooks.
    monkeypatch.setattr(WorkerWrapperBase, "load_model", None, raising=False)
    monkeypatch.delattr(WorkerWrapperBase, "load_model")
    load_patch()
    events = []

    class Extension:
        def before_load_model(self, *, load_dummy_weights):
            events.append(("before", load_dummy_weights))

        def after_load_model(self, *, load_dummy_weights):
            events.append(("after", load_dummy_weights))

    class Worker(Extension):
        def load_model(self):
            events.append(("load", False))

    instance = WorkerWrapperBase()
    instance.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(worker_extension_cls="Extension"))
    instance.worker = Worker()
    return SimpleNamespace(wrapper=instance, events=events, extension=Extension)


def test_hooks_bracket_no_argument_model_load(wrapper):
    wrapper.wrapper.load_model()
    assert wrapper.events == [("before", False), ("load", False), ("after", False)]


@pytest.mark.parametrize("extension", ["", "without_hooks", "non_callable_hooks"])
def test_model_load_without_hooks(wrapper, extension):
    if extension == "non_callable_hooks":
        wrapper.extension.before_load_model = None
        wrapper.extension.after_load_model = None
    else:

        class Extension:
            pass

        class Worker(Extension):
            def load_model(self):
                wrapper.events.append(("load", False))

        wrapper.wrapper.worker = Worker()
    wrapper.wrapper.vllm_config.parallel_config.worker_extension_cls = extension
    wrapper.wrapper.load_model()
    assert wrapper.events == [("load", False)]


def test_unconfigured_extension_skips_inherited_hooks(wrapper):
    # A reused worker class can still inherit a previously configured mixin.
    wrapper.wrapper.vllm_config.parallel_config.worker_extension_cls = ""
    wrapper.wrapper.load_model()
    assert wrapper.events == [("load", False)]


def test_explicit_dummy_weight_flag_is_forwarded(wrapper):
    def load(*, load_dummy_weights):
        wrapper.events.append(("load", load_dummy_weights))

    wrapper.wrapper.worker.load_model = load
    wrapper.wrapper.load_model(load_dummy_weights=True)
    assert wrapper.events == [("before", True), ("load", True), ("after", True)]


@pytest.mark.parametrize("hook_type", [classmethod, staticmethod])
@pytest.mark.parametrize("load_dummy_weights", [False, True])
def test_descriptor_hooks_bind_on_worker(wrapper, monkeypatch, hook_type, load_dummy_weights):
    if hook_type is classmethod:

        def before(cls, *, load_dummy_weights):
            assert cls is type(wrapper.wrapper.worker)
            wrapper.events.append(("before", load_dummy_weights))

        def after(cls, *, load_dummy_weights):
            assert cls is type(wrapper.wrapper.worker)
            wrapper.events.append(("after", load_dummy_weights))
    else:

        def before(*, load_dummy_weights):
            wrapper.events.append(("before", load_dummy_weights))

        def after(*, load_dummy_weights):
            wrapper.events.append(("after", load_dummy_weights))

    monkeypatch.setattr(wrapper.extension, "before_load_model", hook_type(before))
    monkeypatch.setattr(wrapper.extension, "after_load_model", hook_type(after))

    if load_dummy_weights:

        def load(*, load_dummy_weights):
            wrapper.events.append(("load", load_dummy_weights))

        wrapper.wrapper.worker.load_model = load

    wrapper.wrapper.load_model(load_dummy_weights=load_dummy_weights)
    assert wrapper.events == [
        ("before", load_dummy_weights),
        ("load", load_dummy_weights),
        ("after", load_dummy_weights),
    ]


def test_load_failure_skips_after_hook(wrapper):
    def fail_load():
        wrapper.events.append(("load", False))
        raise RuntimeError("load failed")

    wrapper.wrapper.worker.load_model = fail_load
    with pytest.raises(RuntimeError, match="load failed"):
        wrapper.wrapper.load_model()
    assert wrapper.events == [("before", False), ("load", False)]


def test_before_hook_failure_prevents_load(wrapper):
    def fail_before(self, *, load_dummy_weights):
        raise RuntimeError("before failed")

    wrapper.extension.before_load_model = fail_before
    with pytest.raises(RuntimeError, match="before failed"):
        wrapper.wrapper.load_model()
    assert wrapper.events == []


def test_after_hook_failure_propagates(wrapper):
    def fail_after(self, *, load_dummy_weights):
        raise RuntimeError("after failed")

    wrapper.extension.after_load_model = fail_after
    with pytest.raises(RuntimeError, match="after failed"):
        wrapper.wrapper.load_model()
    assert wrapper.events == [("before", False), ("load", False)]


def test_existing_upstream_load_method_is_preserved(monkeypatch):
    def native_load(self, *, load_dummy_weights=False):
        self.events.append("native")

    monkeypatch.setattr(WorkerWrapperBase, "load_model", native_load, raising=False)
    load_patch()
    assert WorkerWrapperBase.load_model is native_load
    instance = WorkerWrapperBase()
    instance.events = []
    instance.load_model()
    assert instance.events == ["native"]


def test_reimport_preserves_installed_backport(wrapper):
    installed = WorkerWrapperBase.load_model
    load_patch()
    assert WorkerWrapperBase.load_model is installed
    wrapper.wrapper.load_model()
    assert wrapper.events == [("before", False), ("load", False), ("after", False)]
