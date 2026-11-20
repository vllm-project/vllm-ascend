# SPDX-License-Identifier: Apache-2.0
"""Temporary model-load hook backport for configured worker extensions."""

from vllm.v1.worker.worker_base import WorkerWrapperBase


def _load_model(self: WorkerWrapperBase, *, load_dummy_weights: bool = False) -> None:
    # A reused worker class can retain a mixin from an earlier instance.
    has_extension = bool(self.vllm_config.parallel_config.worker_extension_cls)
    before_hook = getattr(self.worker, "before_load_model", None) if has_extension else None
    if callable(before_hook):
        before_hook(load_dummy_weights=load_dummy_weights)

    if load_dummy_weights:
        self.worker.load_model(load_dummy_weights=True)
    else:
        # NPUWorker supports the default, no-argument model load.
        self.worker.load_model()

    after_hook = getattr(self.worker, "after_load_model", None) if has_extension else None
    if callable(after_hook):
        after_hook(load_dummy_weights=load_dummy_weights)


# Defer to upstream once the wrapper owns model loading. Remove this backport
# when all supported vLLM versions include vllm-project/vllm#68936.
if not hasattr(WorkerWrapperBase, "load_model"):
    WorkerWrapperBase.load_model = _load_model
