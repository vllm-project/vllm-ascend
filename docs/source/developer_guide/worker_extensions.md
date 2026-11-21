### Worker model-load hooks

Configure vLLM's existing `worker_extension_cls` with the importable name of a
mixin class to run setup around model loading. The class can define either or
both of these optional methods:

```python
class ModelLoadExtension:
    def before_load_model(self, *, load_dummy_weights: bool) -> None:
        # The device and model runner exist; weights have not been loaded.
        pass

    def after_load_model(self, *, load_dummy_weights: bool) -> None:
        # Model loading and worker-side loader setup completed successfully.
        model = self.model_runner.get_model()
        assert model is not None
```

Keep the class in an importable module, for example `my_extensions.py`:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    worker_extension_cls="my_extensions.ModelLoadExtension",
)
```

For the API server, pass
`--worker-extension-cls my_extensions.ModelLoadExtension`. The module must be
importable in every worker process. `self` is the worker instance, including
the extension mixin. Do not override `load_model`: vLLM rejects mixin members
that conflict with existing worker methods.

Each load calls the before hook, the normal worker loader, and then the after
hook. An exception in the before hook prevents loading. A loader exception
skips the after hook. Hook exceptions propagate as startup failures; these
hooks do not provide automatic rollback or cleanup.

Normal Ascend loading passes `load_dummy_weights=False` to the hooks and calls
the worker's no-argument loader. An explicit `True` is forwarded to the
worker, which must support that keyword. This does not add dummy-weight
support to `NPUWorker`.

The hooks run during model loading, outside request execution. They may inspect
or prepare weights, but this extension point does not prescribe an IPC layout,
cache implementation, or inter-process synchronization protocol.

This is a temporary backport of
[vLLM PR #68936](https://github.com/vllm-project/vllm/pull/68936). If the installed
upstream `WorkerWrapperBase` already implements `load_model`, the backport
does not replace it, so the upstream method owns hook dispatch. The backport
will be removed when all supported upstream versions provide these hooks.
