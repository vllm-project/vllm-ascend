"""Backport lazy per-tensor filtering until supported vLLM has PR #64395."""

import time

from vllm.model_executor.model_loader import default_loader as loader
from vllm.model_executor.model_loader import weight_utils


def install():
    cls = loader.DefaultModelLoader
    if "filter_unused_per_tensor" in cls.Source.__dataclass_fields__:
        return
    if getattr(cls, "_ascend_external_weight_filter", False):
        return
    original = cls.get_all_weights

    def get_all_weights(self, model_config, model):
        if not getattr(model, "filter_unused_checkpoint_weights", False):
            yield from original(self, model_config, model)
            return
        predicate = getattr(model, "is_unused_checkpoint_weight", None)
        if not callable(predicate):
            raise ValueError("Per-tensor checkpoint filtering requires a predicate")
        config = self.load_config
        if (
            config.load_format not in ("auto", "safetensors")
            or config.safetensors_load_strategy not in (None, "lazy")
            or config.model_loader_extra_config.get("enable_multithread_load")
            or self._encoder_only_lm_prefixes is not None
        ):
            raise ValueError("External weight filtering requires lazy single-threaded safetensors loading")
        primary = cls.Source(
            model_config.model,
            model_config.revision,
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        for source in (primary, *getattr(model, "secondary_weights", ())):
            folder, files, safetensors, _ = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            if not safetensors:
                raise ValueError("External weight filtering requires safetensors")
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            for path in sorted(files, key=weight_utils._natural_sort_key):
                with weight_utils.safe_open(path, framework="pt") as handle:
                    for name in handle.keys():  # noqa: SIM118
                        if (
                            (source.is_unused_weight is not None and source.is_unused_weight(name))
                            or predicate(source.prefix + name)
                            or weight_utils.should_skip_weight(name, self.local_expert_ids)
                        ):
                            continue
                        yield source.prefix + name, handle.get_tensor(name)

    cls.get_all_weights = get_all_weights
    cls._ascend_external_weight_filter = True


install()
