# Reuse the platform patch. EngineCore subprocesses only load global/platform
# patches, while workers also import this compatibility module.
import vllm_ascend.patch.platform.patch_use_v2_model_runner  # noqa: F401
