def pytest_configure(config):
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    adapt_patch()
    _bridge_routed_experts_forward_context()
    _bridge_dspark_test_namespace()


def _bridge_routed_experts_forward_context():
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    import vllm_ascend.patch.worker.patch_routed_experts_capture as patch_mod

    patch_mod.get_forward_context = lambda: rec.get_forward_context()


def _bridge_dspark_test_namespace():
    import vllm.v1.worker.gpu.spec_decode.dspark.speculator as speculator_module
    import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils

    import vllm_ascend.patch.worker.patch_v2.patch_dspark as patch_dspark

    wrapped = patch_dspark._load_dspark_model_with_target_quant
    original = patch_dspark._original_load_dspark_model

    def load_dspark_model(target_model, vllm_config):
        # Upstream tests may pass SimpleNamespace configs without a model path;
        # the Ascend wrapper compares those paths, so use the original loader.
        draft_model_config = vllm_config.speculative_config.draft_model_config
        if not hasattr(draft_model_config, "model") or not hasattr(vllm_config.model_config, "model"):
            return original(target_model, vllm_config)
        return wrapped(target_model, vllm_config)

    # The speculator binds load_dspark_model by name at import time.
    dspark_utils.load_dspark_model = load_dspark_model
    speculator_module.load_dspark_model = load_dspark_model
