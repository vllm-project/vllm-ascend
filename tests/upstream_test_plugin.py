def pytest_configure(config):
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    adapt_patch()
    _bridge_routed_experts_forward_context()


def _bridge_routed_experts_forward_context():
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec
    
    import vllm_ascend.patch.worker.patch_routed_experts_capture as patch_mod

    patch_mod.get_forward_context = lambda: rec.get_forward_context()
