def pytest_configure(config):
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    adapt_patch()
    _use_spawn_for_upstream_tests()


def _use_spawn_for_upstream_tests():
    import tests.utils as vllm_test_utils  # type: ignore[import-untyped]

    vllm_test_utils.requires_spawn_multiprocessing = lambda: True
