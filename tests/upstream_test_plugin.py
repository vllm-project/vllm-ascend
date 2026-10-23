def pytest_configure(config):
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    adapt_patch()
