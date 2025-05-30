from tests.conftest import VllmRunner
from vllm_ascend.ascend_config import ASCEND_CONFIG


def test_ascend_config():
    assert not ASCEND_CONFIG.is_initialized

    with VllmRunner("Qwen/Qwen2.5-0.5B-Instruct"):
        assert ASCEND_CONFIG.is_initialized
        assert not ASCEND_CONFIG.torchair_graph_config.enabled
        assert not ASCEND_CONFIG.torchair_graph_config.use_cached_graph
        assert not ASCEND_CONFIG.ascend_scheduler_config.enabled
        assert ASCEND_CONFIG.expert_tensor_parallel_size == 1

    input_additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "use_cached_graph": True
        },
        "ascend_scheduler_config": {
            "enabled": True
        },
        "expert_tensor_parallel_size": 2
    }
    with VllmRunner("Qwen/Qwen2.5-0.5B-Instruct",
                    additional_config=input_additional_config):
        assert ASCEND_CONFIG.is_initialized
        assert ASCEND_CONFIG.torchair_graph_config.enabled
        assert ASCEND_CONFIG.torchair_graph_config.use_cached_graph
        assert ASCEND_CONFIG.ascend_scheduler_config.enabled
        assert ASCEND_CONFIG.expert_tensor_parallel_size == 2
