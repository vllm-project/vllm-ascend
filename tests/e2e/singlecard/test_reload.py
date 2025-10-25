import pytest
from vllm.attention import Attention
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor)

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import InputBatch

if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
    WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE = current_platform.device_type


def initialize_kv_cache(runner: NPUModelRunner):
    """
    Only perform necessary steps in NPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(
            runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.model_config.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context[
        "layer.0"] = Attention(num_heads, head_size, 0.1)
    runner = NPUModelRunner(vllm_config, DEVICE)
    initialize_kv_cache(runner)
    return runner


model_runner_2 = model_runner


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_load_model_weights_inplace(dist_init, model_runner, model_runner_2):
    # In this test, model_runner loads model + weights in one go, while
    # model_runner_2 loads dummy weights first then load real weights inplace
    model_runner.load_model()
    original_load_format = model_runner_2.load_config.load_format
    model_runner_2.update_config({"load_config": {"load_format": "dummy"}})
    model_runner_2.load_model()  # Initial model loading with dummy weights
    assert str(model_runner.get_model().state_dict()) != str(
        model_runner_2.get_model().state_dict())
    model_runner_2.update_config(
        {"load_config": {
            "load_format": original_load_format
        }})
    model_runner_2.reload_weights()  # Load real weights inplace
    assert str(model_runner.get_model().state_dict()) == str(
        model_runner_2.get_model().state_dict())
