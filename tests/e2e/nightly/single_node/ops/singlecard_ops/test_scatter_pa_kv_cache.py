import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from vllm_ascend.utils import enable_custom_op


def _scatter(key, value, key_cache, value_cache, slots, scatter_mode="None"):
    return torch.ops._C_ascend.npu_scatter_pa_kv_cache(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        cache_mode="Norm",
        scatter_mode=scatter_mode,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scatter_mode", ["None", "NHSD"])
def test_scatter_strided_views_and_padding(dtype, scatter_mode):
    enable_custom_op()
    # Match the model's interleaved K/V storage, including V's nonzero offset.
    shape = (2, 2, 4, 128, 128) if scatter_mode == "NHSD" else (2, 2, 128, 4, 128)
    cache = torch.zeros(shape, device="npu", dtype=dtype)
    key_cache, value_cache = cache[:, 0], cache[:, 1]
    original_storage = cache.untyped_storage().data_ptr()
    # The original CANN NHSD kernel requires nonnegative slots. Exercise its
    # valid contract separately; Norm is the padded Qwen model/graph path.
    num_tokens = 2 if scatter_mode == "NHSD" else 3
    key = torch.arange(num_tokens * 4 * 128, dtype=torch.float32).reshape(num_tokens, 4, 128).to("npu", dtype)
    value = -key
    slot_values = [0, 129] if scatter_mode == "NHSD" else [0, 129, -1]
    slots = torch.tensor(slot_values, dtype=torch.int32, device="npu")
    assert not key_cache.is_contiguous()
    assert value_cache.storage_offset() != 0
    for scale in (1, 2, 3):
        _scatter(key * scale, value * scale, key_cache, value_cache, slots, scatter_mode)
        torch.npu.synchronize()
        expected = torch.zeros_like(cache, device="cpu")
        for token, slot in enumerate((0, 129)):
            block, offset = divmod(slot, 128)
            if scatter_mode == "NHSD":
                expected[block, 0, :, offset] = (key * scale)[token].cpu()
                expected[block, 1, :, offset] = (value * scale)[token].cpu()
            else:
                expected[block, 0, offset] = (key * scale)[token].cpu()
                expected[block, 1, offset] = (value * scale)[token].cpu()
        torch.testing.assert_close(cache.cpu(), expected)
        assert cache.untyped_storage().data_ptr() == original_storage


@pytest.mark.parametrize("scatter_mode", ["None", "NHSD"])
def test_scatter_aclgraph_replay_updates_slots(scatter_mode):
    enable_custom_op()
    shape = (2, 2, 4, 128, 128) if scatter_mode == "NHSD" else (2, 2, 128, 4, 128)
    cache = torch.zeros(shape, device="npu", dtype=torch.bfloat16)
    key_cache, value_cache = cache[:, 0], cache[:, 1]
    key = torch.ones((1, 4, 128), device="npu", dtype=torch.bfloat16)
    value = key * -1
    slots = torch.tensor([0], device="npu", dtype=torch.int32)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        _scatter(key, value, key_cache, value_cache, slots, scatter_mode)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        _scatter(key, value, key_cache, value_cache, slots, scatter_mode)
    for slot, scale in ((1, 3), (129, 5)):
        key.fill_(scale)
        value.fill_(-scale)
        slots.fill_(slot)
        graph.replay()
        torch.npu.synchronize()
        block, offset = divmod(slot, 128)
        actual_key = key_cache[block, :, offset] if scatter_mode == "NHSD" else key_cache[block, offset]
        actual_value = value_cache[block, :, offset] if scatter_mode == "NHSD" else value_cache[block, offset]
        torch.testing.assert_close(actual_key.cpu(), torch.full((4, 128), scale, dtype=cache.dtype))
        torch.testing.assert_close(actual_value.cpu(), torch.full((4, 128), -scale, dtype=cache.dtype))
    if scatter_mode == "None":
        before = cache.clone()
        slots.fill_(-1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu(), before.cpu())
