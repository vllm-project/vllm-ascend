import pytest
import torch

from vllm_ascend.ops import rope_dsv4


LAYER_NAME = "model.layers.0.attn"
CONFIG_KEY = "cfg"


def setup_function():
    state = rope_dsv4._ROPE_STATE
    state.static_cache = {}
    state.runtime_buffer = {}
    state.layer_info = {}
    state.registry_summary = {}


def teardown_function():
    setup_function()


def _install_dsa_rope_state():
    state = rope_dsv4._ROPE_STATE
    cos = torch.arange(16 * 4, dtype=torch.float32).view(16, 1, 1, 4)
    sin = (torch.arange(16 * 4, dtype=torch.float32) + 1000).view(16, 1, 1, 4)
    state.static_cache[CONFIG_KEY] = torch.cat((cos, sin), dim=-1)
    state.registry_summary[CONFIG_KEY] = {"default", "c4"}
    state.layer_info[LAYER_NAME] = (CONFIG_KEY, ["default", "c4"])
    state.runtime_buffer[CONFIG_KEY] = {
        "default": torch.cat((torch.full((8, 1, 1, 4), -1.0), torch.full((8, 1, 1, 4), -2.0)), dim=-1),
        "c4": torch.cat((torch.full((8, 1, 1, 4), -3.0), torch.full((8, 1, 1, 4), -4.0)), dim=-1),
    }
    return cos, sin, state.runtime_buffer[CONFIG_KEY]


def test_dsa_rope_cache_proxy_keeps_compressed_positions_grouped():
    _install_dsa_rope_state()
    global_positions = torch.tensor([2, 3])
    compress_positions = torch.tensor([0, 4])

    default_view = rope_dsv4.get_dsa_rope_cache_proxy(global_positions)[LAYER_NAME]
    compressed_view = rope_dsv4.get_dsa_rope_cache_proxy({"c4": compress_positions})[LAYER_NAME]

    assert default_view.group_name == "default"
    assert torch.equal(default_view.positions, global_positions)
    assert compressed_view.group_name == "c4"
    assert torch.equal(compressed_view.positions, compress_positions)


def test_dsa_rope_cache_proxy_writes_group_specific_runtime_buffer():
    cos_cache, sin_cache, buffers = _install_dsa_rope_state()
    compress_positions = torch.tensor([1, 5])

    compressed_view = rope_dsv4.get_dsa_rope_cache_proxy(
        {"c4": compress_positions},
        use_cache=True,
    )[LAYER_NAME]
    cos, sin = compressed_view.materialize(inverse=True)

    expected_cos = cos_cache[compress_positions]
    expected_sin = sin_cache[compress_positions]
    expected_cos_sin = torch.cat((expected_cos, expected_sin), dim=-1)
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, -expected_sin)
    assert torch.equal(buffers["c4"][:2], expected_cos_sin)
    assert torch.equal(
        buffers["default"],
        torch.cat((torch.full((8, 1, 1, 4), -1.0), torch.full((8, 1, 1, 4), -2.0)), dim=-1),
    )


def test_dsa_rope_cache_view_exposes_standard_cos_sin_cache():
    cos_cache, sin_cache, _ = _install_dsa_rope_state()
    positions = torch.tensor([2, 3])

    view = rope_dsv4.get_dsa_rope_cache_proxy(positions)[LAYER_NAME]

    assert isinstance(view.cos_sin_cache, torch.Tensor)
    assert torch.equal(view.cos_sin_cache, torch.cat((cos_cache, sin_cache), dim=-1))
    assert torch.equal(view.backend_cos_sin_cache(), view.cos_sin_cache)


def test_dsa_rope_cache_proxy_isolates_same_layer_different_configs():
    state = rope_dsv4._ROPE_STATE
    positions = torch.tensor([1, 2])
    config_a = "cfg_a"
    config_b = "cfg_b"
    layer_key_a = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, config_a, ["default"])
    layer_key_b = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, config_b, ["default"])

    cache_a = torch.full((4, 1, 1, 8), 1.0)
    cache_b = torch.full((4, 1, 1, 8), 2.0)
    state.static_cache[config_a] = cache_a
    state.static_cache[config_b] = cache_b
    state.registry_summary[config_a] = {"default"}
    state.registry_summary[config_b] = {"default"}
    state.layer_info[layer_key_a] = (config_a, ["default"])
    state.layer_info[layer_key_b] = (config_b, ["default"])

    proxy = rope_dsv4.get_dsa_rope_cache_proxy(positions)

    assert proxy[layer_key_a].config_key == config_a
    assert proxy[layer_key_b].config_key == config_b
    assert torch.equal(proxy[layer_key_a].cos_sin_cache, cache_a)
    assert torch.equal(proxy[layer_key_b].cos_sin_cache, cache_b)
    with pytest.raises(KeyError):
        proxy[LAYER_NAME]


def test_dsa_forward_by_cache_output_uses_selected_config_cache(monkeypatch):
    state = rope_dsv4._ROPE_STATE
    positions = torch.tensor([1, 2])
    config_a = "cfg_a"
    config_b = "cfg_b"
    layer_key_a = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, config_a, ["default"])
    layer_key_b = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, config_b, ["default"])

    cos_a = torch.full((4, 1, 1, 4), 1.0)
    sin_a = torch.full((4, 1, 1, 4), 10.0)
    cos_b = torch.full((4, 1, 1, 4), 2.0)
    sin_b = torch.full((4, 1, 1, 4), 20.0)
    state.static_cache[config_a] = torch.cat((cos_a, sin_a), dim=-1)
    state.static_cache[config_b] = torch.cat((cos_b, sin_b), dim=-1)
    state.registry_summary[config_a] = {"default"}
    state.registry_summary[config_b] = {"default"}
    state.layer_info[layer_key_a] = (config_a, ["default"])
    state.layer_info[layer_key_b] = (config_b, ["default"])

    def fake_inplace_partial_rotary_mul_dsa_by_cache(x_arg, rope_cache_arg, **kwargs):
        cos, sin = rope_cache_arg.cos_sin_cache.index_select(0, rope_cache_arg.positions).chunk(2, dim=-1)
        if kwargs.get("inverse", False):
            sin = -sin
        x_arg.copy_(cos + sin)

    monkeypatch.setattr(
        rope_dsv4,
        "inplace_partial_rotary_mul_dsa_by_cache",
        fake_inplace_partial_rotary_mul_dsa_by_cache,
    )

    proxy = rope_dsv4.get_dsa_rope_cache_proxy(positions)
    emb = object.__new__(rope_dsv4.ComplexExpRotaryEmbedding)
    out_a = emb.forward_by_cache(torch.zeros(2, 4), proxy[layer_key_a])
    out_b = emb.forward_by_cache(torch.zeros(2, 4), proxy[layer_key_b])

    expected_a = (cos_a.index_select(0, positions) + sin_a.index_select(0, positions)).view(2, 4)
    expected_b = (cos_b.index_select(0, positions) + sin_b.index_select(0, positions)).view(2, 4)
    torch.testing.assert_close(out_a, expected_a)
    torch.testing.assert_close(out_b, expected_b)
    assert not torch.equal(out_a, out_b)


def test_dsa_rope_layer_key_keeps_group_sets_distinct():
    default_key = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, CONFIG_KEY, ["default"])
    compressed_key = rope_dsv4._dsa_rope_layer_key(LAYER_NAME, CONFIG_KEY, ["default", "c4"])

    assert default_key != compressed_key


def test_complex_exp_rotary_embedding_forward_by_cache_uses_dsa_adapter(monkeypatch):
    _install_dsa_rope_state()
    positions = torch.tensor([2, 3])
    rope_cache = rope_dsv4.get_dsa_rope_cache_proxy(positions)[LAYER_NAME]
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    calls = []

    def fake_inplace_partial_rotary_mul_dsa_by_cache(x_arg, rope_cache_arg, **kwargs):
        calls.append((x_arg, rope_cache_arg, kwargs))
        x_arg.add_(1)

    monkeypatch.setattr(
        rope_dsv4,
        "inplace_partial_rotary_mul_dsa_by_cache",
        fake_inplace_partial_rotary_mul_dsa_by_cache,
    )

    emb = object.__new__(rope_dsv4.ComplexExpRotaryEmbedding)
    output = emb.forward_by_cache(x, rope_cache, inverse=True)

    assert output is x
    assert torch.equal(x, torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4) + 1)
    assert calls[0][0].shape == (2, 1, 3, 4)
    assert calls[0][1] is rope_cache
    assert calls[0][2] == {
        "rotary_mode": "interleave",
        "partial_slice": [0, 4],
        "inverse": True,
    }
