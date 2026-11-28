from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as Ns

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, get_kv_cache_groups
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm.v1.worker.gpu.attn_utils import prepare_kernel_block_sizes
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.kimi_c8_static import map_pa_nz_slots, read_static_fp8_cache, write_static_fp8_cache
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.utils import AscendV2KVBlockZeroer


def spec(**changes):
    base = dict(block_size=1664, num_kv_heads=1, head_size=576, dtype=torch.float8_e4m3fn, static_fp8_pa_nz=True)
    return AscendMLAAttentionSpec(**(base | changes))


@pytest.fixture
def environment(monkeypatch):
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads, layer.head_size, layer.kv_lora_rank, layer.qk_rope_head_dim = 96, 512, 512, 64
    layer.impl = Ns(
        kimi_c8_static_fp8=True,
        fa_quant_layer=True,
        dtype=torch.bfloat16,
        use_flashmla=False,
        supports_strided_mla_writer=lambda: False,
    )
    layer.get_kv_cache_spec = lambda _: spec(dtype=torch.bfloat16, static_fp8_pa_nz=False)
    layer.get_attn_backend = lambda: AscendMLABackend
    config = Ns(
        additional_config={"mla_cache_mode": "C8"},
        kv_transfer_config=None,
        quant_config=None,
        model_config=Ns(
            dtype=torch.bfloat16, hf_config=Ns(model_type="kimi_linear"), hf_text_config=Ns(model_type="kimi_linear")
        ),
        scheduler_config=Ns(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        parallel_config=Ns(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        attention_config=Ns(indexer_kv_dtype="auto", hisparse_config=None),
        cache_config=Ns(block_size=1664, cache_dtype="auto"),
        compilation_config=Ns(static_forward_context={"mla": layer}),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *a, **kw: {"mla": layer})
    for name in ("enable_sfa", "enable_sfa_dcp_replicated_indexer", "_is_dsv4_model"):
        monkeypatch.setattr(attn_utils, name, lambda *a: False)
    return layer, config


@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("graph", [False, True])
def test_cache_pipeline_preserves_other_pages(environment, padding, graph, monkeypatch):
    layer, config = environment
    stride, count = (13 + padding) * 73728, 3
    attention = spec(page_size_padded=stride)
    state = MambaSpec(
        block_size=1664, shapes=((16,), (200000,)), dtypes=(torch.bfloat16, torch.float32), page_size_padded=stride
    )
    cfg = KVCacheConfig(
        num_blocks=count,
        kv_cache_tensors=[
            KVCacheTensor(
                size=count * stride, layers=[name], layer_stride=count * stride, block_stride=stride, offset=0
            )
            for name in ("mla", "state")
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=value)
            for name, value in (("mla", attention), ("state", state))
        ],
    )
    raw = attn_utils._allocate_kv_cache(cfg, {}, torch.device("cpu"))
    assert raw["mla"].data_ptr() == raw["state"].data_ptr()
    groups = [
        AttentionGroup(backend=AscendMLABackend, layer_names=["mla"], kv_cache_spec=attention, kv_cache_group_id=0)
    ]
    assert prepare_kernel_block_sizes(replace(cfg, kv_cache_groups=cfg.kv_cache_groups[:1]), [groups]) == [128]
    cache = attn_utils._reshape_kv_cache_v2(groups, raw, "auto", [128, 1664], {}, cfg)["mla"]
    assert cache[0].shape == (count, 1, 18, 1664, 32)
    assert cache[0].data_ptr() == raw["mla"].data_ptr() and layer.impl.kimi_c8_page_stride_bytes == stride
    raw["mla"].fill_(17)
    _, ssm = attn_utils._reshape_mamba_kv_cache(raw["state"], state)
    ssm[0].fill_(0.125)
    protected = raw["mla"][:stride].clone()
    slots = map_pa_nz_slots(torch.tensor([-1, 1664, 1791, 1792, 3327]), 1664, stride)
    features = torch.arange(576)
    values = (torch.arange(5)[:, None] * 3 + (features % 31) / 4 + (features // 32) * 2).to(torch.bfloat16)
    writer = write_static_fp8_cache
    if graph:
        from vllm_ascend.device import kimi_c8_cache_write as module

        def scatter(raw, payload, indices):
            assert payload.shape == (5, 576)  # Negative slots retain fixed capture shape.
            for row, slot in enumerate(indices.tolist()):
                if slot >= 0:
                    raw[slot // raw.shape[3], 0, :, slot % raw.shape[3]] = payload[row].view(18, 32)

        monkeypatch.setattr(module, "_scatter_pa_nz_bytes", scatter)
        writer = module.write_static_fp8_cache_graph
    writer(values, cache[0], slots, 2.0)
    torch.testing.assert_close(raw["mla"][:stride], protected, atol=0, rtol=0)
    assert raw["mla"][stride + 1664 * 576 :].eq(17).all()
    table = torch.tensor([[1]], dtype=torch.int32)
    expected = ((values.float() / 2).to(torch.float8_e4m3fn).float() * 2).to(torch.bfloat16)
    actual = read_static_fp8_cache(cache[0], table.expand(3, -1), [2, 1, 1], 2.0, starts=[127, 1663, 0])
    torch.testing.assert_close(actual[:, 0], expected[[2, 3, 4, 1]])
    copy_kv_cache_blocks_inplace([cache], count, [KVCacheBlockCopy(1, 2)], static_c8=True)
    torch.testing.assert_close(raw["mla"][stride : 2 * stride], raw["mla"][2 * stride :], atol=0, rtol=0)
    torch.testing.assert_close(raw["mla"][:stride], protected, atol=0, rtol=0)
    layer.kv_cache = cache
    zeroer = AscendV2KVBlockZeroer(torch.device("cpu"), groups, [128, 1664], {"mla": layer}, count)
    addresses, strides, sizes, _, _, segments = zeroer._zeroers[0]._meta
    assert segments == 1 and strides.tolist() == [stride // 4]
    assert sizes.tolist() == [stride // 4]
    assert addresses.tolist() == [cache[0].data_ptr()]


@pytest.mark.parametrize("marker", ["layer", "impl"])
def test_real_specs_preserve_draft_dtype_and_per_layer_identity(environment, marker):
    layer, config = environment
    layer.impl.kimi_c8_static_fp8, layer.kimi_c8_static_fp8 = marker == "impl", marker == "layer"
    target = attn_utils.get_kv_cache_spec(config)["mla"]
    assert target.static_fp8_pa_nz and target.page_size_bytes == 1664 * 576
    layer.get_kv_cache_spec = lambda _: target
    config._kimi_c8_draft_layer_prefixes = ("mla",)
    draft = attn_utils.get_kv_cache_spec(config)["mla"]
    assert not draft.static_fp8_pa_nz and draft.dtype == torch.bfloat16 and draft.block_size == 1664
    groups = get_kv_cache_groups(config, {"target": target, "draft": draft})
    assert len(groups) == 1
    per_layer = groups[0].kv_cache_spec.kv_cache_specs
    assert per_layer["target"].static_fp8_pa_nz and not per_layer["draft"].static_fp8_pa_nz
    with pytest.raises(ValueError, match="padding"):
        attn_utils._uses_single_raw_mla_cache(config, "mla", spec(page_size_padded=1664 * 576 + 256))


@pytest.mark.parametrize("state_size", [250000, 600000])
def test_hybrid_planner_rounds_common_page_to_physical_kernel(environment, monkeypatch, state_size):
    layer, config = environment
    state = MambaSpec(block_size=1664, shapes=((16,), (state_size,)), dtypes=(torch.bfloat16, torch.float32))
    draft = deepcopy(layer)
    draft.impl.kimi_c8_static_fp8 = False
    draft.impl.is_draft_cache = True
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *a, **kw: {
            "mla": layer,
            "state": Ns(get_kv_cache_spec=lambda _: state, kv_sharing_target_layer_name=None),
            "draft": draft,
        },
    )
    specs = attn_utils.get_kv_cache_spec(config)
    expected = ((state.real_page_size_bytes + 73727) // 73728) * 73728
    assert specs["mla"].page_size_bytes == specs["state"].page_size_bytes == expected
    assert specs["draft"].page_size_bytes == 1664 * 576 * 2
    assert specs["mla"].static_fp8_page_bytes == 1664 * 576


def test_active_history_addresses_are_validated_without_reading_stale_tail():
    cache = torch.zeros(1, 1, 18, 128, 32, dtype=torch.uint8)
    assert read_static_fp8_cache(cache, torch.tensor([[0, -1], [-1, -1]]), [2, 0], 2.0).shape == (2, 1, 576)
    for table, lengths, starts in (([[0, -1]], [129], [0]), ([[9]], [1], [0]), ([[0]], [2], [127])):
        with pytest.raises(ValueError, match="block table|cache bounds"):
            read_static_fp8_cache(cache, torch.tensor(table), lengths, 1.0, starts=starts)
    for descale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            write_static_fp8_cache(torch.ones(1, 576), cache, torch.tensor([0]), descale)
    with pytest.raises(ValueError, match="capacity"):
        write_static_fp8_cache(torch.zeros(1, 576), cache, torch.tensor([128]), 1.0)


def test_prefill_history_plan_reads_only_active_pages(monkeypatch):
    from vllm_ascend.attention import kimi_c8_static as c8

    page = 1664
    stride = (page + 128) * 576
    raw = torch.full((3 * stride,), 0x3C, dtype=torch.uint8)
    cache = c8.pa_nz_cache_view(raw, 3, page, page_stride_bytes=stride)
    cache[0].fill_(0x40)
    cache[2].fill_(0x38)
    table = torch.tensor([[2, 0, -1], [0, -1, 2], [-1, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([[2, 0, 0], [0, 3, 0], [0, 0, 0]])
    starts = torch.tensor([[page - 1, 4 * page, 0], [0, 2 * page + 1, 0], [0, 0, 0]])
    scale = torch.tensor([2.0])
    expected = [
        c8.read_static_fp8_cache(cache, table, lens, scale, starts=offsets) for lens, offsets in zip(lengths, starts)
    ]
    plan = c8.C8PrefillHistory.prepare(cache, table, lengths, starts)

    def unexpected_scalar_read(*args, **kwargs):
        pytest.fail("a prepared history read must not perform host scalar reads")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", unexpected_scalar_read)
        patch.setattr(torch.Tensor, "item", unexpected_scalar_read)
        actual = [plan.read(cache, i, scale, dtype=torch.bfloat16) for i in range(len(expected))]

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for bad_id in (-1, cache.shape[0]):
        invalid = table.clone()
        invalid[0, 1] = bad_id  # The first chunk crosses into this page.
        with pytest.raises(ValueError, match="cache bounds"):
            c8.C8PrefillHistory.prepare(cache, invalid, lengths, starts)
    with pytest.raises(ValueError, match="cover requested positions"):
        c8.C8PrefillHistory.prepare(cache, table, lengths + 1, starts)


@pytest.mark.parametrize("mode,expected", [("C8", True), (" c8 ", True), ("C16", False), (None, False)])
def test_native_c8_page_selection_uses_only_mode(monkeypatch, mode, expected):
    from vllm_ascend.patch.platform.patch_mamba_config import _using_kimi_c8

    config = Ns(additional_config={"mla_cache_mode": mode}, cache_config=Ns(cache_dtype="auto"))
    model = Ns(hf_text_config=Ns(model_type="kimi_linear", kv_lora_rank=512, qk_rope_head_dim=64))
    for legacy_value in ("0", "1"):
        monkeypatch.setenv("KIMI_C8_PACKED_PAGE", legacy_value)
        assert _using_kimi_c8(config, model) is expected


@pytest.mark.parametrize("c8", [False, True])
def test_mixed_pages_are_opt_in_for_c8(c8):
    from vllm_ascend.patch.platform.patch_kv_cache_utils import _get_kimi_k3_dspark_mixed_kv_cache_groups

    target = spec(static_fp8_pa_nz=c8, dtype=torch.float8_e4m3fn if c8 else torch.bfloat16)
    draft = spec(static_fp8_pa_nz=False, dtype=torch.bfloat16)
    state = MambaSpec(block_size=1664, shapes=((16,),), dtypes=(torch.float32,), page_size_padded=73728)
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        {
            "language_model.model.layers.0.self_attn.attn": target,
            "model.layers.0.self_attn.attn": draft,
            "language_model.model.layers.1.self_attn.attn": state,
        }
    )
    assert (groups is not None) == c8
