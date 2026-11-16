# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native FlashMLA page allocation, selection and current-row writer checks.

Load production definitions without importing vLLM/torch_npu. Device kernels
are replaced with CPU operations; this is not evidence of NPU compatibility.
"""

import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"
SIZES = tuple(range(128, 1025, 128))


def load(path, name, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text()).body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), path, "exec"), scope)
    return scope[name]


def backends():
    scope = {"FLASHMLA_SUPPORTED_BLOCK_SIZES": SIZES, "MLA_KERNEL_BLOCK_SIZE": 128, "AttentionBackend": object}
    base = load(MLA, "AscendMLABackend", scope)
    native = load(MLA, "AscendFlashMLABackend", scope)

    class LayerBase:
        def get_attn_backend(self):
            return base

    scope["MLAAttention"] = LayerBase
    layer_type = load(MLA, "AscendMLAAttention", scope)
    return base, native, layer_type


@pytest.mark.parametrize("page", [128, 512, 768, 896, 1024])
def test_native_backend_exposes_unsplit_manager_page(page):
    base, native, layer_type = backends()
    layer = layer_type()
    layer.impl = SimpleNamespace(use_flashmla=True)
    assert layer.get_attn_backend() is native
    assert native.get_supported_kernel_block_sizes(SimpleNamespace(block_size=page)) == [page]
    assert page in native.get_supported_kernel_block_sizes()
    layer.impl.use_flashmla = False
    assert layer.get_attn_backend() is base
    assert base.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize("draft_only", [False, True])
def test_native_pages_preserve_target_fia_prefill_detection(draft_only):
    base, native, _ = backends()
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and any(isinstance(target, ast.Attribute) and target.attr == "use_fia" for target in node.targets)
    )
    runner = SimpleNamespace(attn_groups=[[SimpleNamespace(backend=native, layer_names=["mla"])]])
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "<fia-detection>", "exec"),
        {
            "self": runner,
            "draft_layer_names": {"mla"} if draft_only else set(),
            "AscendAttentionBackend": object,
            "AscendMLABackend": base,
            "AscendFlashMLABackend": native,
        },
    )
    assert runner.use_fia is not draft_only


@pytest.mark.parametrize(
    "page,interleave,dcp,expected",
    [
        (128, 128, 8, True),
        (512, 512, 8, True),
        (768, 768, 8, True),
        (896, 896, 8, True),
        (896, 128, 8, False),
        (896, 896, 1, True),
        (2048, 2048, 8, False),
        (900, 900, 8, False),
    ],
)
def test_selection_keeps_reader_page_and_interleave_consistent(page, interleave, dcp, expected):
    scope = {
        "torch": torch,
        "FLASHMLA_SUPPORTED_BLOCK_SIZES": SIZES,
        "FLASHMLA_V_DIM": 512,
        "FLASHMLA_QK_DIM": 576,
        "MLA_FLASH_SUPPORTED_Q_HEADS": frozenset((8, 12, 64, 96)),
        "HardwareCapability": SimpleNamespace(MLA_FLASH="mla"),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "supports_component_major_mla_pd": lambda _: True,
        "enable_sfa": lambda _: False,
        "get_flashmla_ops": lambda: (object(), object()),
    }
    select = load(MLA, "_can_use_flashmla", scope, "AscendMLAImpl")
    impl = SimpleNamespace(
        num_heads=12,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=None,
            cache_config=SimpleNamespace(block_size=page),
            parallel_config=SimpleNamespace(decode_context_parallel_size=dcp, cp_kv_cache_interleave_size=interleave),
        ),
    )
    assert select(impl) is expected


def allocate(page, kernel_page):
    # Real allocation/reshape functions; unrelated cache families are stubs.
    class Spec:
        pass

    spec = Spec()
    spec.block_size, spec.num_kv_heads, spec.dtype = page, 1, torch.bfloat16
    spec.page_size_bytes = (page * 576 + 32) * 2
    raw = torch.full((3 * spec.page_size_bytes,), 0xA5, dtype=torch.uint8)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"mla": SimpleNamespace(impl=SimpleNamespace(use_flashmla=True))}
        )
    )
    scope = {
        "torch": torch,
        "math": math,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "get_current_vllm_config": lambda: config,
        "_is_dsv4_model": lambda _: False,
        "_get_layer_kv_cache_specs": lambda _: {"mla": spec},
        "is_deepseek_v41_cache": lambda _: False,
        "get_storage_block_size": lambda item: item.block_size,
        "is_hidden_state_cache_spec": lambda _: False,
        "_uses_single_raw_mla_cache": lambda *args: True,
        "_get_attention_kv_cache_dims": lambda *args: (512, 64),
    }
    for name in (
        "FullAttentionSpec",
        "AscendMLAAttentionSpec",
        "AscendSFAIndexerCacheSpec",
        "AscendIndexerKPoolTailSpec",
        "MambaSpec",
    ):
        scope[name] = type(name, (), {})
    for name in ("row_major_strides", "make_page_strided_cache_view", "get_single_raw_mla_backing"):
        load("vllm_ascend/worker/utils.py", name, scope)
    reshape = load("vllm_ascend/worker/v2/attn_utils.py", "_reshape_kv_cache_v2", scope)
    cache = reshape(
        [SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=spec, layer_names=["mla"])],
        {"mla": raw},
        "auto",
        [kernel_page],
        {},
        SimpleNamespace(num_blocks=3),
    )["mla"]
    return cache, raw


@pytest.mark.parametrize("page", [512, 768, 896])
def test_allocator_rejects_128_split_of_native_flashmla_page(page):
    with pytest.raises(ValueError, match="requires its native"):
        allocate(page, 128)


@pytest.mark.parametrize("page", [128, 512, 768, 896])
def test_current_rows_scatter_from_128_workspace_to_native_page(page):
    cache, raw = allocate(page, page)
    assert cache.shape == (3, page, 1, 576)
    assert cache.stride(0) == page * 576 + 32
    assert cache.data_ptr() == raw.data_ptr()
    snapshot = raw.clone()
    tokens = 130  # Cross a workspace page while writing different reader pages.
    slots = torch.full((tokens,), -1, dtype=torch.int64)
    slots[:3] = torch.tensor([2 * page + page - 1, 0, page + 1])
    slots[-1] = page
    current_nope = torch.arange(tokens, dtype=torch.bfloat16)[:, None, None].expand(-1, 1, 512)
    current_rope = -current_nope[..., :64]
    scope = {
        "torch": torch,
        "FLASHMLA_BLOCK_SIZE": 128,
        "MLAPO_MAX_SUPPORTED_TOKENS": 1024,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    impl = SimpleNamespace(
        use_flashmla=True,
        use_mla_rope=True,
        _flashmla_writer_buffers=None,
        W_UV=torch.empty(1),
        dtype=torch.bfloat16,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
            cache_config=SimpleNamespace(block_size=page),
        ),
    )
    for name in ("_prepare_flashmla_writer_buffers", "_prolog_v3_writer_workspace", "_run_prolog_v3"):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    impl._prepare_flashmla_writer_buffers()
    assert impl._flashmla_writer_buffers[0].shape == (2, 128, 1, 512)
    impl.q_a_layernorm = impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-5)
    impl.fa_quant_layer = False
    impl._prepare_prolog_v3_token = lambda x: (
        x,
        {},
        {name: None for name in ("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr")},
    )
    impl._flatten_prolog_rope_table = lambda table, count: table

    def prolog(**kwargs):
        assert kwargs["tile_size"] == 128
        assert kwargs["kv_cache"].shape[1] == 128
        torch.testing.assert_close(kwargs["cache_index"], torch.arange(tokens))
        kwargs["kv_cache"].flatten(0, 1)[:tokens].copy_(current_nope)
        kwargs["kr_cache"].flatten(0, 1)[:tokens].copy_(current_rope)
        return (torch.zeros(tokens, 12, 512), torch.zeros(tokens, 12, 64))

    def scatter(*, key, value, key_cache, value_cache, slot_mapping):
        for token, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                key_cache[slot // page, slot % page].copy_(key[token])
                value_cache[slot // page, slot % page].copy_(value[token])

    scope["torch_npu"] = SimpleNamespace(npu_mla_prolog_v3=prolog)
    scope["DeviceOperator"] = SimpleNamespace(reshape_and_cache=scatter)
    impl._run_prolog_v3(
        torch.zeros(tokens, 1),
        torch.zeros(tokens, 64),
        torch.zeros(tokens, 64),
        (cache[..., :512], cache[..., 512:]),
        slots,
        quantize_query=False,
    )
    changed = torch.zeros_like(raw, dtype=torch.bool)
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            row = cache[slot // page, slot % page]
            torch.testing.assert_close(row[..., :512], current_nope[token])
            torch.testing.assert_close(row[..., 512:], current_rope[token])
            start = (slot // page * cache.stride(0) + slot % page * 576) * 2
            changed[start : start + 576 * 2] = True
    assert torch.equal(raw[~changed], snapshot[~changed])
