# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for Qwen3.8-Flash-Next MTP integration."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "model.py"
MTP = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "mtp.py"
PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "eagle_proposer.py"
DISPATCHER = ROOT / "vllm_ascend" / "spec_decode" / "__init__.py"
QSA = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "qsa.py"
EAGLE = ROOT / "vllm_ascend" / "spec_decode" / "eagle_proposer.py"
LOCAL_PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "qwen4_exp.py"
BASE_PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"
MODEL_RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
OPS = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "ops.py"
TRITON_QSA = ROOT / "vllm_ascend" / "ops" / "triton" / "qwen4_exp" / "qsa.py"
LIGHTNING_INDEXER = (
    ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "lightning_indexer.py"
)
INDEXER_ROPE = (
    ROOT
    / "vllm_ascend"
    / "models"
    / "qwen4_exp"
    / "nvidia"
    / "ops"
    / "qsa_indexer_rope.py"
)
BUILD_ACLNN = ROOT / "csrc" / "build_aclnn.sh"
ENVS = ROOT / "vllm_ascend" / "envs.py"


def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {path}")


def _method(path: Path, cls_name: str, method_name: str) -> ast.FunctionDef:
    for node in _class(path, cls_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {cls_name}.{method_name} not found")


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def test_qwen4_exp_mtp_uses_local_drafter_inputs_on_last_pp_stage() -> None:
    source = ast.unparse(_method(MTP, "AscendQwen4ExpMultiTokenPredictor", "forward"))
    assert "get_pp_group().is_first_rank or intermediate_tensors is None" in source
    assert "positions.ndim > 1" in source
    assert "hidden_states.reshape(-1, hidden_size)" in source


def test_qwen4_exp_mtp_accepts_multimodal_embedding_arguments() -> None:
    method = _method(MTP, "AscendQwen4ExpMTP", "embed_input_ids")
    arguments = [argument.arg for argument in method.args.args]
    assert arguments == [
        "self",
        "input_ids",
        "multimodal_embeddings",
        "is_multimodal",
    ]


def test_qwen4_exp_proposer_uses_text_hyperconnection_width() -> None:
    source = ast.unparse(_method(PROPOSER, "AscendQwen4ExpMTPProposer", "_get_hidden_size"))
    assert "self.draft_model_config.hf_text_config" in source
    assert "text_config.hidden_size * text_config.hc_count" in source
    assert ".get_hidden_size()" not in source


def test_qwen4_exp_proposer_is_dispatched_before_generic_mtp() -> None:
    source = DISPATCHER.read_text()
    qwen_dispatch = source.index("use_qwen4_exp_mtp")
    generic_dispatch = source.index("return AscendEagleProposer", qwen_dispatch)
    assert qwen_dispatch < generic_dispatch


def test_qwen4_exp_graph_does_not_mark_query_start_loc_dynamic() -> None:
    source = ast.unparse(_method(MODEL, "AscendQwen4ExpModel", "__init__"))
    assert "name !=" in source
    assert "query_start_loc" in source


def test_qwen4_exp_wrappers_use_vendored_model_sources() -> None:
    sources = "\n".join(path.read_text() for path in (MODEL, MTP, QSA))
    assert "vllm.models.qwen4_exp" not in sources
    assert "from .nvidia" in sources


def test_qwen4_exp_proposer_uses_local_backport() -> None:
    source = EAGLE.read_text()
    assert "from vllm_ascend.spec_decode.qwen4_exp import" in source
    assert "from vllm.v1.spec_decode.qwen4_exp import" not in source


def test_qwen4_exp_proposer_bypasses_generic_tp_padding() -> None:
    source = ast.unparse(_class(PROPOSER, "AscendQwen4ExpMTPProposer"))
    assert "def maybe_pad_and_reduce" in source
    assert "def maybe_all_gather_and_unpad" in source


def test_qwen4_exp_mtp_type_is_registered_for_vllm_026() -> None:
    patch = (ROOT / "vllm_ascend" / "patch" / "platform" / "patch_speculative_config.py").read_text()
    assert "qwen4_exp_mtp" in patch
    assert "MTPModelTypes" in patch


def test_qwen4_exp_proposer_allocates_full_hc_hidden_buffer() -> None:
    source = ast.unparse(_method(PROPOSER, "AscendQwen4ExpMTPProposer", "__init__"))
    assert "qwen_hidden_size = self._get_hidden_size()" in source
    assert "self.hidden_states = torch.zeros" in source


def test_qwen4_exp_proposer_accepts_both_cache_group_forms() -> None:
    source = ast.unparse(_method(LOCAL_PROPOSER, "Qwen4ExpMTPProposer", "_map_draft_layers_to_groups"))
    assert "isinstance(group_spec, UniformTypeKVCacheSpecs)" in source
    assert "spec = group_spec" in source


def test_qwen4_exp_runner_forwards_per_group_block_tables() -> None:
    source = MODEL_RUNNER.read_text()
    assert "AscendQwen4ExpMTPProposer" in source
    assert "set_per_group_block_table" in source


def test_spec_proposer_normalizes_multiple_of_block_size() -> None:
    source = ast.unparse(_method(BASE_PROPOSER, "AscendSpecDecodeBaseProposer", "load_model"))
    assert "isinstance(kernel_block_size, MultipleOf)" in source
    assert "kernel_block_size.base" in source


def test_qsa_indexer_uses_portable_torch_chain() -> None:
    source = ast.unparse(_method(QSA, "AscendQSAIndexer", "forward"))
    project = source.index("self.project_qk")
    # The feature-gated branch has its own explicit unsupported-contract
    # fallback. Anchor this assertion on the default portable branch.
    update = source.index("self._update_and_compress", project)
    select = source.index("self._select", update)
    assert project < update < select


def test_qsa_graph_path_has_no_host_sync_or_debug_probes() -> None:
    metadata = ast.unparse(_method(QSA, "AscendQSAIndexer", "_metadata"))
    assert "torch.equal" not in metadata
    qsa_source = QSA.read_text()
    for diagnostic in (
        "QSA_CAPTURE_",
        "sys.getrefcount",
        "data_ptr()",
        "logging.getLogger",
        "logger.warning_once",
        "logger.info_once",
    ):
        assert diagnostic not in qsa_source


def test_qsa_cache_write_keeps_static_update_width() -> None:
    source = ast.unparse(
        next(
            node
            for node in ast.parse(OPS.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "qsa_store_cache_rows"
        )
    )
    assert "masked_select" not in source
    assert "index_put_" in source
    assert "accumulate=True" in source


def test_qsa_triton_cache_write_uses_aligned_static_prefix() -> None:
    source = ast.unparse(
        next(
            node
            for node in ast.parse(TRITON_QSA.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "qsa_store_cache_rows"
        )
    )
    assert "num_updates = min(slot_mapping.shape[0], rows.shape[0])" in source
    assert "slot_mapping = slot_mapping[:num_updates]" in source
    assert "rows = rows[:num_updates]" in source
    assert "slot_mapping.numel()" not in source
    assert ".item()" not in source


def test_qsa_triton_attention_loads_k_in_source_contiguous_order() -> None:
    source = ast.unparse(_function(TRITON_QSA, "_qsa_sparse_paged_gqa_splitk_kernel"))
    key_load = source[source.index("key_rows = tl.load") : source.index("values = tl.load")]
    assert "safe_page[:, None] * stride_k_block" in key_load
    assert "page_offset[:, None] * stride_k_token" in key_load
    assert "dim_offsets[None, :]" in key_load
    assert "mask=valid[:, None]" in key_load
    assert "keys = tl.trans(key_rows)" in key_load


def test_qsa_indexer_defaults_to_triton_selector() -> None:
    source = ast.unparse(_method(QSA, "AscendQSAIndexer", "_select"))
    assert "envs.VLLM_ASCEND_FORCE_QSA_REFERENCE" in source
    assert "qsa_select_paged_tokens_reference" in source
    assert "qsa_select_paged_tokens_triton" in source
    assert source.index("qsa_select_paged_tokens_reference") < source.index(
        "qsa_select_paged_tokens_triton"
    )


def test_qsa_triton_selector_defines_padding_logits() -> None:
    source = ast.unparse(_function(TRITON_QSA, "_qsa_mqa_paged_kernel"))
    assert "if tile_start * BLOCK_N >= visible" not in source
    assert "tl.where(page_valid, score, -float('inf'))" in source
    assert "mask=columns < num_columns" in source


def test_qsa_triton_selector_bounds_page_programs_to_safe_row_batches() -> None:
    source = ast.unparse(_function(TRITON_QSA, "qsa_mqa_paged"))
    assert "page_program = 1 < q.shape[0] <= 32" in source
    assert "if page_program else 1" in source
    kernel = ast.unparse(_function(TRITON_QSA, "_qsa_mqa_paged_kernel"))
    assert "tl.static_range(0, SUBTILES_PER_PAGE)" in kernel
    assert "page_base = k_cache_ptr + safe_physical_page * stride_cache_block" in kernel


def test_qsa_triton_selector_bounds_topk_row_workspace() -> None:
    source = ast.unparse(_function(TRITON_QSA, "qsa_select_paged_tokens"))
    assert "max_rows_per_chunk = 128" in source
    assert "rows_per_chunk = min(max_rows_per_chunk" in source
    assert "range(0, rows, rows_per_chunk)" in source


def test_qsa_lightning_selector_is_one_li_and_one_expand_per_step() -> None:
    function = _function(LIGHTNING_INDEXER, "qsa_select_paged_tokens_lightning")
    source = ast.unparse(function)
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(function))
    assert source.count("torch.ops.npu.npu_lightning_indexer.default") == 1
    assert source.count("expand(") == 1
    assert "query[row_slice]" not in source
    assert "out[row_slice]" not in source
    assert "row_block_table = block_table[row_requests]" in source


def test_qsa_triton_selector_rejects_only_true_capacity_overflow() -> None:
    source = ast.unparse(_function(TRITON_QSA, "qsa_select_paged_tokens"))
    assert "columns = page_table.shape[1] * k_cache.shape[1]" in source
    assert "block_topk = token_topk // compress_ratio" in source
    assert "block_topk > columns" in source
    assert "QSA top-k exceeds the compressed cache capacity" in source


def test_qsa_ascend_backend_uses_six_slab_kv_views() -> None:
    source = MODEL_RUNNER.read_text()
    assert "owner.role == QSA_MAIN" in source
    assert 'six_region_layout.region("r2")' in source
    assert 'six_region_layout.region("r3")' in source
    assert "kv_caches[layer_name] = (k_cache, v_cache)" in source


def test_qsa_expand_e3_is_packaged_for_ascend910_93() -> None:
    source = BUILD_ACLNN.read_text()
    a3_start = source.index("matched SOC branch: ascend910_93")
    a5_start = source.index("matched SOC branch: ascend950", a3_start)
    assert '"qsa_expand_e3"' in source[a3_start:a5_start]


def test_qsa_main_fused_norm_rope_is_gated_and_falls_back() -> None:
    source = ast.unparse(
        _method(QSA, "AscendQwen4ExpQSAAttention", "_project_qkv_gate")
    )
    assert "envs.VLLM_ASCEND_ENABLE_QSA_MAIN_FUSED_NORM_ROPE" in source
    assert "torch.ops.vllm.triton_split_qkv_rmsnorm_mrope" in source
    assert source.count("super()._project_qkv_gate(qkv, positions)") == 2
    assert "cos_sin_cache[positions]" in source


def test_qsa_main_fused_norm_rope_contract_is_exact() -> None:
    source = ast.unparse(
        _method(QSA, "AscendQwen4ExpQSAAttention", "_main_fused_norm_rope_eligible")
    )
    for contract in (
        "self.num_heads == 3",
        "self.num_kv_heads == 1",
        "self.head_dim == 256",
        "== 64",
        "[11, 11, 10]",
        "mrope_interleaved",
    ):
        assert contract in source
    assert "VLLM_ASCEND_ENABLE_QSA_MAIN_FUSED_NORM_ROPE" in ENVS.read_text()


def test_qsa_indexer_split_norm_rope_is_gated_and_falls_back() -> None:
    source = ast.unparse(_function(QSA, "apply_qsa_rope"))
    gate = "envs.VLLM_ASCEND_ENABLE_QSA_INDEXER_SPLIT_NORM_ROPE"
    assert gate in source
    assert "qsa_merge_mrope_cos_sin" in source
    assert "cache[positions]" in source
    assert "tensor.shape[0] >= 16" in source
    assert "VLLM_ASCEND_ENABLE_QSA_INDEXER_SPLIT_NORM_ROPE" in ENVS.read_text()


def test_qsa_indexer_split_norm_rope_keeps_pooling_before_k_norm() -> None:
    update = ast.unparse(_method(QSA, "AscendQSAIndexer", "_update_and_compress"))
    compress = update.index("qsa_compress_groups_with_ratio")
    normalize = update.index("self.normalize_compressed_keys")
    assert compress < normalize

    normalize_source = ast.unparse(
        _method(QSA, "AscendQSAIndexer", "normalize_compressed_keys")
    )
    assert "self.k_layernorm" in normalize_source
    assert "upstream_indexer._gemma_rmsnorm" in normalize_source


def test_qsa_indexer_rope_kernel_only_merges_cache_rows() -> None:
    kernel = ast.unparse(_function(INDEXER_ROPE, "_qsa_merge_mrope_cache_kernel"))
    assert "selected_position" in kernel
    assert "pos_t" in kernel and "pos_h" in kernel and "pos_w" in kernel
    assert "norm" not in kernel.lower()
    assert "weight" not in kernel.lower()

    wrapper = ast.unparse(_function(INDEXER_ROPE, "qsa_merge_mrope_cos_sin"))
    assert wrapper.count("_qsa_merge_mrope_cache_kernel") == 1
    assert "positions.shape[0] != 3" in wrapper
    assert "cache.shape[1] != 64" in wrapper
