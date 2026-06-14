import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _get_call_enclosing_functions(rel_path: str, call_name: str) -> list[str]:
    tree = ast.parse(_read(rel_path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    functions: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != call_name:
            continue
        parent = parents.get(node)
        while parent is not None and not isinstance(parent, ast.FunctionDef):
            parent = parents.get(parent)
        functions.append(parent.name if isinstance(parent, ast.FunctionDef) else "<module>")
    return functions


def _dataclass_fields(rel_path: str, class_name_suffix: str = "Metadata") -> dict[str, set[str]]:
    tree = ast.parse(_read(rel_path))
    fields: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or class_name_suffix not in node.name:
            continue
        class_fields: set[str] = set()
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_fields.add(item.target.id)
        fields[node.name] = class_fields
    return fields


def test_runner_does_not_update_global_rope_slices():
    for rel_path in [
        "vllm_ascend/worker/model_runner_v1.py",
        "vllm_ascend/worker/v2/model_runner.py",
        "vllm_ascend/worker/v2/input_batch.py",
    ]:
        text = _read(rel_path)
        assert "set_cos_and_sin" not in text
        assert "update_cos_sin" not in text


def test_310p_runner_does_not_prepare_external_mrope_slices():
    for rel_path in [
        "vllm_ascend/_310p/model_runner_310p.py",
        "vllm_ascend/_310p/ops/rotary_embedding.py",
    ]:
        text = _read(rel_path)
        assert "prepare_mrope_cos_sin_slices_from_runner" not in text
        assert "set_mrope_apply_rotary_slices" not in text
        assert "_mrope_cos_slice" not in text
        assert "_mrope_sin_slice" not in text
        assert "select_cos_sin_from_cache" not in text


def test_mla_sfa_do_not_materialize_sincos_in_metadata_builders():
    for rel_path in [
        "vllm_ascend/attention/mla_v1.py",
        "vllm_ascend/attention/sfa_v1.py",
    ]:
        text = _read(rel_path)
        assert "get_cos_and_sin_mla" not in text
        assert "attn_metadata.prefill.cos" not in text
        assert "attn_metadata.prefill.sin" not in text
        assert "attn_metadata.decode.cos" not in text
        assert "attn_metadata.decode.sin" not in text
        assert "attn_metadata.cos" not in text
        assert "attn_metadata.sin" not in text


def test_mla_sfa_device_hot_paths_use_rope_cache_adapters():
    no_direct_select_paths = [
        "vllm_ascend/attention/mla_v1.py",
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/context_parallel/mla_cp.py",
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "vllm_ascend/device/device_op.py",
    ]
    for rel_path in no_direct_select_paths:
        text = _read(rel_path)
        assert "select_cos_sin_from_cache" not in text

    attention_hot_paths = [
        "vllm_ascend/attention/mla_v1.py",
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/context_parallel/mla_cp.py",
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
    ]
    forbidden = [
        "torch_npu.npu_interleave_rope(",
        "torch_npu.npu_kv_rmsnorm_rope_cache(",
    ]
    for rel_path in attention_hot_paths:
        text = _read(rel_path)
        for needle in forbidden:
            assert needle not in text

    for rel_path in [
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
    ]:
        assert "torch_npu.npu_rotary_mul(" not in _read(rel_path)

    device_text = _read("vllm_ascend/device/device_op.py")
    assert "torch.ops._C_ascend.mla_preprocess(" not in device_text
    assert "torch_npu.npu_mla_prolog_v2(" not in device_text
    assert "torch_npu.npu_mla_prolog_v3(" not in device_text


def test_mla_cp_pcp_prefill_uses_by_cache_kv_preprocess():
    text = _read("vllm_ascend/attention/context_parallel/mla_cp.py")
    assert "self.exec_kv_prefill(" in text
    assert "no_cache_slots = torch.full" in text
    assert "device=attn_metadata.slot_mapping.device" in text
    assert text.index("self.exec_kv_prefill(") < text.index("get_pcp_group().all_gather")
    assert text.index("get_pcp_group().all_gather") < text.index("DeviceOperator.reshape_and_cache(")

    assert "kv_c, k_pe = prefill_kv_no_split.split" not in text
    assert "self.kv_a_layernorm(" not in text
    assert "prefill_k_pe[num_decode_tokens:num_actual_tokens]" not in text


def test_model_patches_do_not_index_cos_sin_cache_directly():
    for rel_path in [
        "vllm_ascend/patch/worker/patch_qwen3_5.py",
        "vllm_ascend/patch/worker/patch_qwen3vl.py",
    ]:
        text = _read(rel_path)
        assert "cos_sin_cache[positions]" not in text
        assert "select_cos_sin_cache" not in text
        assert "select_cos_sin_from_cache" not in text
        assert "cos, sin" not in text
        assert "torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(" not in text
        assert "split_qkv_rmsnorm_mrope_by_cache" in text


def test_mrope_adapter_passes_full_cache_and_positions_to_fused_op():
    text = _read("vllm_ascend/ops/rope_cache_ops.py")
    assert "triton_split_qkv_rmsnorm_mrope_by_cache" in text
    assert "triton_split_qkv_rmsnorm_mrope\"" not in text
    assert "get_rope_cache(rotary_emb, qkv)[positions]" not in text
    assert "cos_sin=" not in text
    assert "cos_sin_cache=get_rope_cache(rotary_emb, qkv)" in text
    assert "positions=positions" in text


def test_minimax_m2_patch_uses_rope_cache_adapter():
    text = _read("vllm_ascend/patch/worker/patch_minimax_m2.py")
    assert "select_cos_sin_from_cache" not in text
    assert "torch.ops.vllm.split_qkv_tp_rmsnorm_rope(" not in text
    assert "split_qkv_tp_rmsnorm_rope_by_cache" in text


def test_dsa_hot_paths_use_rope_cache_adapters():
    for rel_path in [
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
    ]:
        text = _read(rel_path)
        assert "materialize_dsa_cos_sin" not in text
        assert "torch.ops._C_ascend.inplace_partial_rotary_mul(" not in text
        assert "torch.ops._C_ascend.compressor(" not in text
        assert "compressed_cos" not in text
        assert "compressed_sin" not in text
        assert "get_dsa_rope_cache_proxy" in text
        assert "inplace_partial_rotary_mul_dsa_by_cache" in text
        assert "compressor_dsa_by_cache" in text


def test_dsa_rope_cache_view_uses_standard_cos_sin_cache_contract():
    rope_text = _read("vllm_ascend/ops/rope_dsv4.py")
    adapter_text = _read("vllm_ascend/ops/rope_cache_ops.py")
    binding_text = _read("csrc/torch_binding.cpp")
    compressor_def_text = _read("csrc/attention/compressor/op_host/compressor_dsa_by_cache_def.cpp")
    compressor_tiling_text = _read("csrc/attention/compressor/op_host/compressor_tiling.h")
    compressor_kernel_text = _read("csrc/attention/compressor/op_kernel/compressor.cpp")
    inplace_def_text = _read(
        "csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_dsa_by_cache_def.cpp"
    )
    inplace_proto_text = _read(
        "csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_proto.cpp"
    )
    inplace_kernel_text = _read(
        "csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul_dsa_by_cache.cpp"
    )

    assert "def cos_sin_cache(self) -> torch.Tensor" in rope_text
    assert "def backend_cos_sin_cache(self) -> torch.Tensor" in rope_text
    assert "return split_dsa_cos_sin_cache(self.cos_sin_cache)" not in rope_text
    assert "torch.cat(\n                (cos.unsqueeze" in rope_text
    assert "cos_cache, sin_cache = dsa_rope_cache.cos_sin_cache" not in adapter_text
    assert "_get_dsa_cos_sin_cache" in adapter_text
    assert "return torch.cat(cos_sin_cache" not in adapter_text
    assert '_raise_missing_true_by_cache_backend("DSA cos_sin_cache")' in adapter_text
    assert "split_dsa_cos_sin_cache" not in binding_text
    assert 'Input("cos_sin_cache")' in compressor_def_text
    assert 'Input("cos_sin_cache")' in inplace_def_text
    assert 'Input("cos_cache")' not in compressor_def_text
    assert 'Input("sin_cache")' not in compressor_def_text
    assert 'Input("cos_cache")' not in inplace_def_text
    assert 'Input("sin_cache")' not in inplace_def_text
    assert ".INPUT(cos_sin_cache" in inplace_proto_text
    assert ".INPUT(cos_cache" not in inplace_proto_text
    assert ".INPUT(sin_cache" not in inplace_proto_text
    assert "BY_CACHE_COS_SIN_CACHE_INPUT_INDEX = 7" in compressor_tiling_text
    assert "BY_CACHE_STATE_BLOCK_TABLE_INPUT_INDEX = 8" in compressor_tiling_text
    assert "__gm__ uint8_t *cosSinCache" in compressor_kernel_text
    assert "GM_ADDR cos_sin_cache" in inplace_kernel_text


def test_dsa_prefill_compressed_positions_are_group_keyed():
    text = _read("vllm_ascend/attention/dsa_v1.py")
    assert "compress_positions = _get_padded_compressed_position(" not in text
    assert 'f"c{self.compressor_ratio}": compressed_positions' in text


def test_deepseek_v4_model_local_rope_uses_adapter_boundary():
    text = _read("vllm_ascend/models/deepseek_v4.py")
    assert "torch_npu.npu_rotary_mul(" not in text
    assert "rotary_mul_by_cache" in text
    assert "rotary_mul_materialized" not in text
    assert "legacy cos/sin" not in text


def test_helper_rope_paths_use_adapter_boundary():
    for rel_path in [
        "vllm_ascend/ops/rotary_embedding.py",
        "vllm_ascend/ops/rope_dsv4.py",
    ]:
        text = _read(rel_path)
        assert "torch_npu.npu_rotary_mul(" not in text
        assert "rotary_mul_materialized" in text


def test_legacy_materialized_rope_helpers_stay_at_explicit_boundaries():
    allowed_paths = {
        "select_cos_sin_cache": {
            "vllm_ascend/ops/rotary_embedding.py",
            "vllm_ascend/_310p/ops/rotary_embedding.py",
        },
        "select_cos_sin_from_cache": {
            "vllm_ascend/ops/rotary_embedding.py",
        },
        "rotary_mul_materialized": {
            "vllm_ascend/ops/rope_cache_ops.py",
            "vllm_ascend/ops/rotary_embedding.py",
            "vllm_ascend/ops/rope_dsv4.py",
        },
        "materialize_dsa_cos_sin": {
            "vllm_ascend/ops/rope_dsv4.py",
        },
        "split_dsa_cos_sin_cache": {
            "vllm_ascend/ops/rope_dsv4.py",
        },
    }

    for symbol, allowed in allowed_paths.items():
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")
        offenders = []
        for path in (REPO_ROOT / "vllm_ascend").rglob("*.py"):
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            if pattern.search(path.read_text(encoding="utf-8")) is None:
                continue
            if rel_path not in allowed:
                offenders.append(rel_path)

        assert offenders == [], f"{symbol} appears outside legacy boundary: {offenders}"


def test_mrotary_triton_path_uses_cache_and_positions():
    text = _read("vllm_ascend/ops/rotary_embedding.py")
    assert "mrope_forward_triton_by_cache" in text
    assert "from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope" not in text
    assert "triton_mrope(" not in text
    assert "select_cos_sin_cache(self, positions, query)" not in text
    assert "self.cos = None" not in text
    assert "self.sin = None" not in text


def test_rotary_oot_offsets_fail_closed_and_cache_matching_is_side_effect_free():
    text = _read("vllm_ascend/ops/rotary_embedding.py")
    tree = ast.parse(text)
    mrotary_forward_oot = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "AscendMRotaryEmbedding":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward_oot":
                mrotary_forward_oot = item
                break

    assert mrotary_forward_oot is not None
    assert "offsets" in [arg.arg for arg in mrotary_forward_oot.args.args]
    assert text.count('raise NotImplementedError("Batched rotary embedding with offsets') >= 2
    assert text.count('raise NotImplementedError("Batched MRoPE with offsets') >= 2
    assert "positions, query, key, self.cos_sin_cache" not in text
    assert "rotary_emb._match_cos_sin_cache_dtype" not in text
    assert "rotary_emb.cos_sin_cache = cos_sin_cache" not in text
    assert "self.cos_sin_cache = self.cos_sin_cache.to" not in text

    qwen_text = _read("vllm_ascend/ops/qwen2_decoder.py")
    assert "self.cos_sin_cache = self.cos_sin_cache.to" not in qwen_text


def test_qwen2_decoder_uses_rope_cache_adapter():
    text = _read("vllm_ascend/ops/qwen2_decoder.py")
    assert "torch_npu.npu_rotary_mul(" not in text
    assert "rotary_mul_materialized" not in text
    assert "optimized_apply_rotary_pos_emb(" not in text
    assert "self.rotary_emb(hidden_states, position_ids)" not in text
    assert "cos, sin" not in text
    assert "rotary_mul_by_cache" in text
    assert "AscendQwen2RotaryEmbedding" in text


def test_attention_metadata_does_not_expose_external_sincos_fields():
    forbidden = {"sin", "cos", "compress_sin", "compress_cos", "local_sin", "local_cos"}
    for rel_path in [
        "vllm_ascend/attention/mla_v1.py",
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
    ]:
        for class_name, fields in _dataclass_fields(rel_path).items():
            assert fields.isdisjoint(forbidden), f"{rel_path}:{class_name} exposes {fields & forbidden}"


def test_attention_impls_do_not_keep_legacy_sincos_cache_fields():
    for rel_path in [
        "vllm_ascend/attention/mla_v1.py",
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/attention/context_parallel/mla_cp.py",
        "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
    ]:
        text = _read(rel_path)
        assert "self.cos_cache" not in text
        assert "self.sin_cache" not in text


def test_c_ascend_by_cache_bindings_fail_closed_without_native_opapi():
    text = _read("csrc/torch_binding.cpp")
    hooks = {
        "compressor_by_cache": (
            'is_aclnn_api_available("aclnnCompressorByCache")',
            "compressor_by_cache requires aclnnCompressorByCache",
        ),
        "compressor_dsa_by_cache": (
            'is_aclnn_api_available("aclnnCompressorDsaByCache")',
            "compressor_dsa_by_cache requires aclnnCompressorDsaByCache",
        ),
        "inplace_partial_rotary_mul_by_cache": (
            'is_aclnn_api_available("aclnnInplacePartialRotaryMulByCache")',
            "inplace_partial_rotary_mul_by_cache requires aclnnInplacePartialRotaryMulByCache",
        ),
        "inplace_partial_rotary_mul_dsa_by_cache": (
            'is_aclnn_api_available("aclnnInplacePartialRotaryMulDsaByCache")',
            "inplace_partial_rotary_mul_dsa_by_cache requires aclnnInplacePartialRotaryMulDsaByCache",
        ),
    }
    for wrapper_name, (native_probe, fail_closed_msg) in hooks.items():
        assert native_probe in text, f"{wrapper_name} does not probe native by-cache opapi"
        assert fail_closed_msg in text, f"{wrapper_name} does not fail closed without native by-cache opapi"
        assert text.index(native_probe) < text.index(fail_closed_msg)

    assert 'Tensor positions, Tensor cos_sin_cache, str rotary_mode' in text
    assert "Tensor compress_positions, Tensor cos_sin_cache" in text
    assert "expects cos_sin_cache with shape [..., 2 * rotary_dim]" in text
    assert re.search(r"(?<!_)cos_cache\.dim\(\)\s*==\s*2", text) is None
    assert re.search(r"(?<!_)sin_cache\.dim\(\)\s*==\s*2", text) is None
    assert "if (!resolve_rotary_mode" not in text

    forbidden = [
        "select_cos_sin_from_cache_tensor",
        "select_dsa_cos_sin_from_cache_tensor",
        "at::index_select(cos_sin_cache",
        "at::index_select(cos_cache",
        "at::index_select(sin_cache",
    ]
    for needle in forbidden:
        assert needle not in text


def test_mla_preprocess_by_cache_uses_native_cos_sin_cache_kernel():
    rope_cache_text = _read("vllm_ascend/ops/rope_cache_ops.py")
    binding_text = _read("csrc/torch_binding.cpp")
    adapter_text = _read("csrc/mla_preprocess/mla_preprocess_torch_adpt.h")
    kernel_text = _read("csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp")
    tiling_text = _read("csrc/mla_preprocess/op_host/tiling/mla_preprocess_tiling.h")
    by_cache_test_text = _read("tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_by_cache.py")
    assert "select_cos_sin_from_cache" not in rope_cache_text
    assert "_raise_missing_legacy_by_cache_backend" not in rope_cache_text
    assert '_get_c_ascend_op("mla_preprocess")' not in rope_cache_text
    assert "return has_mla_preprocess_by_cache_kernel()" in rope_cache_text
    assert "requires a true kernel-side by-cache implementation" not in binding_text
    assert "mla_preprocess_by_cache_impl" in adapter_text
    assert "positions_ptr" in adapter_text
    assert "cos_sin_cache_ptr" in adapter_text
    assert "enable_raw_q_out" in adapter_text
    assert "raw_q_out" in adapter_text
    assert "mla_preprocess_by_cache_impl" in kernel_text
    assert "positions," in kernel_text
    assert "cos_sin_cache," in kernel_text
    assert "rawQOut" in kernel_text
    assert "ropeByCache" in tiling_text
    assert "cosSinCacheStride0" in tiling_text
    assert "cosSinCacheHalfDim" in tiling_text
    assert "enableRawQOut" in tiling_text
    assert "torch.ops._C_ascend.mla_preprocess_by_cache(" in by_cache_test_text
    assert "torch.ops._C_ascend.mla_preprocess(" not in by_cache_test_text
    assert "positions" in by_cache_test_text
    assert "cos_sin_cache" in by_cache_test_text
    assert "enable_raw_q_out=True" in by_cache_test_text
    assert "VLLM_ASCEND_RUN_MLA_PREPROCESS_BY_CACHE_OP" in by_cache_test_text
