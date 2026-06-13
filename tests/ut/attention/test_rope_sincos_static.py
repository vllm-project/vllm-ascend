import ast
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


def test_model_patches_do_not_index_cos_sin_cache_directly():
    for rel_path in [
        "vllm_ascend/patch/worker/patch_qwen3_5.py",
        "vllm_ascend/patch/worker/patch_qwen3vl.py",
    ]:
        assert "cos_sin_cache[positions]" not in _read(rel_path)


def test_dsa_rope_materialization_stays_in_impl_helpers():
    allowed = {"_materialize_dsa_cos_sin"}
    for rel_path in [
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
    ]:
        callers = set(_get_call_enclosing_functions(rel_path, "materialize_dsa_cos_sin"))
        assert callers <= allowed


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
