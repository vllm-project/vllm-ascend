# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DCUT_ROOT.parent


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def _load_piecewise_helpers():
    tree = ast.parse(_read("patch_piecewise.py"))
    names = {
        "_env_flag",
        "_is_enabled",
        "_ensure_gdn_splitting_ops",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id.startswith(
                ("ENV_", "LEGACY_ENV_", "_WHOLE_", "_RECURRENT_")
            )
            for target in node.targets
        )
    ]
    module = ast.Module(body=assignments + functions, type_ignores=[])
    namespace = {"os": __import__("os")}
    exec(compile(module, "patch_piecewise.py", "exec"), namespace)
    return namespace


def _load_capture_qsl_helper():
    tree = ast.parse(_read("gdn_buffers.py"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_graph_capture_qsl"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "gdn_buffers.py",
            "exec",
        ),
        namespace,
    )
    return namespace["_dcut_graph_capture_qsl"]


def test_piecewise_capture_qsl_supports_ragged_token_buckets() -> None:
    build_qsl = _load_capture_qsl_helper()

    qsl_1480 = build_qsl(1480, 16, 96)
    assert len(qsl_1480) == 94
    assert qsl_1480[-2:] == (1472, 1480)

    qsl_1408 = build_qsl(1408, 16, 96)
    assert len(qsl_1408) == 89
    assert qsl_1408[-1] == 1408

    assert build_qsl(1536, 16, 96)[-1] == 1536
    assert build_qsl(1537, 16, 96) == ()


def test_piecewise_switch_is_registered_and_default_off() -> None:
    envs = (REPO_ROOT / "vllm_ascend" / "envs.py").read_text(
        encoding="utf-8"
    )
    piecewise = _read("patch_piecewise.py")

    assert '"VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE": lambda: bool(' in envs
    assert (
        'os.getenv("VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE", "0")'
        in envs
    )
    assert 'ENV_DCUT_CONFIG = "VLLM_DCUT_CONFIG"' in piecewise
    assert (
        'ENV_GDN_PIECEWISE = "VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE"'
        in piecewise
    )
    assert "if not _is_enabled():" in piecewise
    assert "self.cudagraph_mode != CUDAGraphMode.PIECEWISE" in piecewise


def test_enabled_switch_keeps_native_boundary_and_captures_recurrent(
    monkeypatch,
) -> None:
    helpers = _load_piecewise_helpers()
    monkeypatch.setenv("VLLM_DCUT_CONFIG", "/tmp/dcut.json")
    monkeypatch.setenv("VLLM_DCUT_GDN_PIECEWISE", "1")

    assert helpers["_is_enabled"]()
    ops = helpers["_ensure_gdn_splitting_ops"](
        [
            "vllm::qwen_gdn_attention_core",
            "vllm::dcut_gdn_recurrent",
            "vllm::unrelated_boundary",
        ]
    )
    assert ops.count("vllm::qwen_gdn_attention_core") == 1
    assert ops.count("vllm::dcut_gdn_recurrent") == 0
    assert ops.count("vllm::unrelated_boundary") == 1

    monkeypatch.setenv("VLLM_DCUT_GDN_PIECEWISE", "0")
    assert not helpers["_is_enabled"]()


def test_runner_expands_only_pure_spec_piecewise_gdn() -> None:
    runner = _read("patch_runner.py")

    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "_dcut_gdn_piecewise_enabled" in runner
    assert "_dcut_gdn_recurrent_piecewise_safe" in runner
    assert "clear_unused_rows=True" in runner
    assert 'getattr(self, "pcp_size", 1) == 1' in runner
    assert 'getattr(self, "dcp_size", 1) == 1' in runner
    assert "CUDAGraphMode.NONE" not in runner
    assert "runtime_mode_overridden" not in runner
    assert "_dcut_gdn_local_graph" not in runner
    assert "_dcut_gdn_recurrent_piecewise_enabled" not in runner
    assert "_dcut_piecewise_capture_dummy" in runner
    assert "_should_build_dummy_attn_metadata" in runner
    assert "_dcut_prepare_gdn_graph_capture" in runner
    assert "self.uniform_decode_query_len" in runner
    assert "(capture_dummy or not native_gdn_batch)" in runner
    assert "uniform_decode=%s enabled=%s" in runner


def test_forward_places_recurrent_wrapper_in_piecewise_graph() -> None:
    core = _read("gdn_forward_v023.py")
    patch = _read("patch_gdn_v023.py")
    buffers = _read("gdn_buffers.py")

    assert "forward_with_graphable_recurrent" in core
    assert "torch.ops.vllm.dcut_gdn_recurrent" in core
    assert "pure_piecewise_spec = piecewise_spec_bufs is not None" in core
    assert "0 if pure_piecewise_spec" in core
    assert 'op_name="dcut_gdn_recurrent"' in core
    assert 'mutates_args=["state"]' in core
    assert "if use_graphable_recurrent" in core
    assert "torch.npu.NPUGraph" not in core
    assert "graph.replay()" not in core
    assert "_dcut_gdn_local_graph" not in core
    assert "_dcut_gdn_recurrent_piecewise_safe" in patch
    assert "if piecewise_graph_safe or full_graph_safe:" in patch
    assert "return native_forward(self, hidden_states, output)" in patch
    assert "target_class.forward = _dcut_forward" in patch
    assert "_dcut_gdn_local_graph_expected_prefixes" not in buffers


def test_full_attention_remains_an_eager_piecewise_boundary() -> None:
    install = _read("install.py")
    attention = _read("patch_attention.py")

    assert "and not _patch_attention()" in install
    assert 'patch_marker = "_dcut_piecewise_fia_patched"' in attention
    assert "mode == CUDAGraphMode.PIECEWISE" in attention
    assert "ctx.capturing = False" in attention
    assert "ctx.capturing = orig_capturing" in attention