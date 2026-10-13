# SPDX-License-Identifier: Apache-2.0
"""Standalone regression checks: python this_file.py (one visible NPU).

Extract only layout/DP glue from the production functions so this test does
not require a model or a distributed process group. Tensor operators are real.
"""

import ast
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu

from vllm_ascend import utils
from vllm_ascend.ops import dummy_quant_matmul

ROOT = Path(__file__).resolve().parents[4]


def config(load_format):
    return SimpleNamespace(
        load_config=SimpleNamespace(load_format=load_format),
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(mode=1, cudagraph_mode=0),
    )


def test_dummy_isolation():
    with patch.dict(os.environ, {"VLLM_ASCEND_FXRT_DUMMY_QUANT": "1"}):
        original = lambda *a, **kw: (a, kw)
        with (
            patch.object(torch_npu, "npu_quant_matmul", original),
            patch.object(
                torch_npu, "npu_dynamic_quant", return_value=(torch.ones(2, 4, dtype=torch.int8), torch.ones(2))
            ) as quant,
            patch.object(dummy_quant_matmul, "_INSTALLED", False),
        ):
            dummy_quant_matmul.install_dummy_quant_matmul_shim()
            x = torch.ones(2, 4, dtype=torch.bfloat16)
            w = torch.ones(4, 4, dtype=torch.int8)
            for load_format in ("auto", "safetensors", "dummy", "auto"):
                utils.configure_fxrt_prefill_decompose(config(load_format))
                args, kwargs = torch_npu.npu_quant_matmul(x, w, torch.ones(4))
                assert utils.fxrt_dummy_quant_enabled() == (load_format == "dummy")
                assert (args[0].dtype == torch.int8) == (load_format == "dummy")
                assert args[1] is w
                if load_format != "dummy":
                    assert args[0] is x and not kwargs
            assert quant.call_count == 1
            utils.configure_fxrt_prefill_decompose(config("dummy"))
            quant.reset_mock()
            q = torch.ones(2, 4, dtype=torch.int8)
            args, _ = torch_npu.npu_quant_matmul(q, w, torch.ones(4))
            assert args[0] is q
            quant.assert_not_called()
    with patch.dict(os.environ, {"VLLM_ASCEND_FXRT_DUMMY_QUANT": "0"}):
        utils.configure_fxrt_prefill_decompose(config("dummy"))
        assert not utils.fxrt_dummy_quant_enabled()


def layout_code(cp):
    path = "context_parallel/dsa_cp.py" if cp else "dsa_v1.py"
    tree = ast.parse((ROOT / "vllm_ascend/attention" / path).read_text())
    # Find the actual block immediately preceding the batchmatmul call.
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            block = node.orelse
            for i, stmt in enumerate(block):
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "wo_a_weight" for t in stmt.targets
                ):
                    selected = block[i : i + 2]
                    assert isinstance(selected[1], ast.If)
                    return compile(ast.Module(body=selected, type_ignores=[]), path, "exec")
    raise AssertionError("Missing guarded production wo_a layout block")


def test_layouts():
    torch.manual_seed(1024)
    for cp in (False, True):
        code = layout_code(cp)
        for groups in (2, 4, 8):
            raw = torch.randn(groups * 16, 64, device="npu", dtype=torch.bfloat16)
            loaded = raw.view(groups, 16, 64).transpose(1, 2).contiguous()
            x = torch.randn(16, groups, 64, device="npu", dtype=torch.bfloat16)

            def op(w, x=x):
                return torch_npu.npu_transpose_batchmatmul(
                    x,
                    w,
                    bias=None,
                    scale=None,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                    batch_split_factor=1,
                )

            expected = op(loaded)
            for weight in (raw, loaded):
                ns = dict(
                    self=SimpleNamespace(wo_a=SimpleNamespace(weight=weight), n_local_groups=groups),
                    o_proj_input=x,
                    o_proj_groups=groups,
                    group_hidden_dim=64,
                )
                exec(code, ns)
                actual_weight = ns["wo_a_weight"]
                if weight is loaded:
                    assert actual_weight is weight  # retain storage and format
                assert torch.equal(op(actual_weight), expected)
            broken = loaded.view(groups, -1, 64).transpose(1, 2)
            assert not torch.equal(op(broken), expected)
    print(json.dumps({"check": "CP/nonCP loaded3D/raw2D layouts", "pass": True}))


def test_prefill_padding():
    tree = ast.parse((ROOT / "vllm_ascend/attention/dsa_v1.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDSAImpl")
    forward = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    branch = next(
        n
        for n in forward.body
        if isinstance(n, ast.If)
        and any(
            isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "actual_tokens" for t in s.targets)
            for s in n.body
        )
    )
    code = compile(ast.Module(body=branch.body, type_ignores=[]), "prefill_token_count", "exec")
    for actual, padded, local in ((305, 306, 153), (737, 738, 369), (256, 256, 128), (1, 32, 16)):
        hidden = torch.randn(padded, 16)
        ns = dict(
            hidden_states=hidden,
            num_actual_tokens=local,
            self=SimpleNamespace(n_local_heads=4, head_dim=64),
            layer_name="attn",
            attn_metadata=[
                SimpleNamespace(
                    cos={"attn": torch.empty(padded, 1, 1, 64)},
                    prefill=SimpleNamespace(cos={"attn": torch.empty(actual, 1, 1, 64)}),
                )
            ],
            _require_prefill_metadata=lambda m: m.prefill,
        )
        exec(code, ns)
        assert ns["actual_tokens"] == actual
        assert ns["o_proj_input_shape"] == (padded, 4, 64)
        assert torch.equal(ns["hidden_states"], hidden[:actual])
    print(json.dumps({"check": "nonCP real token count excludes model-input padding", "pass": True}))


def test_dummy_dp_isolation():
    tree = ast.parse((ROOT / "vllm_ascend/ops/register_custom_ops.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_pad_and_reduce_impl")
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([2, 2])),
        flash_comm_v1_enabled=True,
        is_draft_model=False,
    )
    extra = SimpleNamespace(flash_comm_v1_enabled=True, pad_size=0, padded_length=2)
    ns = dict(
        torch=torch,
        get_forward_context=lambda: ctx,
        _EXTRA_CTX=extra,
        enable_sp_by_pass=lambda: False,
        get_dp_group=lambda: SimpleNamespace(world_size=2, rank_in_group=1),
        get_ep_group=lambda: SimpleNamespace(reduce_scatter=lambda x, dim: x),
        fxrt_dummy_quant_enabled=utils.fxrt_dummy_quant_enabled,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "pad_reduce", "exec"), ns)
    from vllm_ascend.ops.fxrt_moe import fxrt_moe_gating_top_k_hash

    with patch.dict(os.environ, {"VLLM_ASCEND_FXRT_DUMMY_QUANT": "1"}):
        for load_format in ("auto", "safetensors", "dummy"):
            utils.configure_fxrt_prefill_decompose(config(load_format))
            complete = torch.arange(8).view(4, 2)
            assert torch.equal(ns[fn.name](complete, True), complete)
            local = complete[:2]
            if load_format == "dummy":
                result = ns[fn.name](local, True)
                assert torch.equal(result[:2], torch.zeros_like(local))
                assert torch.equal(result[2:], local)
            else:
                try:
                    ns[fn.name](local, True)
                except RuntimeError:
                    pass  # Real mis-sized tensors must not be silently repaired.
                else:
                    raise AssertionError("Real tensor entered dummy DP fill")
            ids = torch.arange(4)
            with (
                patch("vllm_ascend.ops.fxrt_moe.get_dp_group", return_value=SimpleNamespace(rank_in_group=1)),
                patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", create=True) as original,
            ):
                fxrt_moe_gating_top_k_hash._init_fn(
                    torch.ones(2, 4), 2, None, ids, None, 1, 1, 1.0, 1e-6, 0, 0, 0, False
                )
                passed_ids = original.call_args.args[3]
                assert torch.equal(passed_ids, ids[2:] if load_format == "dummy" else ids)


def test_q_rms_arithmetic():
    from vllm_ascend.ops.dsa_q_rms import fxrt_dsa_q_rms
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

    compiled = torch.compile(fxrt_dsa_q_rms, backend=lambda gm, _: gm.forward, fullgraph=True, dynamic=True)
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (8, 32, 257):
            x = torch.randn(rows, 4, 512, dtype=dtype, device="npu")
            expected = triton_q_rms(x, 1e-6)
            actual = compiled(x, 1e-6)
            torch.npu.synchronize()
            assert torch.equal(expected, actual)
    print(json.dumps({"check": "Q RMS preserves Triton BF16/FP16 arithmetic under fullgraph", "pass": True}))


def test_routing_active_num():
    for rows in (8, 257):
        x = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu")
        ids = torch.randint(0, 8, (rows, 2), device="npu", dtype=torch.int32)
        for quant_mode in (-1, 1):
            outputs = [
                torch_npu.npu_moe_init_routing_v2(
                    x,
                    ids,
                    active_num=active,
                    expert_num=8,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[2, 6],
                    quant_mode=quant_mode,
                )
                for active in (rows * 2, -1)
            ]
            torch.npu.synchronize()
            a, b = outputs
            assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
            valid_rows = int(a[2].sum())
            # Only rows routed to this EP rank are initialized/consumed.
            assert torch.equal(a[0][:valid_rows], b[0][:valid_rows])
            if quant_mode != -1:
                assert torch.equal(a[3][:valid_rows], b[3][:valid_rows])
    print(json.dumps({"check": "routing -1 vs tokens*topk valid rows/indices/counts/scales", "pass": True}))


def test_fused_rms_meta():
    utils.bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.ops.rms_quant_meta import register_rms_quant_meta

    register_rms_quant_meta()
    register_rms_quant_meta()  # Safe after another import path registered it.

    def fn(x, gamma):
        return torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, gamma, epsilon=1e-6)

    for dtype in (torch.bfloat16, torch.float16):
        compiled = torch.compile(fn, backend=lambda gm, _: gm.forward, fullgraph=True, dynamic=True)
        for rows in (256, 257):
            x = torch.randn(rows, 512, dtype=dtype, device="npu")
            gamma = torch.randn(512, dtype=dtype, device="npu")
            expected = fn(x, gamma)
            torch.npu.synchronize()
            actual = compiled(x, gamma)
            torch.npu.synchronize()
            assert actual[0].dtype == torch.int8 and actual[1].dtype == torch.float32
            assert all(torch.isfinite(t).all() for t in actual)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            meta = fn(x.to("meta"), gamma.to("meta"))
            assert [t.dtype for t in meta] == [t.dtype for t in actual], (
                dtype,
                [t.dtype for t in meta],
                [t.dtype for t in actual],
            )
            assert [t.shape for t in meta] == [t.shape for t in actual]
    print(json.dumps({"check": "fused RMS quant Meta + dynamic fullgraph gm.forward", "pass": True}))


def test_serialized_cv_prolog():
    # Exercise the production scheduling/body, substituting only the linear
    # weights and cache/RoPE sinks. Quantization and normalization run on NPU.
    tree = ast.parse((ROOT / "vllm_ascend/attention/dsa_v1.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_mla_prolog_multistream")

    class Linear:
        def quantize(self, x):
            return torch_npu.npu_dynamic_quant(x)

        def matmul(self, x, scale):
            return (x.float() * scale.reshape(-1, 1)).to(torch.bfloat16)

    gamma = torch.ones(512, device="npu", dtype=torch.bfloat16)
    rms = lambda x: torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)[0]
    state = SimpleNamespace(
        multistream_dsv4_dsa_overlap=True,
        _fxrt_prefill_decompose=False,
        wq_b=None,
        cv_wq_a=Linear(),
        cv_wkv=Linear(),
        cv_wq_b=Linear(),
        q_norm=rms,
        kv_norm=rms,
        n_local_heads=1,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        eps=1e-6,
        q_norm_without_weight=None,
    )
    aux = torch.npu.Stream()
    scope = dict(
        torch=torch,
        torch_npu=torch_npu,
        _is_w8a8_dynamic=lambda _: True,
        dsv4_dsa_overlap_stream=lambda: aux,
        npu_stream_switch=lambda stream, enabled: torch.npu.stream(stream) if enabled else contextlib.nullcontext(),
        _record_dsa_event=lambda name, decompose, stream: stream.record_event(),
        _wait_dsa_event=lambda event, decompose, stream: stream.wait_event(event),
        _wait_dsa_stream=lambda stream, other, decompose: stream.wait_stream(other),
        DeviceOperator=SimpleNamespace(
            dsa_kv_compress_scatter=lambda cache, kv, mapping: cache.copy_(kv),
            apply_dsa_q_rms=lambda q, eps, norm: rms(q.squeeze(1)).unsqueeze(1),
        ),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "cv_prolog", "exec"), scope)
    fn = scope[method.name]
    x = torch.randn(257, 512, device="npu", dtype=torch.bfloat16)
    cache = torch.empty(257, 1, 512, device="npu", dtype=torch.bfloat16)
    with patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *a, **k: None):
        expected = fn(state, x, None, None, cache, None, is_prefill=True)
        torch.npu.synchronize()
        expected_cache = cache.clone()
        state.multistream_dsv4_dsa_overlap = False
        state._fxrt_prefill_decompose = True
        # Serial tracing must neither obtain a stream nor issue an event.
        with patch.object(torch.npu, "current_stream", side_effect=AssertionError("serial stream lookup")):
            actual = fn(state, x, None, None, cache, None, is_prefill=True)
        torch.npu.synchronize()
        assert actual[1].dtype == torch.bfloat16 and actual[2] is None
        assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
        assert torch.equal(expected_cache, cache)
        compiled = torch.compile(
            lambda value, sink: fn(state, value, None, None, sink, None, is_prefill=True),
            backend=lambda gm, _: gm.forward,
            fullgraph=True,
            dynamic=True,
        )
        traced = compiled(x, cache)
        torch.npu.synchronize()
        assert torch.equal(expected[0], traced[0]) and torch.equal(expected[1], traced[1])
        assert torch.equal(expected_cache, cache)
    print(json.dumps({"check": "CV prefill serial vs multistream Q/qr/KV", "pass": True}))


if __name__ == "__main__":
    test_dummy_isolation()
    test_dummy_dp_isolation()
    test_prefill_padding()
    print(json.dumps({"check": "dummy vs real loader gating and INT8 passthrough", "pass": True}))
    torch.npu.set_device(0)
    test_layouts()
    test_q_rms_arithmetic()
    test_routing_active_num()
    test_fused_rms_meta()
    test_serialized_cv_prolog()
    torch.npu.synchronize()
