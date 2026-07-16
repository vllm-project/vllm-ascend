# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.fla import chunk, chunk_o, chunk_o_update, chunk_scaled_dot_kkt
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


class _FakeKernel:
    def __init__(self):
        self.grid = None
        self.grid_result = None
        self.launch_kwargs: dict[str, object] | None = None

    def __getitem__(self, grid):
        self.grid = grid
        self.grid_result = grid({"BV": 128})

        def launch(**kwargs):
            self.launch_kwargs = kwargs

        return launch


class _DummyTensor:
    def __init__(self, name: str):
        self.name = name
        self.shape = (1,)
        self.dtype = torch.float32

    def unsqueeze(self, dim: int):
        return self

    def new_empty(self, *shape):
        return _DummyTensor(f"{self.name}.new_empty")

    def __getitem__(self, item):
        return self

    def __setitem__(self, item, value):
        return None

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def transpose(self, dim0, dim1):
        return self

    def contiguous(self):
        return self

    def movedim(self, source, destination):
        return self

    def to(self, *args, **kwargs):
        return self


class _GatherResult:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.items[item]


def _patch_missing_cdiv(monkeypatch: pytest.MonkeyPatch, module) -> None:
    if hasattr(module.triton, "cdiv"):
        return
    monkeypatch.setattr(
        module.triton,
        "cdiv",
        lambda x, y: (x + y - 1) // y,
        raising=False,
    )


@pytest.mark.parametrize("target", ["chunk_o", "chunk_o_update"])
def test_chunk_leaf_wrappers_use_prebuilt_chunk_offsets(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
):
    fake_kernel = _FakeKernel()
    sentinel = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)

    if target == "chunk_o":
        _patch_missing_cdiv(monkeypatch, chunk_o)
        monkeypatch.setattr(chunk_o, "chunk_fwd_kernel_o", fake_kernel)
        monkeypatch.setattr(
            chunk_o,
            "prepare_chunk_offsets",
            lambda *args, **kwargs: pytest.fail("prepare_chunk_offsets should not be called"),
        )
        chunk_o.chunk_fwd_o(
            q=torch.zeros((2, 4, 1, 8), dtype=torch.float32),
            k=torch.zeros((2, 4, 1, 8), dtype=torch.float32),
            v=torch.zeros((2, 4, 1, 16), dtype=torch.float32),
            h=torch.zeros((4, 1, 8, 16), dtype=torch.float32),
            g=torch.zeros((2, 4, 1), dtype=torch.float32),
            cu_seqlens=cu_seqlens,
            chunk_offsets=sentinel,
        )
    else:
        _patch_missing_cdiv(monkeypatch, chunk_o_update)
        monkeypatch.setattr(chunk_o_update, "chunk_fwd_kernel_o_update", fake_kernel)
        monkeypatch.setattr(
            chunk_o_update,
            "prepare_chunk_offsets",
            lambda *args, **kwargs: pytest.fail("prepare_chunk_offsets should not be called"),
        )
        chunk_o_update.chunk_fwd_o_update(
            q=torch.zeros((2, 4, 1, 8), dtype=torch.float32),
            v=torch.zeros((2, 4, 1, 16), dtype=torch.float32),
            h=torch.zeros((4, 1, 8, 16), dtype=torch.float32),
            h_update=torch.zeros((5, 1, 8, 8), dtype=torch.float32),
            updated_h_state=torch.zeros((1, 8, 16), dtype=torch.float32),
            cu_seqlens=cu_seqlens,
            chunk_offsets=sentinel,
        )

    assert fake_kernel.launch_kwargs is not None
    assert fake_kernel.launch_kwargs["chunk_offsets"] is sentinel


def test_chunk_gated_delta_rule_fwd_threads_prebuilt_chunk_offsets(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk_offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    update_chunk_offsets = torch.tensor([0, 3, 7], dtype=torch.int32)
    final_chunk_indices = torch.tensor([1, 3], dtype=torch.int32)
    prebuilt_meta = type(
        "PrebuiltMeta",
        (),
        {
            "block_indices_cumsum": None,
            "cu_seqlens_host": (0, 4, 7),
            "chunk_indices_chunk64_host": (0, 0, 1, 0),
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": chunk_offsets,
            "update_chunk_offsets_chunk64": update_chunk_offsets,
            "final_chunk_indices_chunk64": final_chunk_indices,
            "chunk_indices_large_block": None,
            "keep_meta": None,
            "cu_seqlens_kern": None,
        },
    )()

    q = _DummyTensor("q")
    k = _DummyTensor("k")
    v = _DummyTensor("v")
    g = _DummyTensor("g")
    beta = _DummyTensor("beta")
    initial_state = _DummyTensor("initial_state")

    non_pcp_calls: list[tuple[str, object]] = []
    pcp_calls: list[tuple[str, object]] = []

    def run_case(world_size: int, calls: list[tuple[str, object]]):
        group = type(
            "Group",
            (),
            {
                "world_size": world_size,
                "rank_in_group": 0,
                "all_gather": lambda self, value, dim: _GatherResult([_DummyTensor("g0"), _DummyTensor("g1")]),
            },
        )()

        monkeypatch.setattr(chunk, "get_forward_context", lambda: type("Ctx", (), {"attn_metadata": None})())
        monkeypatch.setattr(chunk, "get_pcp_group", lambda: group)
        monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda *args, **kwargs: _DummyTensor("g_cumsum"))
        monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", lambda *args, **kwargs: _DummyTensor("A"))
        monkeypatch.setattr(chunk, "solve_tril", lambda *args, **kwargs: _DummyTensor("A_solved"))
        monkeypatch.setattr(chunk, "recompute_w_u_fwd", lambda *args, **kwargs: (_DummyTensor("w"), _DummyTensor("u")))
        monkeypatch.setattr(
            chunk,
            "chunk_gated_delta_rule_fwd_h",
            lambda *args, **kwargs: (_DummyTensor("h"), _DummyTensor("v_new"), _DummyTensor("final_state")),
        )
        monkeypatch.setattr(
            chunk,
            "chunk_gated_delta_rule_fwd_hupdate",
            lambda *args, **kwargs: _DummyTensor("h_update"),
        )
        monkeypatch.setattr(
            chunk.torch,
            "matmul",
            lambda *args, **kwargs: _DummyTensor("matmul"),
            raising=False,
        )
        monkeypatch.setattr(
            chunk.torch,
            "zeros_like",
            lambda *args, **kwargs: _DummyTensor("zeros_like"),
            raising=False,
        )
        monkeypatch.setattr(
            torch.ops._C_ascend,
            "chunk_gated_delta_rule_fwd_h",
            lambda *args, **kwargs: (_DummyTensor("h"), _DummyTensor("v_new"), _DummyTensor("final_state")),
            raising=False,
        )
        monkeypatch.setattr(
            torch.ops._C_ascend,
            "chunk_fwd_o",
            lambda *args, **kwargs: _DummyTensor("o_ascend"),
            raising=False,
        )

        def fake_chunk_fwd_o(*args, **kwargs):
            calls.append(("o", kwargs["chunk_offsets"]))
            return _DummyTensor("o")

        def fake_chunk_fwd_o_update(*args, **kwargs):
            calls.append(("o_update", kwargs["chunk_offsets"]))
            return _DummyTensor("h_updated")

        monkeypatch.setattr(chunk, "chunk_fwd_o", fake_chunk_fwd_o)
        if world_size > 1:
            monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd_hupdate", fake_chunk_fwd_o_update)

        chunk.chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=torch.tensor([0, 4, 7], dtype=torch.int32),
            prebuilt_meta=prebuilt_meta,
        )

    run_case(1, non_pcp_calls)
    assert non_pcp_calls == [("o", chunk_offsets)]

    run_case(2, pcp_calls)
    assert pcp_calls == [("o_update", chunk_offsets), ("o", chunk_offsets)]


def test_chunk_gated_delta_rule_fwd_uses_prebuilt_metadata_without_runtime_tolist(
    monkeypatch: pytest.MonkeyPatch,
):
    prebuilt_meta = type(
        "PrebuiltMeta",
        (),
        {
            "block_indices_cumsum": None,
            "cu_seqlens_host": (0, 4, 7),
            "chunk_indices_chunk64_host": (0, 0, 1, 0),
            "chunk_indices_chunk64": torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
            "chunk_offsets_chunk64": torch.tensor([0, 1, 2], dtype=torch.int32),
            "update_chunk_offsets_chunk64": torch.tensor([0, 2, 4], dtype=torch.int32),
            "final_chunk_indices_chunk64": torch.tensor([1, 3], dtype=torch.int32),
            "chunk_indices_large_block": None,
            "keep_meta": None,
            "cu_seqlens_kern": None,
        },
    )()

    q = _DummyTensor("q")
    k = _DummyTensor("k")
    v = _DummyTensor("v")
    g = _DummyTensor("g")
    beta = _DummyTensor("beta")
    initial_state = _DummyTensor("initial_state")

    captured: dict[str, tuple[int, ...] | None] = {}

    monkeypatch.setattr(chunk, "get_forward_context", lambda: type("Ctx", (), {"attn_metadata": None})())
    monkeypatch.setattr(
        chunk,
        "get_pcp_group",
        lambda: type("Group", (), {"world_size": 1, "rank_in_group": 0})(),
    )
    monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda *args, **kwargs: _DummyTensor("g_cumsum"))
    monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", lambda *args, **kwargs: _DummyTensor("A"))
    monkeypatch.setattr(chunk, "solve_tril", lambda *args, **kwargs: _DummyTensor("A_solved"))
    monkeypatch.setattr(chunk, "recompute_w_u_fwd", lambda *args, **kwargs: (_DummyTensor("w"), _DummyTensor("u")))
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "chunk_gated_delta_rule_fwd_h",
        lambda *args, **kwargs: (
            captured.update(
                {
                    "cu_seqlens": kwargs["cu_seqlens"],
                    "chunk_indices": kwargs["chunk_indices"],
                }
            )
            or (_DummyTensor("h"), _DummyTensor("v_new"), _DummyTensor("final_state"))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "chunk_fwd_o",
        lambda *args, **kwargs: _DummyTensor("o_ascend"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.Tensor,
        "tolist",
        lambda self: pytest.fail("runtime should not convert device tensors to host tuples"),
    )

    chunk.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=torch.tensor([0, 4, 7], dtype=torch.int32),
        prebuilt_meta=prebuilt_meta,
    )

    assert captured["cu_seqlens"] == prebuilt_meta.cu_seqlens_host
    assert captured["chunk_indices"] == prebuilt_meta.chunk_indices_chunk64_host


def test_chunk_gated_delta_rule_fwd_pcp_chaining_subtracts_initial_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """PCP chaining uses (updated_state[i-1] - initial_state), not updated_state[i-1].

    With s0 != 0 (subsequent prefill chunk), the fix subtracts s0 to avoid
    double-counting Φ_i·s0. Verified by checking the returned final_state
    matches the sequential result Φ_1·(Φ_0·s0+p_0)+p_1.
    """
    torch.manual_seed(42)
    N, H, K, V = 1, 2, 4, 4
    s0 = torch.randn(N, H, K, V)
    phi_0 = torch.randn(N, H, K, K)
    phi_1 = torch.randn(N, H, K, K)
    p_0 = torch.randn(N, H, K, V)
    p_1 = torch.randn(N, H, K, V)

    # Each rank computes final_state = Φ_i · s0 + p_i (from shared s0)
    rank0_fs = torch.matmul(phi_0, s0) + p_0
    rank1_fs = torch.matmul(phi_1, s0) + p_1
    # h_update shape [1, N, H, K, K]; after [:, [0], :, :, :] → [1, N, H, K, K]
    h_update_tensor = phi_0.unsqueeze(0)

    prebuilt_meta = type(
        "PrebuiltMeta",
        (),
        {
            "block_indices_cumsum": None,
            "cu_seqlens_host": (0, N),
            "chunk_indices_chunk64_host": (0, 0),
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": torch.tensor([0, 1], dtype=torch.int32),
            "update_chunk_offsets_chunk64": torch.tensor([0, 2], dtype=torch.int32),
            "final_chunk_indices_chunk64": torch.tensor([0], dtype=torch.int32),
            "chunk_indices_large_block": None,
            "num_decodes": 0,
            "keep_meta": None,
            "cu_seqlens_kern": None,
        },
    )()

    all_gather_returns = [
        torch.stack([rank0_fs, rank1_fs]),  # all_final_state: [2, N, H, K, V]
        torch.stack([phi_0, phi_1]),  # all_final_h_update: [2, N, H, K, K]
    ]

    group = type(
        "Group",
        (),
        {
            "world_size": 2,
            "rank_in_group": 0,
            "all_gather": lambda self, value, dim: all_gather_returns.pop(0),
        },
    )()

    monkeypatch.setattr(chunk, "get_forward_context", lambda: type("Ctx", (), {"attn_metadata": None})())
    monkeypatch.setattr(chunk, "get_pcp_group", lambda: group)
    monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda *a, **kw: _DummyTensor("g_cumsum"))
    monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", lambda *a, **kw: _DummyTensor("A"))
    monkeypatch.setattr(chunk, "solve_tril", lambda *a, **kw: _DummyTensor("A_solved"))
    monkeypatch.setattr(chunk, "recompute_w_u_fwd", lambda *a, **kw: (_DummyTensor("w"), _DummyTensor("u")))
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "chunk_gated_delta_rule_fwd_h",
        lambda *a, **kw: (_DummyTensor("h"), _DummyTensor("v_new"), rank0_fs),
        raising=False,
    )
    monkeypatch.setattr(
        chunk,
        "chunk_gated_delta_rule_fwd_hupdate",
        lambda *a, **kw: h_update_tensor,
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "chunk_fwd_o",
        lambda *a, **kw: _DummyTensor("o_ascendc"),
        raising=False,
    )

    result = chunk.chunk_gated_delta_rule_fwd(
        q=_DummyTensor("q"),
        k=_DummyTensor("k"),
        v=_DummyTensor("v"),
        g=_DummyTensor("g"),
        beta=_DummyTensor("beta"),
        scale=1.0,
        initial_state=s0,
        output_final_state=False,
        cu_seqlens=torch.tensor([0, N], dtype=torch.int32),
        prebuilt_meta=prebuilt_meta,
    )

    final_state = result[3]
    # Sequential: Φ_1·(Φ_0·s0 + p_0) + p_1
    expected = torch.matmul(phi_1, torch.matmul(phi_0, s0) + p_0) + p_1
    torch.testing.assert_close(final_state, expected, rtol=1e-4, atol=1e-4)


def test_chunk_ascendc_wrappers_preserve_bhtd_layout(monkeypatch: pytest.MonkeyPatch):
    batch, qk_heads, value_heads, tokens, head_dim, chunk_size = 1, 2, 3, 5, 4, 2
    k = torch.randn(batch, qk_heads, tokens, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch, value_heads, tokens, head_dim, dtype=torch.bfloat16)
    beta = torch.randn(batch, value_heads, tokens, dtype=torch.float32)
    g = torch.randn(batch, value_heads, tokens, dtype=torch.float32)
    A = torch.randn(batch, value_heads, tokens, chunk_size, dtype=torch.float32)
    w = torch.randn_like(k)
    u = torch.randn_like(v)
    h = torch.randn(batch, value_heads, 3, head_dim, head_dim, dtype=torch.bfloat16)
    captures: dict[str, tuple[torch.Tensor, ...]] = {}

    def fake_recompute(*args, **kwargs):
        captures["recompute"] = args
        return w, u

    def fake_fwd_h(*args, **kwargs):
        captures["fwd_h"] = args
        captures["fwd_h_g"] = kwargs["g"]
        return h, v, torch.empty(0)

    def fake_fwd_o(*args, **kwargs):
        captures["fwd_o"] = args
        captures["fwd_o_g"] = kwargs["g"]
        return v

    monkeypatch.setattr(torch.ops._C_ascend, "npu_recompute_wu_fwd", fake_recompute, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_gated_delta_rule_fwd_h", fake_fwd_h, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_fwd_o", fake_fwd_o, raising=False)

    assert chunk.recompute_w_u_fwd(k, v, beta, g, A) == (w, u)
    assert chunk.chunk_gated_delta_rule_fwd_h(k, w, u, g=g)[0] is h
    assert chunk.chunk_fwd_o(k, k, v, h, g=g) is v

    assert captures["recompute"][0] is k
    assert captures["recompute"][1] is v
    assert captures["recompute"][2] is beta
    assert captures["recompute"][3] is A
    assert captures["recompute"][4] is g
    assert captures["fwd_h"][0] is k
    assert captures["fwd_h"][1] is w
    assert captures["fwd_h"][2] is u
    assert captures["fwd_h_g"] is g
    assert captures["fwd_o"][0] is k
    assert captures["fwd_o"][1] is k
    assert captures["fwd_o"][2] is v
    assert captures["fwd_o"][3] is h
    assert captures["fwd_o"][4] == head_dim**-0.5
    assert captures["fwd_o_g"] is g


def test_chunk_fwd_preserves_head_major_k_for_kkt_and_ascendc_calls(monkeypatch: pytest.MonkeyPatch):
    q = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 3, 5, 4, dtype=torch.bfloat16)
    g = torch.randn(1, 5, 3, dtype=torch.float32)
    beta = torch.rand(1, 5, 3, dtype=torch.bfloat16)
    initial_state = torch.randn(1, 3, 4, 4, dtype=torch.bfloat16)
    captured: dict[str, tuple[torch.Tensor, ...] | torch.Tensor] = {}

    monkeypatch.setattr(chunk, "get_forward_context", lambda: type("Ctx", (), {"attn_metadata": None})())
    monkeypatch.setattr(
        chunk,
        "get_pcp_group",
        lambda: type("Group", (), {"world_size": 1, "rank_in_group": 0})(),
    )
    monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda value, **kwargs: value)

    def fake_kkt(**kwargs):
        captured["kkt_k"] = kwargs["k"]
        return torch.empty(1, 5, 3, 2, dtype=torch.float32)

    monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", fake_kkt)
    monkeypatch.setattr(chunk, "solve_tril", lambda A, **kwargs: A)

    def fake_recompute(**kwargs):
        captured["recompute"] = tuple(kwargs[name] for name in ("k", "v", "beta", "A", "g_cumsum"))
        return (
            torch.empty(1, 3, 5, 4, dtype=torch.bfloat16),
            torch.empty(1, 3, 5, 4, dtype=torch.bfloat16),
        )

    def fake_fwd_h(**kwargs):
        captured["fwd_h"] = tuple(kwargs[name] for name in ("k", "w", "u", "g"))
        return (
            torch.empty(1, 3, 1, 4, 4, dtype=torch.bfloat16),
            torch.empty(1, 3, 5, 4, dtype=torch.bfloat16),
            initial_state,
        )

    def fake_fwd_o(**kwargs):
        captured["fwd_o"] = tuple(kwargs[name] for name in ("q", "k", "v", "h", "g"))
        return torch.empty(1, 3, 5, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(chunk, "recompute_w_u_fwd", fake_recompute)
    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd_h", fake_fwd_h)
    monkeypatch.setattr(chunk, "chunk_fwd_o", fake_fwd_o)

    _, output, _, final_state, _, _, _ = chunk.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=0.5,
        initial_state=initial_state,
        output_final_state=True,
    )

    assert captured["kkt_k"] is k
    assert captured["recompute"][0].shape == (1, 2, 5, 4)
    assert captured["recompute"][1].shape == (1, 3, 5, 4)
    assert captured["recompute"][2].shape == (1, 3, 5)
    assert captured["recompute"][2].is_contiguous()
    assert captured["recompute"][3].shape == (1, 3, 5, 2)
    assert captured["recompute"][4].shape == (1, 3, 5)
    assert captured["fwd_h"][0].shape == (1, 2, 5, 4)
    assert captured["fwd_o"][0].shape == (1, 2, 5, 4)
    assert captured["fwd_o"][2].shape == (1, 3, 5, 4)
    assert output.shape == (1, 5, 3, 4)
    assert final_state is initial_state


def test_kkt_preserves_bth_gate_layout(monkeypatch: pytest.MonkeyPatch):
    k = torch.randn(1, 2, 5, 4)
    beta = torch.rand(1, 5, 3)
    g_cumsum = torch.randn(1, 5, 3)
    captured: dict[str, torch.Tensor] = {}

    def fake_kkt(**kwargs):
        captured["beta"] = kwargs["beta"]
        captured["g_cumsum"] = kwargs["g_cumsum"]
        return kwargs["A"]

    monkeypatch.setattr(chunk_scaled_dot_kkt, "get_aicore_num", lambda: 1)
    monkeypatch.setattr(
        DeviceOperator,
        "chunk_scaled_dot_kkt_fwd",
        staticmethod(fake_kkt),
    )

    A = chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        chunk_size=2,
    )

    assert captured["beta"] is beta
    assert captured["g_cumsum"] is g_cumsum
    assert A.shape == (1, 5, 3, 2)
