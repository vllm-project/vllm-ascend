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

from vllm_ascend.ops.triton.fla import chunk, chunk_o, chunk_o_update
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
    assert non_pcp_calls == []

    run_case(2, pcp_calls)
    assert pcp_calls == [("o_update", chunk_offsets)]


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


def test_chunk_gated_delta_rule_fwd_skip_pcp_all_gather(monkeypatch: pytest.MonkeyPatch):
    """Test skip_pcp_all_gather=True skips PCP all_gather block.

    When skip_pcp_all_gather=True, the PCP all_gather block should be skipped
    even when world_size > 1. This is used when PCP rank has 0 prefill tokens.
    """
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

    all_gather_called = []

    def run_case(world_size: int, skip_pcp_all_gather: bool):
        group = type(
            "Group",
            (),
            {
                "world_size": world_size,
                "rank_in_group": 0,
                "all_gather": lambda self, value, dim: (
                    all_gather_called.append(True) or _GatherResult([_DummyTensor("g0"), _DummyTensor("g1")])
                ),
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
            skip_pcp_all_gather=skip_pcp_all_gather,
        )

    # Test 1: world_size=2, skip_pcp_all_gather=False → all_gather should be called
    all_gather_called.clear()
    run_case(2, False)
    assert len(all_gather_called) > 0, "all_gather should be called when skip_pcp_all_gather=False"

    # Test 2: world_size=2, skip_pcp_all_gather=True → all_gather should NOT be called
    all_gather_called.clear()
    run_case(2, True)
    assert len(all_gather_called) == 0, "all_gather should NOT be called when skip_pcp_all_gather=True"


def test_chunk_gated_delta_rule_fwd_uses_compact_cu_seqlens_kern(monkeypatch: pytest.MonkeyPatch):
    """Test chunk_fwd_o uses cu_seqlens_kern (compact) instead of cu_seqlens_host.

    When prebuilt_meta has cu_seqlens_kern (compressed version with empty segments
    removed), chunk_fwd_o should use it to avoid indexing errors in AscendC kernels.
    """
    cu_seqlens_host = (0, 4, 4, 7)  # Has empty segment (4, 4)
    cu_seqlens_kern = [0, 4, 7]  # Compressed version

    prebuilt_meta = type(
        "PrebuiltMeta",
        (),
        {
            "block_indices_cumsum": None,
            "cu_seqlens_host": cu_seqlens_host,
            "cu_seqlens_kern": cu_seqlens_kern,
            "chunk_indices_chunk64_host": (0, 0, 1, 0),
            "chunk_indices_chunk64": torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
            "chunk_offsets_chunk64": torch.tensor([0, 1, 2], dtype=torch.int32),
            "update_chunk_offsets_chunk64": torch.tensor([0, 2, 4], dtype=torch.int32),
            "final_chunk_indices_chunk64": torch.tensor([1, 3], dtype=torch.int32),
            "chunk_indices_large_block": None,
            "keep_meta": None,
        },
    )()

    q = _DummyTensor("q")
    k = _DummyTensor("k")
    v = _DummyTensor("v")
    g = _DummyTensor("g")
    beta = _DummyTensor("beta")
    initial_state = _DummyTensor("initial_state")

    captured_cu_seqlens = []

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
        lambda *args, **kwargs: (_DummyTensor("h"), _DummyTensor("v_new"), _DummyTensor("final_state")),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "chunk_fwd_o",
        lambda *args, **kwargs: captured_cu_seqlens.append(kwargs["cu_seqlens"]) or _DummyTensor("o_ascend"),
        raising=False,
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
        cu_seqlens=torch.tensor([0, 4, 4, 7], dtype=torch.int32),
        prebuilt_meta=prebuilt_meta,
    )

    # chunk_fwd_o should use cu_seqlens_kern (compressed), not cu_seqlens_host
    assert len(captured_cu_seqlens) > 0, "chunk_fwd_o should be called"
    assert captured_cu_seqlens[0] == cu_seqlens_kern, "chunk_fwd_o should use cu_seqlens_kern (compressed version)"
