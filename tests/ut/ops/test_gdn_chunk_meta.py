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

from vllm_ascend.ops.triton import gdn_chunk_meta
from vllm_ascend.ops.triton.fla import chunk, chunk_o, chunk_o_update
from vllm_ascend.ops.triton.gdn_chunk_meta import build_chunk_meta_device


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

    def transpose(self, dim0, dim1):
        return self

    def contiguous(self):
        return self


class _GatherResult:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.items[item]


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


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
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": chunk_offsets,
            "update_chunk_offsets_chunk64": update_chunk_offsets,
            "final_chunk_indices_chunk64": final_chunk_indices,
            "chunk_indices_large_block": None,
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
                "all_gather": lambda self, value, dim: _GatherResult(
                    [_DummyTensor("g0"), _DummyTensor("g1")]
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

        def fake_chunk_fwd_o(*args, **kwargs):
            calls.append(("o", kwargs["chunk_offsets"]))
            return _DummyTensor("o")

        def fake_chunk_fwd_o_update(*args, **kwargs):
            calls.append(("o_update", kwargs["chunk_offsets"]))
            return _DummyTensor("h_updated")

        monkeypatch.setattr(chunk, "chunk_fwd_o", fake_chunk_fwd_o)
        monkeypatch.setattr(chunk, "chunk_fwd_o_update", fake_chunk_fwd_o_update)

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


def test_build_chunk_meta_device_rejects_non_npu_input():
    cu_seqlens = torch.tensor([0, 4, 4, 12], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be on NPU"):
        build_chunk_meta_device(
            cu_seqlens=cu_seqlens,
            chunk_size=64,
            out_chunk_indices=torch.empty((2, 2), dtype=torch.int32),
        )


def test_build_chunk_offsets_falls_back_without_triton_kernel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gdn_chunk_meta, "_build_chunk_offsets_kernel", object())

    chunk_counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    chunk_offsets = torch.empty(4, dtype=torch.int32)
    update_chunk_offsets = torch.empty(4, dtype=torch.int32)

    gdn_chunk_meta._build_chunk_offsets(chunk_counts, chunk_offsets, add_one=0)
    gdn_chunk_meta._build_chunk_offsets(chunk_counts, update_chunk_offsets, add_one=1)

    assert torch.equal(chunk_offsets, torch.tensor([0, 2, 2, 5], dtype=torch.int32))
    assert torch.equal(update_chunk_offsets, torch.tensor([0, 3, 4, 8], dtype=torch.int32))


def test_build_chunk_offsets_does_not_launch_triton_prefix_sum_kernel(monkeypatch: pytest.MonkeyPatch):
    class _RaisingKernel:
        def __getitem__(self, grid):
            raise AssertionError("_build_chunk_offsets should use torch.cumsum instead of the Triton prefix-sum kernel")

    monkeypatch.setattr(gdn_chunk_meta, "_build_chunk_offsets_kernel", _RaisingKernel())

    chunk_counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    chunk_offsets = torch.empty(4, dtype=torch.int32)
    update_chunk_offsets = torch.empty(4, dtype=torch.int32)

    gdn_chunk_meta._build_chunk_offsets(chunk_counts, chunk_offsets, add_one=0)
    gdn_chunk_meta._build_chunk_offsets(chunk_counts, update_chunk_offsets, add_one=1)

    assert torch.equal(chunk_offsets, torch.tensor([0, 2, 2, 5], dtype=torch.int32))
    assert torch.equal(update_chunk_offsets, torch.tensor([0, 3, 4, 8], dtype=torch.int32))


def test_build_final_chunk_indices_falls_back_without_triton_kernel(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(gdn_chunk_meta, "_build_final_chunk_indices_kernel", object())

    chunk_counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    update_chunk_offsets = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
    out_final_chunk_indices = torch.empty(3, dtype=torch.int32)

    gdn_chunk_meta._build_final_chunk_indices(
        chunk_counts,
        update_chunk_offsets,
        out_final_chunk_indices,
    )

    assert torch.equal(out_final_chunk_indices, torch.tensor([2, 3, 7], dtype=torch.int32))
