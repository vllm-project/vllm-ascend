# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cover the fused KPool tail key/gate projection.

The argument-level cases run on CPU; the value-level case re-runs the Triton
cache write on a real device in a clean subprocess, because the CPU conftest
stubs ``torch.npu`` and the launch arguments alone cannot catch a wrong row
stride.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_ascend.attention.indexer_kpool as backend_module
import vllm_ascend.ops.triton.glm5_next_cann_keypool as keypool_module
from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile

HEAD_DIM = 4
HIDDEN = 8
LAYERNORM_EPS = 1e-6


def _backend(monkeypatch, fused: bool) -> Glm5NextKPoolIndexerBackend:
    profile = get_hardware_profile(AscendDeviceType.A5)
    monkeypatch.setattr(backend_module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(backend_module, "get_ascend_config", lambda: SimpleNamespace(enable_indexer_tail_kg_cat=fused))
    monkeypatch.setattr(Glm5NextKPoolIndexerBackend, "_init_cann_ops", lambda self: None)
    torch.manual_seed(0)
    source = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
        ),
        n_head=2,
        head_dim=HEAD_DIM,
        topk_tokens=8,
        q_lora_rank=4,
        index_kpool=4,
        wq_b=nn.Linear(4, 2 * HEAD_DIM, bias=False),
        wk_weights_proj=nn.Linear(HIDDEN, 2 * HEAD_DIM, bias=False),
        k_norm=nn.LayerNorm(HEAD_DIM, eps=LAYERNORM_EPS, dtype=torch.float32),
        softmax_scale=0.5,
        index_kpool_compress_ape=nn.Parameter(torch.zeros(4, HEAD_DIM)),
        index_kpool_compress_gate=nn.Parameter(torch.randn(HEAD_DIM, HIDDEN)),
        k_cache=SimpleNamespace(prefix="indexer.k_cache", dtype=torch.bfloat16),
        tail_cache=SimpleNamespace(prefix="indexer.tail"),
        topk_indices_buffer=None,
    )
    backend = Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)
    backend.process_weights_after_loading()
    return backend


def test_flag_off_keeps_two_tail_projections(monkeypatch):
    backend = _backend(monkeypatch, fused=False)
    assert backend._kg_weight is None
    tail_hidden = torch.randn(6, HIDDEN)
    norm_weight, norm_bias = backend.k_norm.weight, backend.k_norm.bias

    tail_key, tail_gate = backend._project_tail_key_gate(tail_hidden, norm_weight, norm_bias, LAYERNORM_EPS)

    expected_key = torch.nn.functional.linear(tail_hidden, backend.wk_weights_proj.weight[:HEAD_DIM]).float()
    expected_key = torch.nn.functional.layer_norm(
        expected_key, (HEAD_DIM,), norm_weight, norm_bias, LAYERNORM_EPS
    )
    expected_gate = torch.nn.functional.linear(tail_hidden, backend.index_kpool_compress_gate).float()
    torch.testing.assert_close(tail_key, expected_key)
    torch.testing.assert_close(tail_gate, expected_gate)
    assert tail_key.data_ptr() != tail_gate.data_ptr()


def test_fused_single_gemm_matches_two_projection_reference(monkeypatch):
    backend = _backend(monkeypatch, fused=True)
    assert backend._kg_weight is not None
    assert backend._kg_weight.shape == (2 * HEAD_DIM, HIDDEN)
    tail_hidden = torch.randn(6, HIDDEN)

    reference_linear = torch.nn.functional.linear
    calls = []

    def counting_linear(input, weight, bias=None):
        calls.append(weight)
        return reference_linear(input, weight, bias)

    monkeypatch.setattr(backend_module.F, "linear", counting_linear)
    tail_key, tail_gate = backend._project_tail_key_gate(tail_hidden, None, None, LAYERNORM_EPS)

    assert len(calls) == 1
    assert calls[0] is backend._kg_weight
    assert tail_key is not tail_gate
    assert tail_key.shape == (6, 2 * HEAD_DIM)
    assert tail_key.is_contiguous()
    # The tail write adds `cols - head_dim` to the gate pointer, so the gate
    # must start at the second half of the fused buffer.
    assert tail_gate.data_ptr() == tail_key.data_ptr() + HEAD_DIM * tail_key.element_size()
    torch.testing.assert_close(tail_gate, tail_key[:, HEAD_DIM:])
    assert tail_gate.stride(0) == 2 * HEAD_DIM
    expected = torch.cat(
        (
            reference_linear(tail_hidden, backend.wk_weights_proj.weight[:HEAD_DIM]),
            reference_linear(tail_hidden, backend.index_kpool_compress_gate),
        ),
        dim=-1,
    ).float()
    torch.testing.assert_close(tail_key, expected)


def test_fused_with_norm_returns_contiguous_key_and_strided_gate(monkeypatch):
    backend = _backend(monkeypatch, fused=True)
    tail_hidden = torch.randn(6, HIDDEN)
    norm_weight, norm_bias = backend.k_norm.weight, backend.k_norm.bias

    tail_key, tail_gate = backend._project_tail_key_gate(tail_hidden, norm_weight, norm_bias, LAYERNORM_EPS)

    fused = torch.nn.functional.linear(tail_hidden, backend._kg_weight).float()
    expected_key = torch.nn.functional.layer_norm(
        fused[:, :HEAD_DIM].contiguous(), (HEAD_DIM,), norm_weight, norm_bias, LAYERNORM_EPS
    )
    torch.testing.assert_close(tail_key, expected_key)
    torch.testing.assert_close(tail_gate, fused[:, HEAD_DIM:])
    assert tail_key.is_contiguous() and tail_key.stride(0) == HEAD_DIM
    # The gate stays a strided view of the fused buffer instead of a copy.
    assert tail_gate.stride(0) == 2 * HEAD_DIM
    assert not tail_gate.is_contiguous()


def test_reload_refreshes_fused_weight_in_place(monkeypatch):
    backend = _backend(monkeypatch, fused=True)
    captured = backend._kg_weight
    assert captured is not None
    assert "_kg_weight" not in dict(backend.named_buffers(recurse=False))
    assert "_kg_weight" not in dict(backend.named_parameters(recurse=False))
    pointer = captured.data_ptr()

    for value, replace_storage in ((1.0, False), (2.0, True)):
        key_rows = backend.wk_weights_proj.weight[:HEAD_DIM]
        gate = backend.index_kpool_compress_gate
        if replace_storage:
            backend.wk_weights_proj.weight.data = torch.full_like(backend.wk_weights_proj.weight, value)
            gate.data = torch.full_like(gate, value + 1)
        else:
            key_rows.copy_(torch.full_like(key_rows, value))
            gate.copy_(torch.full_like(gate, value + 1))
        backend.process_weights_after_loading()
        assert backend._kg_weight is captured
        assert captured.data_ptr() == pointer
        torch.testing.assert_close(
            captured,
            torch.cat((backend.wk_weights_proj.weight[:HEAD_DIM].detach(), gate.detach()), dim=0),
        )


class _KernelRecorder:
    def __init__(self):
        self.args = None
        self.grid = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args):
            self.args = args

        return launch


def _tail_metadata(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        cann_inputs=SimpleNamespace(
            write_indices=torch.zeros(rows, 1, dtype=torch.int64),
            state_indices=torch.zeros(rows, 1, dtype=torch.int64),
            page_rows=1,
        )
    )


@pytest.mark.parametrize("norm", [False, True])
def test_write_keypool_caches_forwards_tail_row_strides(monkeypatch, norm):
    recorder = _KernelRecorder()
    monkeypatch.setattr(keypool_module, "_write_keypool_caches_kernel", recorder)
    rows, blocks, capacity, block_size, head_dim = 3, 2, 4, 8, 128
    tail = torch.zeros(blocks, capacity, 2 * head_dim, dtype=torch.float32)
    cache = torch.zeros(blocks, block_size, head_dim, dtype=torch.bfloat16)
    fused = torch.zeros(rows, 2 * head_dim, dtype=torch.float32)
    if norm:
        # Layer-normed key: contiguous [N, head_dim]. Gate: tail of the same
        # fused buffer, still strided by 2 * head_dim.
        key, gate = fused[:, :head_dim].contiguous(), fused[:, head_dim:]
        expected = (head_dim, 2 * head_dim)
    else:
        # Both halves are views of the fused buffer; the gate pointer is
        # advanced by head_dim because the kernel adds `cols - head_dim`.
        key, gate = fused, fused[:, head_dim:]
        expected = (2 * head_dim, 2 * head_dim)
    metadata = SimpleNamespace(pooled_slot_mapping=torch.zeros(0, dtype=torch.int64))

    keypool_module.write_keypool_caches(
        tail, cache, key, gate, torch.zeros(0, head_dim), metadata, _tail_metadata(rows), torch.zeros(0, 2 * head_dim)
    )

    assert recorder.args[1] is cache
    assert recorder.args[12] == head_dim
    assert recorder.args[-2:] == expected


def test_write_keypool_caches_without_tail_projection_keeps_cache_width(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(keypool_module, "_write_keypool_caches_kernel", recorder)
    head_dim = 128
    cache = torch.zeros(2, 8, head_dim, dtype=torch.bfloat16)
    metadata = SimpleNamespace(pooled_slot_mapping=torch.zeros(0, dtype=torch.int64))

    keypool_module.write_keypool_caches(
        torch.zeros(2, 8, 2 * head_dim, dtype=torch.float32),
        cache,
        None,
        None,
        torch.zeros(0, head_dim),
        metadata,
        _tail_metadata(1),
        torch.zeros(0, 2 * head_dim),
    )

    assert recorder.args[-2:] == (head_dim, head_dim)


_CHILD_EXIT_NO_DEVICE = 4

_CHILD_SNIPPET = f"""
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit({_CHILD_EXIT_NO_DEVICE})
if not torch.npu.is_available():
    raise SystemExit({_CHILD_EXIT_NO_DEVICE})

import torch.nn.functional as F
from torch import nn
from types import SimpleNamespace

from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

head_dim = 128
hidden = 64
rows = 2
device = "npu"

torch.manual_seed(0)
key_weight = torch.randn(head_dim, hidden, dtype=torch.bfloat16, device=device)
gate_weight = torch.randn(head_dim, hidden, dtype=torch.bfloat16, device=device)
head_weight = torch.zeros(head_dim, hidden, dtype=torch.bfloat16, device=device)
tail_hidden = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)


def make_backend(fused):
    module = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(module)
    module.head_dim = head_dim
    module.wk_weights_proj = nn.Linear(hidden, 2 * head_dim, bias=False)
    module.wk_weights_proj.weight.data = torch.cat((key_weight, head_weight), dim=0)
    module.index_kpool_compress_gate = nn.Parameter(gate_weight)
    module._kg_weight = torch.cat((key_weight, gate_weight), dim=0).contiguous() if fused else None
    return module


def write(tail_key, tail_gate, dest):
    tail = torch.zeros(2, 4, 2 * head_dim, dtype=torch.float32, device=device)
    metadata = SimpleNamespace(pooled_slot_mapping=torch.zeros(0, dtype=torch.int64, device=device))
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(
            write_indices=torch.tensor([[dest]], dtype=torch.int64, device=device),
            state_indices=torch.zeros(1, 1, dtype=torch.int64, device=device),
            page_rows=1,
        )
    )
    write_keypool_caches(
        tail,
        torch.zeros(2, 8, head_dim, dtype=torch.bfloat16, device=device),
        tail_key,
        tail_gate,
        torch.zeros(0, head_dim, dtype=torch.bfloat16, device=device),
        metadata,
        tail_metadata,
        torch.zeros(0, 2 * head_dim, dtype=torch.float32, device=device),
    )
    torch.npu.synchronize()
    return tail.reshape(-1)[dest * 2 * head_dim : (dest + 1) * 2 * head_dim]


off_key, off_gate = make_backend(False)._project_tail_key_gate(tail_hidden, None, None, 1e-6)
on_key, on_gate = make_backend(True)._project_tail_key_gate(tail_hidden, None, None, 1e-6)
assert not torch.equal(on_key, on_gate), "distinct key and gate values are required"

off_row = write(off_key, off_gate, 0)
on_row = write(on_key, on_gate, 1)

# Baseline: the two-projection layout lands key then gate in the slot.
torch.testing.assert_close(off_row, torch.cat((off_key, off_gate), dim=1).float(), atol=0, rtol=0)
# Fused layout: the same values, read through the fused row stride. A wrong
# stride or a gate pointer at the row start would put key values here.
torch.testing.assert_close(on_row, torch.cat((on_key, on_gate), dim=1).float(), atol=0, rtol=0)
torch.testing.assert_close(on_row, off_row, atol=1e-2, rtol=1e-2)

# With a norm the key becomes a separate contiguous tensor while the gate
# stays a strided view into the second half of the fused buffer.
norm_weight = torch.ones(head_dim, dtype=torch.float32, device=device)
norm_bias = torch.zeros(head_dim, dtype=torch.float32, device=device)
norm_key, norm_gate = make_backend(True)._project_tail_key_gate(tail_hidden, norm_weight, norm_bias, 1e-6)
norm_row = write(norm_key, norm_gate, 0)
torch.testing.assert_close(norm_row[head_dim:], on_gate.float(), atol=0, rtol=0)
torch.testing.assert_close(
    norm_row[:head_dim],
    F.layer_norm(on_key, (head_dim,), norm_weight, norm_bias, 1e-6),
    atol=0,
    rtol=0,
)
print("CHILD-OK")
"""


def test_fused_tail_write_matches_two_projection_tail_write():
    """Value-level on/off comparison of the changed stride path on a device."""
    torch_module = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch_module.npu.is_available():
        pytest.skip("requires a real Ascend NPU")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in (repo_root, env.get("PYTHONPATH")) if path)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SNIPPET],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("KPool tail cache write timed out")

    if proc.returncode == _CHILD_EXIT_NO_DEVICE:
        pytest.skip("no usable Ascend NPU in a clean subprocess")
    assert proc.returncode == 0 and "CHILD-OK" in proc.stdout, proc.stderr[-2000:]
