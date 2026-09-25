"""UT for AscendCompilationCounter (stage4 final-audit A6, U3/W4 promise).

Deterministic, CPU-only: a synthetic vllm compile-cache tree exercises every
D5 tri-state plus the parsing discipline (string/comment stripping,
default-codegen wrappers skipped, infra namespace ignored). The fallback
authority is injected so classification does not depend on triton_experimental
activation state.
"""

import os

import pytest

from tests.ut.compilation.inductor_track_counter import (
    AscendCompilationCounter,
    _classify,
)

_TE_WRAPPER = """# AOT ID: 0 (fake)
from torch_npu._inductor.triton_experimental import npu_triton_heuristics


@npu_triton_heuristics.pointwise(name="kernel_0")
def kernel_0():
    pass


@npu_triton_heuristics.pointwise(name="kernel_1")
def kernel_1():
    pass


def call():
    x = extern_kernels.mm(a, b)                    # aten.mm -> unclassified
    e = torch.ops.aten.embedding(weights, ids)     # -> fallback_listed
    m = torch.ops.vllm.moe_forward(q, k)           # -> opaque_custom
    r = torch.ops._C_ascend.npu_add_rms_norm(x, y) # -> opaque_custom
    _ = "extern_kernels.cosine_embedding(x, y)"    # string literal: not counted
    _ = torch.ops.inductor.cache_live(...)         # infra namespace: skipped
    # torch.ops.aten.multinomial(probs)            # comment: not counted
    return x, e, m, r
"""

_DEFAULT_WRAPPER = """# AOT ID: 1 (fake)
import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics


def call():
    return extern_kernels.mm(a, b)  # default-codegen piece: scanned, not TE-accounted
"""


@pytest.fixture()
def cache_tree(tmp_path):
    piece = tmp_path / "qwen" / "torch_compile_cache" / "qwen" / "rank_0" / "inductor_cache"
    te_dir = piece / "abc"
    te_dir.mkdir(parents=True)
    (te_dir / "output_code.py").write_text(_TE_WRAPPER, encoding="utf-8")
    default_dir = piece / "def0"
    default_dir.mkdir(parents=True)
    (default_dir / "output_code.py").write_text(_DEFAULT_WRAPPER, encoding="utf-8")
    # a non-wrapper .py (kernel copy starting with import) must be ignored
    ker_dir = piece / "ker0"
    ker_dir.mkdir(parents=True)
    (ker_dir / "kernel.py").write_text("import triton\n", encoding="utf-8")
    return tmp_path


def _authority() -> set[str]:
    # aten.embedding resolves to packet "aten.embedding"; authority entries are
    # overload-qualified strings — the packet + ".default" form must match.
    return {"aten.embedding.default"}


def test_tri_state_accounting(cache_tree):
    counter = AscendCompilationCounter.capture(
        cache_tree, read_metrics=False, fallback_authority=_authority()
    )
    # both wrappers scanned; only the TE one is accounted
    counter.expect(
        num_wrappers_scanned=2,
        num_triton_kernels=2,
        num_aclnn_fallbacks=1,   # aten.embedding
        num_opaque_custom=2,     # vllm.moe_forward + _C_ascend.npu_add_rms_norm
        num_unclassified=1,      # aten.mm (resolves, not in authority)
        num_authority_unavailable=0,
    )


def test_missing_cache_root_is_all_zero(tmp_path):
    counter = AscendCompilationCounter.capture(
        tmp_path / "nope", read_metrics=False, fallback_authority=set()
    )
    counter.expect(num_wrappers_scanned=0, num_aclnn_fallbacks=0, num_unclassified=0)


def test_authority_unavailable_state():
    assert _classify("aten.mm", None) == "authority_unavailable"
    assert _classify("vllm.moe_forward", None) == "opaque_custom"  # decided pre-authority


def test_unresolvable_core_namespace_is_unclassified():
    assert _classify("aten.no_such_op_xyz", {"aten.mm.default"}) == "unclassified"
    # torch.ops auto-creates unknown namespaces, so an unknown custom-namespace
    # op resolves to the namespace object and lands in unclassified (待查) —
    # same semantics as the offline reporter; never silently dropped.
    assert _classify("custom.foo", {"aten.mm.default"}) == "unclassified"


def test_expect_failure_messages(cache_tree):
    counter = AscendCompilationCounter.capture(
        cache_tree, read_metrics=False, fallback_authority=_authority()
    )
    with pytest.raises(AssertionError, match="num_unclassified not as expected"):
        counter.expect(num_unclassified=99)
    with pytest.raises(AssertionError, match="num_aclnn_fallbacks below threshold"):
        counter.expect_at_least(num_aclnn_fallbacks=2)
    # threshold form proves the scan found artifacts (gate usage)
    counter.expect_at_least(num_wrappers_scanned=1, num_triton_kernels=1)


def test_env_cache_root_default(monkeypatch, cache_tree, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(cache_tree))
    counter = AscendCompilationCounter.capture(read_metrics=False, fallback_authority=set())
    assert counter.num_wrappers_scanned == 2
    assert os.environ["VLLM_CACHE_ROOT"] == str(cache_tree)
