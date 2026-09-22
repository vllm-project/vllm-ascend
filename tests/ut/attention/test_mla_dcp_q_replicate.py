# SPDX-License-Identifier: Apache-2.0
"""Check replicated query geometry against independent TP-sharded projections."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_impl():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/mla_cp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMlaDCPImpl")
    impl.bases = []
    impl.body = [
        n
        for n in impl.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"dcp_q_replicate", "_project_query", "_q_proj_and_k_up_proj", "reorg_decode_q"}
    ]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[impl], type_ignores=[]), str(path), "exec"), scope)
    return scope["AscendMlaDCPImpl"]


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_replicated_absorption_matches_sharded_projection(dcp_size, tokens):
    torch.manual_seed(42)
    heads, nope, rope, latent, hidden = 3, 8, 4, 16, 12
    q_weight = torch.randn(dcp_size * heads * (nope + rope), hidden)
    k_weight = torch.randn(dcp_size * heads, nope, latent)
    x = torch.randn(tokens, hidden)
    expected_abs, expected_pe = [], []
    for rank in range(dcp_size):
        wq = q_weight.chunk(dcp_size)[rank]
        q = (x @ wq.T).view(tokens, heads, nope + rope)
        wk = k_weight.chunk(dcp_size)[rank]
        expected_abs.append(torch.einsum("thd,hdl->thl", q[..., :nope], wk))
        expected_pe.append(q[..., nope:])
    expected_abs = torch.cat(expected_abs, dim=1)
    expected_pe = torch.cat(expected_pe, dim=1)

    class Projection:
        qrep_active = True

        def __call__(self, value):
            return value @ q_weight.T, None

        def _local_view(self, value):
            return value[:, rank * heads : (rank + 1) * heads].contiguous()

    impl = load_impl()()
    impl.num_heads, impl.dcp_size = heads, dcp_size
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = nope, rope
    impl.qk_head_dim = nope + rope
    impl.q_proj = Projection()
    impl.dcp_W_UK_T = k_weight
    impl._dcp_all_gather_fragments = lambda *args, **kwargs: pytest.fail("runtime Q all-gather")
    actual_abs, actual_pe = impl._q_proj_and_k_up_proj(x)
    actual_abs, actual_pe = impl.reorg_decode_q(actual_abs, actual_pe)
    torch.testing.assert_close(actual_abs, expected_abs)
    torch.testing.assert_close(actual_pe, expected_pe)
    for rank in range(dcp_size):
        local = impl._project_query(x, local_heads=True)
        expected = (x @ q_weight.chunk(dcp_size)[rank].T).view(tokens, heads, nope + rope)
        torch.testing.assert_close(local, expected)


def test_replicated_prolog_uses_group_k_up_and_keeps_v_up_local():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/mla_cp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMlaDCPImpl")
    cls.bases = [ast.copy_location(ast.Name(id="Base", ctx=ast.Load()), cls.bases[0])]
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in {"dcp_q_replicate", "process_weights_after_loading"}
    ]
    local_k = torch.randn(12, 8, 16)
    local_v = torch.randn(12, 16, 8)
    group_k = torch.randn(96, 8, 16)

    class Base:
        def process_weights_after_loading(self, act_dtype):
            self.W_UK_T, self.W_UV = local_k, local_v
            self.mlapo_W_UK_T = local_k

    scope = dict(
        Base=Base,
        torch=torch,
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        torch_npu=SimpleNamespace(npu_format_cast=lambda x, _: x),
        ACL_FORMAT_FRACTAL_ND=2,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    impl = scope[cls.name]()
    impl.q_proj = SimpleNamespace(qrep_active=True)
    impl.num_heads, impl.dcp_size, impl.mlapo_num_heads = 12, 8, 12
    impl.enable_mlapo, impl.fa_quant_layer = True, False
    impl._dcp_all_gather = lambda weight, dim: group_k.clone()
    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.mlapo_num_heads == 96
    assert impl.W_UV is local_v and impl.W_UK_T is local_k
    assert impl.mlapo_W_UK_T is impl.dcp_W_UK_T
    torch.testing.assert_close(impl.mlapo_W_UK_T, group_k)
    address = impl.dcp_W_UK_T.data_ptr()
    group_k.add_(1)
    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.dcp_W_UK_T.data_ptr() == address
    torch.testing.assert_close(impl.mlapo_W_UK_T, group_k)
