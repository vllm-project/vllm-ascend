# SPDX-License-Identifier: Apache-2.0
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.test_qwen4_exp_cache_config import _allocate_mrv2_kv_cache, _groups
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.models.qwen4_exp import gbsa, qsa
from vllm_ascend.models.qwen4_exp.cache_config import QSAMXFP8Spec, get_qwen4_exp_kv_cache_config


@pytest.mark.parametrize("head_dim", [192, 256])
def test_mxfp8_scale_regions_and_views(monkeypatch, head_dim):
    groups = _groups()
    source = groups[0].kv_cache_spec
    args = {f.name: getattr(source, f.name) for f in fields(source)}
    args.update(
        dtype=torch.float8_e4m3fn,
        head_size=head_dim,
        head_size_v=head_dim,
        num_head_slots=2,
        state_content_bytes=head_dim + head_dim // 32,
    )
    groups[0].kv_cache_spec = QSAMXFP8Spec(**args)
    if head_dim == 256:
        # SSM fits a whole K/V pair but not one dense V manager page.
        groups[3].kv_cache_spec = replace(groups[3].kv_cache_spec, shapes=((64, 4), (384, 128)))
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        groups,
        10**9,
    )
    raw, caches, layout = _allocate_mrv2_kv_cache(config, monkeypatch)
    for name in groups[0].layer_names:
        k, v, ks, vs = caches[name]
        assert k.dtype == v.dtype == torch.float8_e4m3fn
        assert ks.shape == (6, 128, 1, head_dim // 32)
        assert vs.shape == (6, 2, 1, head_dim * 2)
        assert ks.is_contiguous() and vs.is_contiguous()
        assert k.is_contiguous() and v.is_contiguous()
        assert v.data_ptr() >= k.data_ptr() + k.numel()
        assert ks.data_ptr() >= v.data_ptr() + v.numel()
        assert vs.data_ptr() >= ks.data_ptr() + ks.numel()
        ks.fill_(123)
        vs.fill_(125)
        assert torch.all(ks == 123) and torch.all(vs == 125)
    name = groups[0].layer_names[0]
    _, receiver, _ = _allocate_mrv2_kv_cache(config, monkeypatch)
    conv, ssm = receiver["model.layers.0.linear_attn"]
    conv[0].fill_(7)
    ssm[0].fill_(9)
    # Transfer manager block 0 to block 2, each containing two kernel pages.
    for tag, (src, dst) in enumerate(zip(caches[name], receiver[name], strict=True), start=1):
        shape = MooncakeBaseConnectorWorker._get_transfer_block_shape(groups[0].kv_cache_spec, src)
        page_bytes = src[0].numel() * src.element_size()
        assert shape[0] == 1 and page_bytes == src.stride(0) * src.element_size()
        local, remote, lengths = [], [], []
        MooncakePullRecvingThread._append_block_transfer_addresses(
            [[4, 5]],
            [[0, 1]],
            dst.data_ptr(),
            src.data_ptr(),
            dst.stride(0) * dst.element_size(),
            page_bytes,
            page_bytes,
            local,
            remote,
            lengths,
        )
        assert local == [dst[4].data_ptr()] and remote == [src[0].data_ptr()]
        assert lengths == [2 * page_bytes]
        src[:2].view(torch.uint8).fill_(tag)
        dst[4:6].view(torch.uint8).copy_(src[:2].view(torch.uint8))
        assert torch.all(dst[4:6].view(torch.uint8) == tag)
    assert torch.all(conv[0] == 7) and torch.all(ssm[0] == 9)
    assert layout.region("qsa_k_scale").page_size_bytes == 256 * head_dim // 32


def test_missing_static_v_scale_is_rejected(monkeypatch):
    monkeypatch.setattr(qsa.upstream_qsa.Qwen4ExpQSAAttention, "process_weights_after_loading", lambda self: None)
    owner = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.kv_cache_dtype = "mxfp8"
    owner._qsa_model_dtype = torch.bfloat16
    owner.attn = torch.nn.Module()
    owner.attn.register_parameter(
        "v_cache_scale", torch.nn.Parameter(torch.full((256,), 255, dtype=torch.uint8), requires_grad=False)
    )
    with pytest.raises(ValueError, match="static E8M0"):
        owner.process_weights_after_loading()
    owner.attn.v_cache_scale.data.fill_(127)
    owner.process_weights_after_loading()
    assert torch.all(owner.v_cache_scale_reciprocal == 1)
    owner.num_kv_heads, owner.head_dim = 1, 256
    caches = (
        torch.empty(2, 128, 1, 256),
        torch.empty(2, 128, 1, 256),
        torch.empty(2, 128, 1, 8, dtype=torch.uint8),
        torch.empty(2, 2, 1, 512, dtype=torch.uint8),
    )
    owner.bind_kv_cache(caches)
    assert torch.all(caches[3] == 127)


@pytest.mark.parametrize("dim", [192, 256])
def test_gbsa_reassigned_page_restores_channel_scales(dim):
    from vllm_ascend.attention.attention_c8_mxfp import fill_mxfp_v_scale_cache
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    from vllm_ascend.worker.v2.utils import C8KVBlockZeroer

    owner = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.head_dim, owner.num_kv_heads, owner.kv_cache_dtype = dim, 1, "mxfp8"
    owner.attn = torch.nn.Module()
    owner.attn.v_cache_scale = (torch.arange(dim) % 64 + 80).byte()
    cache = torch.empty(6, 2, 1, dim * 2, dtype=torch.uint8)
    owner.bind_kv_cache((None, None, None, cache))
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(num_blocks=3),
        compilation_config=SimpleNamespace(static_forward_context={"gbsa": owner}),
    )
    NPUModelRunner._fill_c8_mxfp_v_scale_caches(runner)
    expected = cache.clone()
    cache.zero_()
    expected[:2].zero_()
    expected[4:].zero_()
    C8KVBlockZeroer(None, runner._c8_mxfp_v_scale_caches, fill_mxfp_v_scale_cache).zero_block_ids([1])
    assert torch.equal(cache, expected)


def test_mxfp8_operator_attributes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gbsa,
        "prepare_qsa_gbsa_inputs",
        lambda *args: (
            torch.zeros(1, 1, 1, dtype=torch.int32),
            torch.ones(1, 1, dtype=torch.int32),
            torch.tensor([4]),
            torch.tensor([0, 1]),
            args[4],
        ),
    )
    monkeypatch.setattr(
        gbsa, "generic_block_sparse_attention_metadata", lambda *a, **kw: calls.append(kw) or torch.empty(0)
    )
    monkeypatch.setattr(
        gbsa,
        "generic_block_sparse_attention",
        lambda *a, **kw: (calls.append(kw) or torch.zeros(1, 12, 256, dtype=torch.bfloat16), torch.empty(0)),
    )
    q = torch.empty(1, 12, 256, dtype=torch.float8_e4m3fn)
    k = torch.empty(2, 128, 1, 256, dtype=torch.float8_e4m3fn)
    qs = torch.empty(1, 12, 4, 2, dtype=torch.float8_e8m0fnu)
    ks = torch.empty(2, 128, 1, 4, 2, dtype=torch.float8_e8m0fnu)
    vs = torch.empty(2, 2, 1, 256, 2, dtype=torch.float8_e8m0fnu)
    gbsa.qsa_gbsa_attention(
        q,
        k,
        k,
        torch.zeros(1, 515, dtype=torch.int32),
        torch.tensor([[0, 1]]),
        torch.tensor([0]),
        torch.tensor([3]),
        torch.tensor([4]),
        torch.empty(1, 12, 256, dtype=torch.bfloat16),
        q_scale=qs,
        k_scale=ks,
        v_scale=vs,
    )
    assert all(c["quant_mode"] == 6 and c["mask_mode"] == 1 for c in calls)
    assert calls[1]["attention_out_dtype"] == torch.bfloat16
    assert calls[1]["k_dequant_scale"] is ks


@pytest.mark.parametrize(
    ("prefix", "initial_scale"), [("model.layers.3.self_attn", 255), ("mtp.layers.48.self_attn", 127)]
)
def test_owner_keeps_indexer_bf16_without_mutating_config(monkeypatch, prefix, initial_scale):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="mxfp8"), model_config=SimpleNamespace(dtype=torch.bfloat16)
    )
    seen = []
    monkeypatch.setattr(qsa.upstream_qsa, "qsa_kv_cache_dtype", lambda cache, prefix: cache.cache_dtype)

    def construct(self, **kwargs):
        torch.nn.Module.__init__(self)
        seen.append(kwargs["vllm_config"].cache_config.cache_dtype)
        self.indexer = SimpleNamespace(raw_key_cache=SimpleNamespace(), compressed_key_cache=SimpleNamespace())
        self.num_kv_heads, self.head_dim = 1, 256
        self.impl = SimpleNamespace(kv_cache_dtype="auto")

    monkeypatch.setattr(qsa.upstream_qsa.Qwen4ExpQSAAttention, "__init__", construct)
    owner = qsa.AscendQwen4ExpQSAAttention(
        vllm_config=config, config=SimpleNamespace(head_dim=256), layer_id=3, prefix=prefix
    )
    assert torch.all(owner.attn.v_cache_scale == initial_scale)
    assert seen == ["auto"] and config.cache_config.cache_dtype == "mxfp8"
    config.cache_config.block_size = 3200
    assert owner.indexer.raw_key_cache.cache_config.block_size == 3200
    assert owner.indexer.compressed_key_cache.cache_config.block_size == 3200
    assert owner.impl.kv_cache_dtype == "mxfp8"
    assert "attn.v_cache_scale" in dict(owner.named_parameters())
    assert not owner.use_fused_qsa_prepare


def test_fp8_cache_write_copies_encoded_bytes(monkeypatch):
    seen = []
    monkeypatch.setattr(qsa, "qsa_store_cache_rows_fused", lambda cache, slots, rows: seen.append((cache, rows)))
    cache = torch.empty(2, 128, 1, 256, dtype=torch.float8_e4m3fn)
    rows = torch.ones(2, 1, 256).to(torch.float8_e4m3fn)
    qsa._store_qsa_cache_rows(cache, torch.tensor([0, 1]), rows)
    assert seen[0][0].dtype == seen[0][1].dtype == torch.uint8
    assert seen[0][0].data_ptr() == cache.data_ptr()
    assert torch.equal(seen[0][1], rows.view(torch.uint8))
