# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

import vllm_ascend.patch.worker.patch_dflash_speculator  # noqa: F401
from vllm_ascend.worker.v2.sample.gumbel import _gumbel_sample
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


@triton.jit
def _stable_gumbel_kernel(uniform_ptr, output_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    uniform = tl.load(uniform_ptr + offsets, offsets < size, other=0.5)
    noise = -tl.log(-_log1p_neg_stable(uniform))
    tl.store(output_ptr + offsets, noise, offsets < size)


def test_stable_gumbel_matches_fp64_reference():
    """Compile the shared transform on NPU and check both tails and its branch."""
    uniform = torch.tensor(
        [2.0**-31, 2.0**-24, 1e-4, 0.24999999, 0.25, 0.25000003, 0.5, 1.0 - 2.0**-24],
        dtype=torch.float32,
    )
    reference = -torch.log(-torch.log1p(-uniform.double()))
    device_uniform = uniform.to("npu")
    output = torch.empty_like(device_uniform)
    _stable_gumbel_kernel[(1,)](device_uniform, output, uniform.numel(), BLOCK=16)
    actual = output.cpu().double()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=4e-6, rtol=2e-6)


@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mrope", [False, True])
def test_context_cache_contains_rotated_keys(monkeypatch, use_triton, is_neox_style, dtype, mrope):
    """Both NPU RoPE backends must write rotated context K, including key-only calls."""
    import vllm_ascend.ops.rotary_embedding as rope_module
    from vllm_ascend.patch.worker.patch_qwen3_dflash import precompute_and_store_context_kv

    monkeypatch.setattr(rope_module, "HAS_TRITON", use_triton)
    torch.manual_seed(7)
    layers, tokens, heads, head_dim = 2, 3, 2, 64
    keys = torch.randn(layers, tokens, heads, head_dim, dtype=dtype, device="npu")
    values = torch.randn_like(keys)
    positions = torch.tensor([2, 4, 7], device="npu")
    angles = torch.outer(torch.arange(16).float(), 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim))
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(device="npu", dtype=dtype)
    rope: rope_module.AscendMRotaryEmbedding | rope_module.AscendRotaryEmbedding
    if mrope:
        rope = rope_module.AscendMRotaryEmbedding.__new__(rope_module.AscendMRotaryEmbedding)
    else:
        rope = rope_module.AscendRotaryEmbedding.__new__(rope_module.AscendRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.head_size = head_dim
    rope.rotary_dim = head_dim
    rope.is_neox_style = is_neox_style
    rope.cos_sin_cache = cache
    if isinstance(rope, rope_module.AscendMRotaryEmbedding):
        rope.mrope_section = [11, 11, 10]
        rope.mrope_interleaved = True
    else:
        rope.use_mtp = False

    def rotate(pos, query, key):
        return rope.forward_oot(pos, query, key)

    caches = [SimpleNamespace(impl=MagicMock(), kv_cache=object()) for _ in range(layers)]
    model = SimpleNamespace(
        _num_attn_layers=layers,
        _kv_size=heads * head_dim,
        _head_dim=head_dim,
        _num_kv_heads=heads,
        _project_context_kv=lambda *args: (keys.clone(), values.clone()),
        layers=[SimpleNamespace(self_attn=SimpleNamespace(k_norm=torch.nn.Identity(), rotary_emb=rotate))] * layers,
        _attn_layers=caches,
    )
    mapping = torch.arange(tokens, dtype=torch.int32, device="npu")
    precompute_and_store_context_kv(model, torch.zeros(tokens, heads * head_dim, device="npu"), positions, mapping)

    original = keys.cpu().float()
    cos, sin = cache.cpu().float()[positions.cpu()].chunk(2, dim=-1)
    cos, sin = cos[None, :, None], sin[None, :, None]
    if is_neox_style:
        left, right = original.chunk(2, dim=-1)
        expected = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
    else:
        left, right = original[..., ::2], original[..., 1::2]
        expected = torch.stack((left * cos - right * sin, right * cos + left * sin), dim=-1).flatten(-2)
    for layer, attn in enumerate(caches):
        attn.impl.do_kv_cache_update.assert_called_once()
        args = attn.impl.do_kv_cache_update.call_args.args
        torch.testing.assert_close(args[1].cpu().float(), expected[layer], atol=0.03, rtol=0.02)
        torch.testing.assert_close(args[2], values[layer])


@pytest.mark.parametrize("probabilistic", [False, True])
def test_candidate_walk_preserves_conditional_scores_and_clears_old_cache(probabilistic):
    """Zero-temperature walks must preserve q along the chosen path for rejection."""
    max_reqs, num_reqs, steps, topk, vocab = 4, 3, 2, 4, 64
    device = "npu"
    sampler = CandidateSampler(max_reqs, steps, topk, torch.device(device))
    logits = torch.full((max_reqs, steps, vocab), -float("inf"), device=device) if probabilistic else None
    mapping = torch.tensor([[2, 2], [0, 0], [-1, -1]], dtype=torch.int32, device=device).flatten()
    positions = torch.full((num_reqs * steps,), 10, dtype=torch.int32, device=device)
    temperatures = torch.zeros(max_reqs, device=device)
    seeds = torch.arange(max_reqs, dtype=torch.int64, device=device)
    tokens = torch.full((num_reqs * steps,), -1, dtype=torch.int64, device=device)
    scores = torch.zeros(num_reqs, steps, topk, topk)
    scores[:, 0, 0] = torch.tensor([0.0, 2.0, 2.0, 1.0])  # Lowest index wins the tie.
    scores[:, 1, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])

    for offset in (1, 25):
        candidates = torch.arange(offset, offset + num_reqs * steps * topk).view(num_reqs, steps, topk)
        sampler.sample(
            candidates.to(device),
            scores.to(device),
            num_reqs,
            positions,
            mapping,
            temperatures,
            seeds,
            tokens,
            logits,
            use_fp64=False,
        )
        expected_tokens = torch.tensor(
            [candidates[0, 0, 1], candidates[0, 1, 3], candidates[1, 0, 1], candidates[1, 1, 3], -1, -1]
        )
        torch.testing.assert_close(tokens.cpu(), expected_tokens)
        if logits is not None:
            expected = torch.full((max_reqs, steps, vocab), -float("inf"))
            for row, state in ((0, 2), (1, 0)):
                for step, previous in ((0, 0), (1, 1)):
                    expected[state, step, candidates[row, step]] = scores[row, step, previous]
            torch.testing.assert_close(logits.cpu(), expected)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_probabilistic_candidate_walk_matches_proposal_distribution(temperature):
    """The NPU walk must draw from the same softmax q supplied to rejection."""
    num_reqs, steps, topk = 8192, 1, 4
    device = "npu"
    sampler = CandidateSampler(num_reqs, steps, topk, torch.device(device))
    unary = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    scores = unary.expand(num_reqs, steps, topk, topk).contiguous().to(device)
    candidates = torch.arange(topk).expand(num_reqs, steps, topk).contiguous().to(device)
    tokens = torch.empty(num_reqs, dtype=torch.int64, device=device)
    logits = torch.full((num_reqs, steps, topk), -float("inf"), device=device)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=device)
    sampler.sample(
        candidates,
        scores,
        num_reqs,
        torch.full((num_reqs,), 10, dtype=torch.int32, device=device),
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.full((num_reqs,), temperature, device=device),
        seeds,
        tokens,
        logits,
        use_fp64=False,
    )
    frequencies = torch.bincount(tokens.cpu(), minlength=topk).float() / num_reqs
    expected = (unary / temperature).softmax(dim=-1)
    torch.testing.assert_close(frequencies, expected, atol=0.025, rtol=0.0)
    torch.testing.assert_close(logits.cpu(), unary.expand(num_reqs, steps, topk))

    target_tokens = _gumbel_sample(
        unary.expand(num_reqs, topk).contiguous().to(device),
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.full((num_reqs,), temperature, device=device),
        seeds,
        torch.full((num_reqs,), 9, dtype=torch.int32, device=device),
        apply_temperature=True,
        is_drafting=False,
    )
    # Independent draws from the same q agree with probability sum(q**2).
    agreement = (tokens == target_tokens).float().mean().cpu()
    torch.testing.assert_close(agreement, expected.square().sum(), atol=0.025, rtol=0.0)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_probabilistic_candidate_walk_caches_realized_conditional_distribution(temperature):
    """Each second-step proposal and cache must depend on the sampled predecessor."""
    num_reqs, steps, topk = 8192, 2, 2
    device = torch.device("npu")
    sampler = CandidateSampler(num_reqs, steps, topk, device)
    start = torch.tensor([0.0, 0.5])
    transitions = torch.tensor([[1.0, -0.5], [-1.0, 0.5]])
    scores = torch.stack((start.expand(topk, topk), transitions)).expand(num_reqs, -1, -1, -1).contiguous()
    candidates = torch.tensor([[0, 1], [2, 3]]).expand(num_reqs, -1, -1).contiguous()
    tokens = torch.empty(num_reqs, steps, dtype=torch.int64, device=device)
    logits = torch.full((num_reqs, steps, 4), -float("inf"), device=device)
    sampler.sample(
        candidates.to(device),
        scores.to(device),
        num_reqs,
        torch.tensor([10, 11], device=device).repeat(num_reqs),
        torch.arange(num_reqs, dtype=torch.int32, device=device).repeat_interleave(steps),
        torch.full((num_reqs,), temperature, device=device),
        torch.arange(num_reqs, dtype=torch.int64, device=device),
        tokens,
        logits,
        use_fp64=False,
    )
    tokens = tokens.cpu()
    joint_index = tokens[:, 0] * topk + tokens[:, 1] - 2
    observed = torch.bincount(joint_index, minlength=topk * topk).float().view(topk, topk) / num_reqs
    expected = (start / temperature).softmax(-1)[:, None] * (transitions / temperature).softmax(-1)
    torch.testing.assert_close(observed, expected, atol=0.025, rtol=0.0)
    cached = logits.cpu()
    torch.testing.assert_close(cached[:, 0, :2], start.expand(num_reqs, -1))
    torch.testing.assert_close(cached[:, 1, 2:], transitions[tokens[:, 0]])
    assert torch.isneginf(cached[:, 0, 2:]).all()
    assert torch.isneginf(cached[:, 1, :2]).all()

    # Feed the actual walk and its sparse q into verification. A fixed target
    # makes every returned position's expected marginal known, including the
    # residual draw at rejection and the bonus after full acceptance.
    target = torch.tensor([0.2, 0.3, 0.1, 0.4], device=device)
    query_len = steps + 1
    sampled, num_sampled = rejection_sample(
        target_logits=target.log().expand(num_reqs * query_len, -1).contiguous(),
        draft_logits=logits,
        draft_sampled=torch.cat((torch.zeros(num_reqs, 1, dtype=torch.int64), tokens), dim=1).flatten().to(device),
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * query_len,
        pos=torch.tensor([9, 10, 11], dtype=torch.int32, device=device).repeat(num_reqs),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device).repeat_interleave(query_len),
        expanded_local_pos=torch.arange(query_len, dtype=torch.int32, device=device).repeat(num_reqs),
        temperature=torch.full((num_reqs,), temperature, device=device),
        seed=torch.arange(num_reqs, dtype=torch.int64, device=device),
        num_speculative_steps=steps,
    )
    sampled, num_sampled = sampled.cpu(), num_sampled.cpu()
    assert ((num_sampled >= 1) & (num_sampled <= query_len)).all()
    for step in range(query_len):
        returned = sampled[num_sampled > step, step]
        assert returned.numel() > 0
        observed = torch.bincount(returned, minlength=4).float() / returned.numel()
        tolerance = 5.0 * (0.25 / returned.numel()) ** 0.5
        torch.testing.assert_close(observed, target.cpu(), atol=tolerance, rtol=0.0)
