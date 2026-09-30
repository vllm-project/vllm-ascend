# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    declared_kwarg,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    resolve_bounded_replay,
    supports_bounded_replay,
)


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2


# --- the fields a hand-rebuilt spec has to carry -----------------------------


def _swa_spec(**overrides):
    return AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
        **overrides,
    )


@pytest.mark.skipif(
    not hasattr(SlidingWindowMLASpec, "bounded_replay"),
    reason="bounded_replay arrives with vLLM #56227. Until the Main2Main pin includes it "
    "no spec can set the field, so there is nothing for merge to carry.",
)
def test_sliding_window_mla_merge_carries_bounded_replay():
    """The Ascend override rebuilds the merged spec instead of calling the
    parent's ``merge``, so upstream's field has to be carried across by hand:
    dropping it would leave a multi-layer group replaying nothing while its
    layers asked to, and nothing would report it -- ``prefix_cacheable`` and
    ``prefix_replay_tokens`` are derived from the field, so the group would
    simply look like an ordinary sliding-window one."""
    merged = AscendSlidingWindowMLASpec.merge([_swa_spec(bounded_replay=True), _swa_spec(bounded_replay=True)])
    assert merged.bounded_replay
    assert not merged.prefix_cacheable

    assert not AscendSlidingWindowMLASpec.merge([_swa_spec(), _swa_spec()]).bounded_replay

    with pytest.raises(AssertionError, match="replay policy"):
        AscendSlidingWindowMLASpec.merge([_swa_spec(bounded_replay=True), _swa_spec(bounded_replay=False)])


def test_sliding_window_mla_merge_carries_the_retained_token_count():
    """The same failure mode as the replay flag, one field over: this override
    rebuilds the merged spec by hand, so anything it forgets is silently reset
    to its default -- here the window would come out narrower than the layers
    asked for, and the blocks of the extra trailing tokens would be dropped
    even though they are what the next request's hit resumes from."""
    if not declared_kwarg(AscendSlidingWindowMLASpec, "extra_retained_tokens", 8):
        pytest.skip("extra_retained_tokens is absent from this baseline's SlidingWindowSpec.")

    merged = AscendSlidingWindowMLASpec.merge([_swa_spec(extra_retained_tokens=8), _swa_spec(extra_retained_tokens=8)])

    assert merged.extra_retained_tokens == 8

    with pytest.raises(AssertionError, match="retained token count"):
        AscendSlidingWindowMLASpec.merge([_swa_spec(extra_retained_tokens=8), _swa_spec(extra_retained_tokens=4)])


# --- the field a pinned baseline may not have yet ----------------------------


class _WithoutTheField:
    def __init__(self, head_dim: int = 1):
        self.head_dim = head_dim


class _WithTheField:
    def __init__(self, head_dim: int = 1, bounded_replay: bool = False):
        self.head_dim = head_dim
        self.bounded_replay = bounded_replay


def test_the_kwarg_is_spelled_out_only_for_a_constructor_that_takes_it():
    """Naming it unconditionally is a ``TypeError`` on a baseline pinned before
    #56227, not a harmless extra, so the probe has to be the constructor's
    parameters rather than the caller's intent."""
    assert declared_kwarg(_WithoutTheField, "bounded_replay", True) == {}
    assert declared_kwarg(_WithTheField, "bounded_replay", True) == {"bounded_replay": True}
    assert declared_kwarg(_WithTheField, "bounded_replay", False) == {"bounded_replay": False}


def test_the_swa_spec_accepts_whatever_the_probe_returns():
    """Whatever the pinned vLLM takes, this is the call the merge path makes."""
    replaying = _swa_spec(**declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", True))
    plain = _swa_spec(**declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", False))

    if hasattr(SlidingWindowMLASpec, "bounded_replay"):
        assert replaying.bounded_replay
        assert not plain.bounded_replay


# --- which layers may declare a replay window --------------------------------


def _vllm_config(*, v2_runner: bool = False, pcp_size: int = 1, architectures=("DeepseekV41ForCausalLM",)):
    return SimpleNamespace(
        use_v2_model_runner=v2_runner,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        model_config=SimpleNamespace(architectures=list(architectures)),
    )


def _cache_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"swa_bounded_replay": True, **overrides})


def test_only_the_v4_1_target_may_replay():
    """The feature is a V4.1 one: upstream wires it into the V4.1 tree and that
    tree alone, picking it by architecture name. A V4 model has to come out off
    however the switch is set -- Ascend serves both from one model class."""
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]))
    assert supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]))
    assert supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41ForConditionalGeneration"]))


def test_the_v4_1_draft_is_not_recognised():
    """The DSpark draft is V4.1 too, but its SWA cache is a different class --
    ``DeepseekV41DSparkSWACache`` -- and that class does not carry the field yet,
    so a draft name here would open a gate onto nothing: the switch reads as on
    and no block replays. Left out until that class is wired as well."""
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41DSparkModel"]))
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DSparkV41DraftModel"]))
    assert not supports_bounded_replay(SimpleNamespace(architectures=["DeepseekV41DSparkModel", "DSparkV41DraftModel"]))


def test_a_model_without_architectures_may_not_replay():
    """Absent, empty or a non-list: none of them is the V4.1 name, and the
    feature must not be inherited by accident."""
    assert not supports_bounded_replay(SimpleNamespace())
    assert not supports_bounded_replay(SimpleNamespace(architectures=None))
    assert not supports_bounded_replay(SimpleNamespace(architectures=[]))
    assert not supports_bounded_replay(None)


def test_a_v4_layer_never_replays_however_the_switch_is_set():
    with patch("vllm_ascend.core.kv_cache_interface.logger.warning_once") as warning:
        assert not resolve_bounded_replay(_vllm_config(architectures=["DeepseekV4ForCausalLM"]), _cache_config())

    # Silent, like upstream: the V4 tree does not read the switch at all, so
    # there is nothing to explain to the user.
    warning.assert_not_called()


def test_a_baseline_without_the_switch_replays_nothing():
    """The switch arrives with #56227. Where it does not exist the feature is
    unreachable rather than off -- there is no flag to have turned on."""
    assert not resolve_bounded_replay(_vllm_config(), SimpleNamespace())
    assert not resolve_bounded_replay(_vllm_config(), None)
    assert not resolve_bounded_replay(_vllm_config(), _cache_config(swa_bounded_replay=False))


def test_the_upstream_switch_is_all_that_is_left():
    """A V4.1 layer on the V1 runner replays without being asked twice: the
    scheduler conditions above have already had their say, and there is no
    capability left for this side to withhold."""
    with patch("vllm_ascend.core.kv_cache_interface.logger.warning_once") as warning:
        assert resolve_bounded_replay(_vllm_config(), _cache_config())

    warning.assert_not_called()


def test_the_v2_runner_turns_the_switch_off():
    """Upstream's guard, inverted for Ascend: it turns the switch off because
    only its V2 runner skips the replayed writes, while here only the V1 runner
    pads them."""
    assert not resolve_bounded_replay(_vllm_config(v2_runner=True), _cache_config())


def test_the_v2_runner_says_why_it_is_not_v1():
    """The upstream wording would be a lie here -- it asks for V2 -- so the
    message has to name the runner Ascend actually pads with."""
    with patch("vllm_ascend.core.kv_cache_interface.logger.warning_once") as warning:
        resolve_bounded_replay(_vllm_config(v2_runner=True), _cache_config())

    (message,), _ = warning.call_args
    assert "model runner V1 on Ascend" in message
    assert "prefix caching" in message


def test_prefill_context_parallelism_turns_the_switch_off():
    assert not resolve_bounded_replay(_vllm_config(pcp_size=2), _cache_config())


def test_the_pcp_message_is_upstreams():
    with patch("vllm_ascend.core.kv_cache_interface.logger.warning_once") as warning:
        resolve_bounded_replay(_vllm_config(pcp_size=4), _cache_config())

    (message,), _ = warning.call_args
    assert "prefill context parallelism" in message
    assert "rank-local batch" in message


def test_a_layer_that_may_replay_warns_about_nothing():
    """A layer that is allowed to replay is the quiet case: whatever is said here
    is said about a run that will not replay. Only the conditions that turn the
    switch off speak."""
    with patch("vllm_ascend.core.kv_cache_interface.logger.warning_once") as warning:
        resolve_bounded_replay(_vllm_config(), _cache_config())

    warning.assert_not_called()
