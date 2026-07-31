#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_qwen3_dflash import apply_context_rope

PATCH_MOD = "vllm_ascend.patch.worker.patch_qwen3_dflash"

NUM_LAYERS = 5  # both target checkpoints have num_hidden_layers == 5
NUM_CTX = 6
NUM_KV_HEADS = 4
HEAD_DIM = 8
KV_SIZE = NUM_KV_HEADS * HEAD_DIM


class OutOfPlaceRope:
    """310P-style rotary: returns new tensors and never writes through its input.

    Marks its output so a caller that drops the return value is caught.
    """

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.calls = []

    def __call__(self, positions, query, key):
        self.calls.append(int(positions.shape[0]))
        return torch.full_like(query, float(self.layer_idx + 1)), torch.empty_like(key)


class InPlaceRope:
    """A2/A3-style rotary: writes through its input and returns that same storage."""

    def __init__(self):
        self.calls = []

    def __call__(self, positions, query, key):
        self.calls.append(int(positions.shape[0]))
        query.mul_(-1.0)
        return query, key


def make_layers(rope_factory):
    return [SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rope_factory(i))) for i in range(NUM_LAYERS)]


def make_k_normed():
    # Distinct per-layer values so a layer mix-up is visible.
    base = torch.arange(NUM_LAYERS * NUM_CTX * KV_SIZE, dtype=torch.float32)
    return base.reshape(NUM_LAYERS, NUM_CTX, NUM_KV_HEADS, HEAD_DIM).clone()


def run(layers, all_k_normed, positions=None):
    return apply_context_rope(
        layers=layers,
        all_k_normed=all_k_normed,
        context_positions=(positions if positions is not None else torch.arange(NUM_CTX, dtype=torch.int32)),
        num_layers=NUM_LAYERS,
        num_ctx=NUM_CTX,
        kv_size=KV_SIZE,
    )


class TestApplyContextRope310P(TestBase):
    def test_rotates_one_layer_at_a_time(self):
        """Each layer gets num_ctx positions, never L * num_ctx.

        The fused form overflows the 310P global cos/sin buffer, which is sized
        max_num_batched_tokens, for any realistic context with 5 layers.
        """
        layers = make_layers(OutOfPlaceRope)
        with mock_patch(f"{PATCH_MOD}.is_310p", return_value=True):
            run(layers, make_k_normed())

        for i, layer in enumerate(layers):
            self.assertEqual(
                layer.self_attn.rotary_emb.calls,
                [NUM_CTX],
                f"layer {i} was not rotated exactly once with num_ctx positions",
            )

    def test_uses_the_rotated_result_not_the_input(self):
        """310P rotary is out of place, so dropping the return value would feed
        unrotated K into the cache -- wrong numbers, no error."""
        layers = make_layers(OutOfPlaceRope)
        with mock_patch(f"{PATCH_MOD}.is_310p", return_value=True):
            out = run(layers, make_k_normed())

        self.assertEqual(tuple(out.shape), (NUM_LAYERS * NUM_CTX, KV_SIZE))
        per_layer = out.view(NUM_LAYERS, NUM_CTX, KV_SIZE)
        for i in range(NUM_LAYERS):
            # Each stub returns its own marker value; seeing it proves that
            # layer's result was kept and landed in that layer's slice.
            self.assertTrue(
                bool((per_layer[i] == float(i + 1)).all()),
                f"layer {i} slice does not hold that layer's rotated output",
            )

    def test_does_not_pass_repeated_positions(self):
        positions = torch.arange(NUM_CTX, dtype=torch.int32)
        layers = make_layers(OutOfPlaceRope)
        with mock_patch(f"{PATCH_MOD}.is_310p", return_value=True):
            run(layers, make_k_normed(), positions=positions)

        for layer in layers:
            self.assertNotEqual(
                layer.self_attn.rotary_emb.calls,
                [NUM_LAYERS * NUM_CTX],
                "positions were repeated across layers; that is the fused form",
            )

    def test_non_310p_keeps_the_single_fused_call(self):
        layers = make_layers(lambda _i: InPlaceRope())
        with mock_patch(f"{PATCH_MOD}.is_310p", return_value=False):
            out = run(layers, make_k_normed())

        self.assertEqual(
            layers[0].self_attn.rotary_emb.calls,
            [NUM_LAYERS * NUM_CTX],
            "non-310P must keep rotating all layers in one fused call",
        )
        for layer in layers[1:]:
            self.assertEqual(layer.self_attn.rotary_emb.calls, [], "only layer 0 drives the fused call")
        self.assertEqual(tuple(out.shape), (NUM_LAYERS * NUM_CTX, KV_SIZE))

    def test_non_310p_result_reflects_the_rotation(self):
        """In-place rotary returns its input storage, so taking the return value
        is a no-op there -- but it must still carry the rotation."""
        k_normed = make_k_normed()
        expected = k_normed.view(NUM_LAYERS * NUM_CTX, KV_SIZE).clone() * -1.0
        layers = make_layers(lambda _i: InPlaceRope())
        with mock_patch(f"{PATCH_MOD}.is_310p", return_value=False):
            out = run(layers, k_normed)

        torch.testing.assert_close(out, expected)
