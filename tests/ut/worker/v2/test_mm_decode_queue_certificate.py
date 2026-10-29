# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU checks for request-completion admission and invalidation.

These exercise the production proof helper directly. They cannot validate NPU
graph capture, HCCL execution, model outputs, or a runtime performance benefit.
"""

import ast
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = ROOT / "vllm_ascend/worker/v2/mm_decode_queue.py"
SPEC = importlib.util.spec_from_file_location("_test_mm_decode_queue_certificate", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
# Load the CPU-only helper without importing the NPU package entry point.
# The temporary registration supports dataclasses without leaking fake modules.
with patch.dict(sys.modules, {SPEC.name: HELPER}):
    SPEC.loader.exec_module(HELPER)


def feature(offset=5, length=8, modality="image", identifier="image-1", is_embed=None):
    return SimpleNamespace(
        modality=modality,
        identifier=identifier,
        data=object(),
        mm_position=SimpleNamespace(offset=offset, length=length, is_embed=is_embed),
    )


def key(req_id="r1", slot=0, prefill_len=32, context=("model-epoch-1",), features=None):
    if features is None:
        features = [feature()]
    return HELPER.make_mm_decode_request_key(req_id, slot, prefill_len, context, features)


class TestAdversarialMMDecodeCertificate(unittest.TestCase):
    def test_optimistic_cpu_count_cannot_bootstrap_certificate(self):
        proof = HELPER.MMDecodeCertificateState()
        request = key()
        self.assertFalse(proof.is_certified([request]))
        self.assertFalse(proof.certify([request], [request.prefill_len + 100]))
        proof.record_snapshot([request])
        self.assertFalse(proof.certify([request], [request.prefill_len - 1]))
        self.assertFalse(proof.is_certified([request]))

    def test_exact_completion_latches_without_reading_next_snapshot(self):
        proof = HELPER.MMDecodeCertificateState()
        request = key()
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [request.prefill_len]))
        proof.record_snapshot([request])
        self.assertTrue(proof.is_certified([request]))
        # A newly enqueued copy is not read by the admission query.
        self.assertEqual(proof.snapshot_generation, 2)

    def test_previous_slot_occupant_cannot_certify_new_request(self):
        proof = HELPER.MMDecodeCertificateState()
        old_request = key("old", slot=7, prefill_len=2048)
        new_request = key("new", slot=7, prefill_len=32)
        proof.record_snapshot([old_request])
        self.assertFalse(proof.certify([new_request], [2048]))
        self.assertFalse(proof.is_certified([new_request]))

    def test_reused_request_id_requires_lifecycle_invalidation_and_new_snapshot(self):
        proof = HELPER.MMDecodeCertificateState()
        request = key()
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [32]))
        proof.invalidate()
        self.assertFalse(proof.is_certified([request]))
        self.assertFalse(proof.certify([request], [32]))
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [32]))

    def test_context_or_media_mutation_revokes_current_key(self):
        proof = HELPER.MMDecodeCertificateState()
        media = [feature()]
        request = key(features=media)
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [32]))
        for changes in (
            dict(slot=1),
            dict(prefill_len=33),
            dict(context=("model-epoch-2",)),
        ):
            self.assertFalse(proof.is_certified([key(features=media, **changes)]))
        media[0].mm_position.offset += 1
        self.assertFalse(proof.is_certified([key(features=media)]))

    def test_media_identity_change_requires_new_exact_proof(self):
        proof = HELPER.MMDecodeCertificateState()
        first_media = [feature()]
        request = key(features=first_media)
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [32]))
        identical_but_new_media = [feature()]
        replacement = key(features=identical_but_new_media)
        self.assertFalse(proof.is_certified([replacement]))
        self.assertFalse(proof.certify([replacement], [32]))

    def test_batch_reordering_preserves_per_request_proof(self):
        proof = HELPER.MMDecodeCertificateState()
        a, b = key("a", 2), key("b", 9, prefill_len=64)
        proof.record_snapshot([a, b])
        self.assertTrue(proof.certify([b, a], [64, 32]))
        self.assertTrue(proof.is_certified([a, b]))
        self.assertTrue(proof.is_certified([b]))

    def test_one_incomplete_request_does_not_admit_mixed_batch(self):
        proof = HELPER.MMDecodeCertificateState()
        a, b = key("a", 2), key("b", 9, prefill_len=64)
        proof.record_snapshot([a, b])
        self.assertFalse(proof.certify([a, b], [32, 63]))
        self.assertFalse(proof.is_certified([a, b]))

    def test_empty_duplicate_or_mismatched_batch_is_not_certified(self):
        proof = HELPER.MMDecodeCertificateState()
        request = key()
        proof.record_snapshot([request])
        self.assertFalse(proof.certify([], []))
        self.assertFalse(proof.certify([request], []))
        self.assertFalse(proof.certify([request, request], [32, 32]))
        self.assertFalse(proof.is_certified([]))

    def test_unsupported_media_layouts_fail_closed(self):
        cases = (
            [feature(-1, 3)],
            [feature(0, 0)],
            [feature(29, 4)],
            [feature(8, 4), feature(9, 4)],
            [feature(10, 4), feature(5, 2)],
            [feature(modality="prompt_embeds")],
            [feature(is_embed=[True, False])],
        )
        for media in cases:
            with self.subTest(media=media):
                self.assertIsNone(key(features=media))
        self.assertIsNone(key(slot=-1))
        self.assertIsNone(key(prefill_len=0))

    def test_adjacent_image_and_video_ranges_can_be_certified(self):
        media = [feature(0, 8), feature(8, 8, modality="video")]
        request = key(features=media)
        self.assertIsNotNone(request)
        proof = HELPER.MMDecodeCertificateState()
        proof.record_snapshot([request])
        self.assertTrue(proof.certify([request], [32]))

    def test_randomized_completion_boundaries_and_request_slot_reuse(self):
        rng = random.Random(0xDEC0DE)
        for generation in range(300):
            proof = HELPER.MMDecodeCertificateState()
            prompt_len = rng.randrange(16, 100_001)
            slot = rng.randrange(32)
            request = key(str(generation), slot, prompt_len, features=[])
            proof.record_snapshot([request])
            exact_count = prompt_len + rng.randrange(-8, 9)
            expected = exact_count >= prompt_len
            self.assertEqual(proof.certify([request], [exact_count]), expected)
            self.assertEqual(proof.is_certified([request]), expected)
            # A different occupant of the same physical row must fail even if
            # the old snapshot exceeds its shorter prompt boundary.
            replacement = key("replacement", slot, 8, features=[])
            self.assertFalse(proof.certify([replacement], [exact_count]))
            self.assertFalse(proof.is_certified([replacement]))


class TestMediaContextGuards(unittest.TestCase):
    """Run the actual admission-context method with CPU-only model stand-ins."""

    @classmethod
    def setUpClass(cls):
        path = HELPER_PATH.with_name("model_runner.py")
        module = ast.parse(path.read_text(encoding="utf-8"))
        runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_qwen_mrv2_mm_decode_context"
        )
        body = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {"ObjectIdentity": HELPER.ObjectIdentity}
        exec(compile(body, str(path), "exec"), namespace)
        cls.context_method = staticmethod(namespace[method.name])

    def make_context(self):
        class Encoder:
            def gather_mm_embeddings(self):
                pass

        class State:
            def prepare_inputs_embeds(self):
                pass

            def gather_mm_embeddings(self):
                pass

        cache = object()
        encoder = Encoder()
        encoder.encoder_cache = cache
        encoder.is_realtime = False
        state = State()
        state.encoder_runner = encoder
        state.encoder_cache = cache
        state.supports_mm_inputs = True
        state.mm_pruner = None
        state.prompt_embeds_state = None
        state.rope_state = object()
        runner = SimpleNamespace(
            _qwen_mrv2_mm_decode_proof_enabled=True,
            _qwen_mrv2_mm_decode_epoch=0,
            req_states=object(),
            model_state=state,
            supports_mm_inputs=True,
            encoder_cache=cache,
            model=SimpleNamespace(use_deepstack=True, deepstack_input_embeds_num_tokens=0),
            model_config=SimpleNamespace(
                enable_prompt_embeds=False,
                is_mm_prefix_lm=False,
                is_encoder_decoder=False,
                hf_text_config=object(),
            ),
        )
        target_module = ModuleType("vllm_ascend.models.qwen4_exp.model_state")
        target_module.AscendQwen4ExpModelState = State
        return runner, target_module

    def get_context(self, runner, target_module):
        with patch.dict(sys.modules, {target_module.__name__: target_module}):
            return self.context_method(runner)

    def test_deepstack_supported_only_after_overlay_clear_was_submitted(self):
        runner, module = self.make_context()
        self.assertIsNotNone(self.get_context(runner, module))
        runner.model.deepstack_input_embeds_num_tokens = 8
        self.assertIsNone(self.get_context(runner, module))

    def test_realtime_pruning_prompt_embeds_and_prefix_lm_fail_closed(self):
        changes = (
            ("encoder", "is_realtime", True),
            ("state", "mm_pruner", object()),
            ("state", "prompt_embeds_state", object()),
            ("config", "enable_prompt_embeds", True),
            ("config", "is_mm_prefix_lm", True),
            ("config", "is_encoder_decoder", True),
        )
        for target, name, value in changes:
            with self.subTest(target=target, name=name):
                runner, module = self.make_context()
                selected = {
                    "encoder": runner.model_state.encoder_runner,
                    "state": runner.model_state,
                    "config": runner.model_config,
                }[target]
                setattr(selected, name, value)
                self.assertIsNone(self.get_context(runner, module))

    def test_encoder_cache_identity_must_match_model_state(self):
        runner, module = self.make_context()
        runner.model_state.encoder_runner.encoder_cache = object()
        self.assertIsNone(self.get_context(runner, module))

    def test_model_rope_and_request_epoch_form_part_of_proof_identity(self):
        for target, name in (
            ("state", "rope_state"),
            ("runner", "model"),
            ("runner", "req_states"),
        ):
            with self.subTest(target=target, name=name):
                runner, module = self.make_context()
                before = self.get_context(runner, module)
                selected = runner.model_state if target == "state" else runner
                setattr(selected, name, object())
                after = self.get_context(runner, module)
                self.assertNotEqual(before, after)
        runner, module = self.make_context()
        before = self.get_context(runner, module)
        runner._qwen_mrv2_mm_decode_epoch += 1
        self.assertNotEqual(before, self.get_context(runner, module))


if __name__ == "__main__":
    unittest.main(verbosity=2)
