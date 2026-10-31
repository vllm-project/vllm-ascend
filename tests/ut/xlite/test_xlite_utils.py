#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""CPU-only unit tests for the pure-python helpers of ``vllm_ascend.xlite.utils``.

Covers the dotted/nested attribute lookup, the layer weight extraction
(``get_layer_weights``) including mask / secondary flattening / post-processing
/ contiguity checks, the tensor representation helpers, the safe attribute
setter mixin, and the ``AttnMetadataRouter`` routing logic (via duck-typed
metadata objects).
"""

import subprocess
import sys
import types
import unittest

import torch


def _ensure_xlite_C_importable() -> None:
    """Make the native ``xlite._C`` extension stand-in importable, if needed.

    ``vllm_ascend.xlite.utils`` imports ``xlite._C`` at module scope and
    subclasses ``Model`` / ``ModelConfig`` from it (``XModel`` /
    ``XModelConfig``). Outside an Ascend/CANN environment that native extension
    aborts the interpreter with a C++ ``std::logic_error`` while importing, so
    it cannot be imported (or even probed) in the CPU-only CI image. The
    helpers under test in this module are pure python and never touch the
    native module at runtime, so probe the native import in a subprocess and,
    only when it is broken, install a minimal ``xlite._C`` module whose
    ``Model`` / ``ModelConfig`` are plain classes (subclassable by
    ``utils.XModel`` / ``utils.XModelConfig`` at import time). When the native
    extension imports cleanly (a full CANN environment) nothing is touched.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import xlite._C"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
    )
    if probe.returncode == 0:
        return
    try:
        import xlite
    except ImportError:
        xlite = types.ModuleType("xlite")
        sys.modules["xlite"] = xlite
    xlite_C = types.ModuleType("xlite._C")

    class _StubModel:
        """Plain stand-in for ``xlite._C.Model`` (native module unavailable)."""

    class _StubModelConfig:
        """Plain stand-in for ``xlite._C.ModelConfig`` (native module unavailable)."""

    xlite_C.Model = _StubModel
    xlite_C.ModelConfig = _StubModelConfig
    xlite._C = xlite_C
    sys.modules["xlite._C"] = xlite_C


_ensure_xlite_C_importable()

from vllm_ascend.xlite import utils as xlite_utils  # noqa: E402
from vllm_ascend.xlite.utils import (  # noqa: E402
    AttnMetadataRouter,
    AttributeSetterMixin,
    _TensorRepr,
    get_dotted_attr,
    get_layer_weights,
    get_nested_attr,
    set_dummy_tensor,
)


def _layer(**attrs) -> object:
    """A lightweight stand-in for a decoder layer with plain attributes."""
    layer = types.SimpleNamespace()
    for key, value in attrs.items():
        setattr(layer, key, value)
    return layer


class TestNestedAttrLookup(unittest.TestCase):
    def test_get_nested_attr_resolves_path(self):
        obj = _layer(a=_layer(b=_layer(c=42)))
        self.assertEqual(get_nested_attr(obj, "a", "b", "c"), 42)

    def test_get_nested_attr_single_attr(self):
        self.assertEqual(get_nested_attr(_layer(x=7), "x"), 7)

    def test_get_nested_attr_empty_path_returns_root(self):
        obj = _layer(x=1)
        self.assertIs(get_nested_attr(obj), obj)

    def test_get_nested_attr_missing_returns_default(self):
        obj = _layer(a=_layer(b=1))
        self.assertIsNone(get_nested_attr(obj, "a", "missing"))
        self.assertEqual(get_nested_attr(obj, "a", "missing", default="d"), "d")

    def test_get_nested_attr_raises_on_missing(self):
        obj = _layer(a=_layer(b=1))
        with self.assertRaises(AttributeError) as ctx:
            get_nested_attr(obj, "a", "missing", raises=True)
        self.assertIn("missing", str(ctx.exception))

    def test_get_dotted_attr_splits_on_dot(self):
        obj = _layer(a=_layer(b=_layer(c="v")))
        self.assertEqual(get_dotted_attr(obj, "a.b.c"), "v")

    def test_get_dotted_attr_missing_default_and_raises(self):
        obj = _layer(a=1)
        self.assertEqual(get_dotted_attr(obj, "a.missing", default=0), 0)
        with self.assertRaises(AttributeError):
            get_dotted_attr(obj, "a.missing", raises=True)


class _Layer:
    """Plain attribute container used as a 'layer' for weight extraction."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class TestGetLayerWeights(unittest.TestCase):
    def setUp(self):
        self.w0 = torch.ones(2, 3)
        self.w1 = torch.ones(4, 3)

    def test_direct_attribute(self):
        layers = [_Layer(w=self.w0), _Layer(w=self.w1)]
        self.assertIs(get_layer_weights(layers, "w")[0], self.w0)
        self.assertIs(get_layer_weights(layers, "w")[1], self.w1)

    def test_missing_layers_are_skipped(self):
        layers = [_Layer(w=self.w0), _Layer(other=1)]
        self.assertEqual(get_layer_weights(layers, "w"), [self.w0])

    def test_first_candidate_wins(self):
        layers = [_Layer(primary=self.w0), _Layer(primary=self.w1)]
        # The second candidate is never consulted because the first yields
        # a non-empty result.
        self.assertEqual(get_layer_weights(layers, "missing", "primary"), [self.w0, self.w1])
        self.assertEqual(get_layer_weights(layers, "missing", "secondary", "primary"), [self.w0, self.w1])

    def test_all_candidates_missing_returns_empty(self):
        self.assertEqual(get_layer_weights([_Layer(a=1)], "nope", "nothing"), [])

    def test_non_string_candidate_raises_type_error(self):
        with self.assertRaises(TypeError):
            get_layer_weights([_Layer(a=1)], 42)

    def test_mask_pads_masked_out_layers_with_dummy(self):
        layers = [_Layer(w=self.w0), _Layer(w=self.w1)]
        weights = get_layer_weights(layers, "w", mask=[True, False])
        self.assertIs(weights[0], self.w0)
        self.assertIs(weights[1], xlite_utils._DUMMY_TENSOR)
        self.assertEqual(weights[1].numel(), 0)

    def test_masked_in_layer_without_attribute_gets_dummy(self):
        layers = [_Layer(w=self.w0), _Layer(other=1)]
        weights = get_layer_weights(layers, "w", mask=[True, True])
        self.assertIs(weights[0], self.w0)
        self.assertIs(weights[1], xlite_utils._DUMMY_TENSOR)

    def test_mask_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            get_layer_weights([_Layer(w=self.w0)], "w", mask=[True, True])

    def test_mask_must_be_booleans(self):
        with self.assertRaises(ValueError):
            get_layer_weights([_Layer(w=self.w0)], "w", mask=[1])

    def test_mask_and_flattening_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            get_layer_weights([_Layer(w=self.w0)], "w", mask=[True], secondary_flattening="count")

    def test_secondary_flattening_by_count_attribute(self):
        layers = [
            _Layer(weights=[self.w0, torch.zeros(1), torch.zeros(1)], count=1),
            _Layer(weights=[self.w1], count=1),
        ]
        self.assertEqual(get_layer_weights(layers, "weights", secondary_flattening="count"), [self.w0, self.w1])

    def test_secondary_flattening_by_slice(self):
        layers = [_Layer(weights=[self.w0, torch.zeros(1), torch.zeros(1)])]
        self.assertEqual(
            get_layer_weights(layers, "weights", secondary_flattening=slice(0, 2)), [self.w0, torch.zeros(1)]
        )

    def test_post_processor_is_applied(self):
        layers = [_Layer(w=self.w0)]
        doubled = get_layer_weights(layers, "w", post_processor=lambda t: t * 2)
        self.assertIsNot(doubled[0], self.w0)
        torch.testing.assert_close(doubled[0], self.w0 * 2)

    def test_non_contiguous_weight_raises(self):
        non_contiguous = torch.zeros(4, 4).t()
        self.assertFalse(non_contiguous.is_contiguous())
        with self.assertRaises(ValueError):
            get_layer_weights([_Layer(w=non_contiguous)], "w")

    def test_contiguity_check_can_be_disabled(self):
        non_contiguous = torch.zeros(4, 4).t()
        self.assertEqual(
            get_layer_weights([_Layer(w=non_contiguous)], "w", check_is_contiguous=False), [non_contiguous]
        )

    def test_set_dummy_tensor_rebinds_placeholder(self):
        sentinel = torch.zeros(2, 2)
        set_dummy_tensor(sentinel)
        try:
            # The placeholder is read from the module global at call time, so
            # the rebind immediately affects the masked-out padding.
            weights = get_layer_weights([_Layer(other=1)], "w", mask=[False])
            self.assertIs(weights[0], sentinel)
        finally:
            set_dummy_tensor(torch.empty(0))


class TestTensorRepr(unittest.TestCase):
    def test_single_tensor_repr(self):
        text = repr(_TensorRepr(torch.zeros(2, 3)))
        self.assertIn("shape=(2, 3)", text)
        self.assertIn("dtype=float32", text)
        self.assertTrue(text.startswith("Tensor("))

    def test_non_contiguous_tensor_includes_stride(self):
        t = torch.zeros(4, 4)[:, :2]
        text = repr(_TensorRepr(t))
        self.assertIn("stride=", text)

    def test_contiguous_tensor_has_no_stride(self):
        self.assertNotIn("stride=", repr(_TensorRepr(torch.zeros(3))))

    def test_iterable_repr_prunes_with_ellipsis(self):
        tensors = [torch.zeros(1), torch.zeros(2), torch.zeros(3), torch.zeros(4)]
        text = repr(_TensorRepr(tensors, max_it=3))
        self.assertTrue(text.startswith("TensorList(len=4,"))
        # max_it=3 keeps three items and appends a single ellipsis.
        self.assertTrue(text.endswith("...])"))
        self.assertEqual(text.count("Tensor("), 3)

    def test_duplicate_reprs_are_collapsed(self):
        tensors = [torch.zeros(2), torch.zeros(2), torch.zeros(3)]
        text = repr(_TensorRepr(tensors))
        # The second occurrence of the same shape is replaced by "...".
        self.assertIn("...", text)
        self.assertEqual(text.count("Tensor("), 2)

    def test_empty_iterable(self):
        self.assertEqual(repr(_TensorRepr([])), "TensorList(len=0)")
        self.assertEqual(repr(_TensorRepr([[]])), "TensorList2D(len=1, items=[TensorList(len=0)])")

    def test_nested_iterable_repr(self):
        text = repr(_TensorRepr([[torch.zeros(1), torch.zeros(2)], [torch.zeros(3)]]))
        self.assertTrue(text.startswith("TensorList2D(len=2,"))
        self.assertIn("TensorList(len=2", text)
        self.assertIn("TensorList(len=1", text)

    def test_non_tensor_value_passes_through(self):
        self.assertEqual(repr(_TensorRepr(5)), "5")

    def test_max_it_none_shows_everything(self):
        tensors = [torch.zeros(1), torch.zeros(2), torch.zeros(3), torch.zeros(4)]
        text = repr(_TensorRepr(tensors, max_it=None))
        self.assertNotIn("...", text)
        self.assertEqual(text.count("Tensor("), 4)

    def test_bool_of_empty_tensors(self):
        self.assertFalse(_TensorRepr(torch.empty(0)))
        self.assertTrue(_TensorRepr(torch.zeros(1)))
        self.assertFalse(_TensorRepr([torch.empty(0), torch.empty(0)]))
        self.assertTrue(_TensorRepr([torch.empty(0), torch.zeros(1)]))
        self.assertTrue(_TensorRepr(5))
        self.assertTrue(_TensorRepr([1, 2]))


class _SetterBase:
    x = 0


class IgnoreSetter(AttributeSetterMixin, _SetterBase):
    _on_missing_attr = "ignore"
    _logger = None


class RaiseSetter(AttributeSetterMixin, _SetterBase):
    _on_missing_attr = "raise"
    _logger = None


class WarnSetter(AttributeSetterMixin, _SetterBase):
    _logger = None  # defaults to "warn"; keep the test quiet


class TestAttributeSetterMixin(unittest.TestCase):
    def test_setting_existing_attribute_works(self):
        obj = IgnoreSetter()
        obj.x = 42
        self.assertEqual(obj.x, 42)

    def test_missing_attribute_ignored(self):
        obj = IgnoreSetter()
        obj.does_not_exist = "value"
        self.assertFalse(hasattr(obj, "does_not_exist"))

    def test_missing_attribute_raises(self):
        obj = RaiseSetter()
        with self.assertRaises(AttributeError) as ctx:
            obj.does_not_exist = "value"
        self.assertIn("does_not_exist", str(ctx.exception))

    def test_missing_attribute_warns_by_default(self):
        obj = WarnSetter()
        # "warn" mode logs through the (here: disabled) logger and drops the
        # value silently.
        obj.does_not_exist = "value"
        self.assertFalse(hasattr(obj, "does_not_exist"))

    def test_condition_gates_assignment(self):
        obj = IgnoreSetter()
        with obj.condition(lambda v: v > 0):
            obj.x = 5
            obj.x = -1  # ignored: -1 > 0 is False
        self.assertEqual(obj.x, 5)

    def test_nested_conditions_combine_with_and(self):
        obj = IgnoreSetter()
        with obj.condition(lambda v: v > 0), obj.condition(lambda v: v < 10):
            obj.x = 5  # 0 < 5 < 10 -> set
            obj.x = 20  # 20 < 10 is False -> ignored
        self.assertEqual(obj.x, 5)

    def test_condition_restored_after_exception(self):
        obj = IgnoreSetter()
        with self.assertRaises(ValueError), obj.condition(lambda v: v > 0):
            raise ValueError("boom")
        # The gate is gone again: a non-positive value can be stored.
        obj.x = -5
        self.assertEqual(obj.x, -5)

    def test_condition_restored_after_exception_inside(self):
        obj = IgnoreSetter()
        try:
            with obj.condition(lambda v: v > 0):
                obj.x = 5
                raise RuntimeError("inner")
        except RuntimeError:
            pass
        obj.x = -5  # gate fully restored even though the first set succeeded
        self.assertEqual(obj.x, -5)

    def test_mixin_must_be_first_parent(self):
        with self.assertRaises(TypeError):

            class _Bad(_SetterBase, AttributeSetterMixin):
                pass

    def test_repr_lists_attributes(self):
        class ReprHolder(AttributeSetterMixin, _SetterBase):
            _on_missing_attr = "ignore"
            _logger = None
            y = None  # must pre-exist as a class attribute to be settable

        obj = ReprHolder()
        obj.x = 1
        obj.y = torch.zeros(2)
        text = repr(obj)
        self.assertIn("x=1", text)
        self.assertIn("shape=(2,)", text)
        self.assertNotIn("_on_missing_attr", text)


class TestAttnMetadataRouter(unittest.TestCase):
    def test_cu_query_lens_prefers_query_start_loc(self):
        meta = types.SimpleNamespace(query_start_loc=[0, 3], cum_query_lens=[9, 9])
        router = AttnMetadataRouter(meta)
        torch.testing.assert_close(router.cu_query_lens, torch.tensor([0, 3]))

    def test_cu_query_lens_cum_query_lens_candidate(self):
        meta = types.SimpleNamespace(cum_query_lens=[0, 2])
        torch.testing.assert_close(AttnMetadataRouter(meta).cu_query_lens, torch.tensor([0, 2]))

    def test_cu_query_lens_actual_seq_lengths_candidate(self):
        meta = types.SimpleNamespace(actual_seq_lengths_q=[0, 5])
        torch.testing.assert_close(AttnMetadataRouter(meta).cu_query_lens, torch.tensor([0, 5]))

    def test_cu_query_lens_missing_raises(self):
        with self.assertRaises(ValueError):
            _ = AttnMetadataRouter(types.SimpleNamespace()).cu_query_lens

    def test_block_tables_prefers_block_tables(self):
        both = torch.arange(4)
        meta = types.SimpleNamespace(block_tables=both, block_table=torch.zeros(2))
        self.assertIs(AttnMetadataRouter(meta).block_tables, torch.as_tensor(both))

    def test_block_tables_falls_back_to_block_table(self):
        table = torch.arange(6).reshape(3, 2)
        meta = types.SimpleNamespace(block_tables=None, block_table=table)
        torch.testing.assert_close(AttnMetadataRouter(meta).block_tables, table)

    def test_block_tables_missing_raises(self):
        # Both candidate attributes present but None -> documented ValueError.
        meta = types.SimpleNamespace(block_tables=None, block_table=None)
        with self.assertRaises(ValueError):
            _ = AttnMetadataRouter(meta).block_tables

    def test_seq_lens_prefers_cpu_variant(self):
        meta = types.SimpleNamespace(seq_lens_cpu=[1, 2], seq_lens=[9])
        torch.testing.assert_close(AttnMetadataRouter(meta).seq_lens, torch.tensor([1, 2]))

    def test_seq_lens_falls_back_to_seq_lens(self):
        meta = types.SimpleNamespace(seq_lens_cpu=None, seq_lens=[4])
        torch.testing.assert_close(AttnMetadataRouter(meta).seq_lens, torch.tensor([4]))

    def test_seq_lens_missing_raises(self):
        meta = types.SimpleNamespace(seq_lens_cpu=None, seq_lens=None)
        with self.assertRaises(ValueError):
            _ = AttnMetadataRouter(meta).seq_lens

    def test_num_counters(self):
        meta = types.SimpleNamespace(num_prefills=2, num_decodes=3, num_decode_tokens=None)
        router = AttnMetadataRouter(meta)
        self.assertEqual(router.num_prefills, 2)
        self.assertEqual(router.num_decodes, 3)
        # num_decode_tokens has a None default in getattr: missing/None -> 0.
        self.assertEqual(router.num_decode_tokens, 0)
        self.assertEqual(AttnMetadataRouter(types.SimpleNamespace()).num_decode_tokens, 0)

    def test_num_prefills_none_counts_as_zero(self):
        meta = types.SimpleNamespace(num_prefills=None, num_decodes=None)
        router = AttnMetadataRouter(meta)
        self.assertEqual(router.num_prefills, 0)
        self.assertEqual(router.num_decodes, 0)

    def test_num_actual_tokens_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            _ = AttnMetadataRouter(types.SimpleNamespace(num_actual_tokens=None)).num_actual_tokens

    def test_num_actual_tokens_missing_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = AttnMetadataRouter(types.SimpleNamespace()).num_actual_tokens

    def test_num_actual_tokens_returns_int(self):
        router = AttnMetadataRouter(types.SimpleNamespace(num_actual_tokens=17))
        self.assertEqual(router.num_actual_tokens, 17)

    def test_getattr_passes_through_existing_attributes(self):
        meta = types.SimpleNamespace(slots=torch.arange(3))
        self.assertTrue(torch.equal(AttnMetadataRouter(meta).slots, torch.arange(3)))

    def test_getattr_missing_raises_descriptive_error(self):
        router = AttnMetadataRouter(types.SimpleNamespace())
        with self.assertRaises(AttributeError) as ctx:
            _ = router.definitely_missing
        self.assertIn("definitely_missing", str(ctx.exception))

    def test_on_device_is_restored(self):
        router = AttnMetadataRouter(types.SimpleNamespace(query_start_loc=[0, 1]))
        with router.on_device(None):
            self.assertIsNone(router.device)
        self.assertEqual(router.device, "cpu")

    def test_on_device_restored_after_exception(self):
        router = AttnMetadataRouter(types.SimpleNamespace())
        with self.assertRaises(RuntimeError), router.on_device(None):
            raise RuntimeError("inner")
        self.assertEqual(router.device, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
