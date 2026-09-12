#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small batch-consistency protocol shared by LI_V2 and QLI_V2 assets."""

import hashlib
import random
from numbers import Integral

import numpy as np
import torch

HIFLOAT8_QUANT_MODE = 4
SUPPORTED_QUANT_MODES = (1, 2, 3, 4, 5)
MXFP4_DECODE_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


class CaseRandomContext:
    """Give batch cases distinct backgrounds without changing normal cases."""

    def __init__(self, attributes):
        fields = tuple(attributes.get(name) for name in ("batch_axis", "batch_slice_info", "batch_seed"))
        self.enabled = any(value is not None for value in fields)
        self.testcase_name = attributes.get("testcase_name", "")
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None

    def __enter__(self):
        if not self.enabled:
            return self
        digest = hashlib.sha256(str(self.testcase_name).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % ((1 << 32) - 1)
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            random.setstate(self.python_state)
            np.random.set_state(self.numpy_state)
            torch.random.set_rng_state(self.torch_state)
        return False


class BatchRelationProtocol:
    """Parse the q-only logical B/S relation contract used by both indexers."""

    def __init__(self, operator_name):
        self.operator_name = operator_name

    @staticmethod
    def relation_slices_overlap(first, second):
        """Return whether two relation samples select the same q output region."""
        first_axes, first_slices = first[0], first[1]
        second_axes, second_slices = second[0], second[1]
        if first_axes != second_axes:
            return False
        first_batch = first_slices[0]
        second_batch = second_slices[0]
        if first_batch[1] <= second_batch[0] or second_batch[1] <= first_batch[0]:
            return False
        if first_axes == (0,):
            return True
        first_sequence = first_slices[1]
        second_sequence = second_slices[1]
        return not (first_sequence[1] <= second_sequence[0] or second_sequence[1] <= first_sequence[0])

    def validate_disjoint_relations(self, relations):
        """Reject duplicate or overlapping samples that would self-compare."""
        for index, relation in enumerate(relations):
            for candidate in relations[index + 1 :]:
                if self.relation_slices_overlap(relation, candidate):
                    raise ValueError(f"{self.operator_name} relation samples must not overlap")

    def parse(self, batch_axis, batch_slice_info, batch_seed):
        fields = (batch_axis, batch_slice_info, batch_seed)
        if all(value is None for value in fields):
            return None
        if any(value is None for value in fields):
            raise ValueError(f"{self.operator_name} batch_axis, batch_slice_info and batch_seed must be set together")
        if not (len(batch_axis) == len(batch_slice_info) == len(batch_seed)):
            raise ValueError(f"{self.operator_name} batch metadata counts differ")
        if not batch_axis or tuple(batch_axis[0]) not in ((0,), (0, 1)):
            raise ValueError(f"{self.operator_name} supports q logical axes (0,) or (0, 1)")
        if any(value is not None for value in batch_axis[1:]):
            raise ValueError(f"{self.operator_name} supports q relations only")
        if batch_slice_info[0] is None or batch_seed[0] is None:
            raise ValueError(f"{self.operator_name} requires q slices and q seeds")
        if any(value is not None for value in batch_slice_info[1:]):
            raise ValueError(f"{self.operator_name} supports q relations only")
        if any(value is not None for value in batch_seed[1:]):
            raise ValueError(f"{self.operator_name} supports q relation seeds only")

        axes = tuple(batch_axis[0])
        axis_slices = batch_slice_info[0]
        axis_seeds = batch_seed[0]
        if len(axis_slices) != len(axes) or len(axis_seeds) != len(axes):
            raise ValueError(f"{self.operator_name} q groups do not match q axes")
        sample_count = len(axis_slices[0])
        if not sample_count or any(len(values) != sample_count for values in (*axis_slices, *axis_seeds)):
            raise ValueError(f"{self.operator_name} q sample counts differ or are empty")

        relations = []
        for sample_index in range(sample_count):
            slices = []
            relation_seed = None
            for axis_group, axis in enumerate(axes):
                value = axis_slices[axis_group][sample_index]
                if not isinstance(value, (tuple, list)) or len(value) != 3:
                    raise ValueError(f"{self.operator_name} invalid q axis {axis} slice: {value!r}")
                if not all(isinstance(item, Integral) for item in value):
                    raise ValueError(f"{self.operator_name} slices must contain integers")
                start, stop, step = (int(item) for item in value)
                if step != 1 or start < 0 or start >= stop:
                    raise ValueError(f"{self.operator_name} slices must be non-empty and contiguous")
                seed = axis_seeds[axis_group][sample_index]
                if not isinstance(seed, Integral):
                    raise ValueError(f"{self.operator_name} batch seed must be an integer")
                seed = int(seed)
                if relation_seed is not None and seed != relation_seed:
                    raise ValueError(f"{self.operator_name} logical B and S must use the same seed")
                relation_seed = seed
                slices.append((start, stop, step))
            if axes == (0, 1) and slices[0][1] - slices[0][0] != 1:
                raise ValueError(f"{self.operator_name} logical (B,S) requires one B per sample")
            relations.append((axes, tuple(slices), relation_seed))
        self.validate_disjoint_relations(relations)
        return relations

    def validate_id(self, batch_consistency_id, relations):
        """Check the framework ID fields that identify a logical relation.

        Framework versions may encode slice bounds or lengths differently.  The
        seed and axis are the stable identity; slice ranges remain validated by
        ``parse`` and the relation/output checks below.
        """
        if not isinstance(batch_consistency_id, (tuple, list)) or len(batch_consistency_id) != 1:
            raise ValueError("batch_consistency_id must contain one q relation group")
        axes = relations[0][0]
        id_groups = batch_consistency_id[0]
        if not isinstance(id_groups, (tuple, list)) or len(id_groups) != len(axes):
            raise ValueError("batch_consistency_id axis groups do not match q axes")
        for group_index, axis in enumerate(axes):
            ids = id_groups[group_index]
            if not isinstance(ids, (tuple, list)):
                raise ValueError("batch_consistency_id samples must be sequences")
            if len(ids) != len(relations):
                raise ValueError("batch_consistency_id sample count does not match q relations")
            for relation_id, relation in zip(ids, relations):
                parts = str(relation_id).split("_", 2)
                if len(parts) < 2:
                    raise ValueError(f"invalid batch_consistency_id relation: {relation_id!r}")
                try:
                    id_seed, id_axis = int(parts[0]), int(parts[1])
                except ValueError as error:
                    raise ValueError(f"invalid batch_consistency_id relation: {relation_id!r}") from error
                if id_seed != int(relation[2]) or id_axis != int(axis):
                    raise ValueError("batch_consistency_id seed/axis does not match q relation")


class IndexerBatchInputNormalizer:
    """Materialize equal logical inputs for declared LI/QLI relations."""

    def __init__(self, data, attributes, operator_name, quantized):
        self.data = data
        self.attributes = attributes
        self.operator_name = operator_name
        self.quantized = quantized
        self.quant_mode = int(attributes.get("quant_mode", 1)) if quantized else None
        self.layout_q = attributes.get("layout_q", attributes.get("layout_query", "BSND"))
        self.layout_k = attributes.get("layout_k", attributes.get("layout_key", "BSND"))
        self.query = data["query"]
        self.key = data["key"]
        self.weights = data["weights"]
        self.query_scale = data.get("query_dequant_scale")
        self.key_scale = data.get("key_dequant_scale")
        self.offset = data.get("output_idx_offset")
        self.block_table = data.get("block_table")
        self.q_prefix = self.tensor_values(data.get("cu_seqlens_query", data.get("cu_seqlens_q")))
        self.k_prefix = self.tensor_values(data.get("cu_seqlens_key", data.get("cu_seqlens_k")))
        self.batch_size = self.resolve_batch_size()
        self.q_lengths = self.resolve_lengths("q")
        self.k_lengths = self.resolve_lengths("k")
        self.residual = self.resolve_vector("cmp_residual_k", 0)
        self.assigned_blocks = {}

    @staticmethod
    def tensor_values(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    def resolve_batch_size(self):
        if self.layout_q == "BSND":
            return int(self.query.shape[0])
        if self.layout_q == "TND" and self.q_prefix is not None:
            return len(self.q_prefix) - 1
        raise ValueError(f"{self.operator_name} batch consistency requires BSND or explicit TND prefix")

    def resolve_vector(self, name, default):
        value = self.attributes.get(f"{name}_values")
        if value is None:
            value = self.data.get(name)
        value = self.tensor_values(value)
        if value is None:
            return [default] * self.batch_size
        if len(value) != self.batch_size:
            raise ValueError(f"{self.operator_name} {name} length must equal B={self.batch_size}")
        return value

    def resolve_lengths(self, target):
        prefix = self.q_prefix if target == "q" else self.k_prefix
        tensor = self.query if target == "q" else self.key
        layout = self.layout_q if target == "q" else self.layout_k
        if prefix is not None:
            if (
                len(prefix) != self.batch_size + 1
                or prefix[0] != 0
                or prefix[-1] != int(tensor.shape[0])
                or any(right <= left for left, right in zip(prefix, prefix[1:]))
            ):
                raise ValueError(f"{self.operator_name} {target} prefix must strictly span its tensor")
            lengths = [right - left for left, right in zip(prefix, prefix[1:])]
        else:
            if layout == "BSND":
                lengths = [int(tensor.shape[1])] * self.batch_size
            else:
                lengths = self.resolve_vector(f"seqused_{target}", 0)
        actual = self.attributes.get(f"seqused_{target}_values")
        if actual is not None:
            actual = [int(item) for item in actual]
            if len(actual) != self.batch_size:
                raise ValueError(f"{self.operator_name} seqused_{target} length must equal B")
            if any(length <= 0 for length in actual):
                raise ValueError(f"{self.operator_name} seqused_{target} must be positive")
            if layout in ("BSND", "TND") and any(
                actual_length > physical_length for actual_length, physical_length in zip(actual, lengths)
            ):
                raise ValueError(f"{self.operator_name} seqused_{target} exceeds its tensor extent")
            return actual
        return lengths

    @staticmethod
    def derived_seed(seed, relative_batch, slot):
        value = f"{int(seed)}:{int(relative_batch)}:{int(slot)}".encode("ascii")
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % ((1 << 63) - 1)

    @classmethod
    def random_tensor(cls, shape, template, seed, relative_batch, slot, positive=False):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cls.derived_seed(seed, relative_batch, slot))
        dtype = template.dtype
        if dtype == torch.bool:
            value = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
        elif "float4" in str(dtype):
            # CPU cannot cast into Float4, but its packed byte view is writable.
            packed = torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
            return packed.view(dtype)
        elif dtype.is_floating_point:
            value = torch.rand(shape, generator=generator, dtype=torch.float32)
            value = value * 0.75 + 0.25 if positive else value - 0.5
        elif dtype == torch.uint8:
            value = torch.randint(0, 16, shape, generator=generator, dtype=torch.int64)
        else:
            value = torch.randint(-8, 9, shape, generator=generator, dtype=torch.int64)
        return value.to(dtype=dtype)

    @staticmethod
    def copy_selection(tensor, selector, value):
        if tensor is None:
            return
        source = value
        if torch.is_tensor(source) and "float4" in str(source.dtype) and "float4" not in str(tensor.dtype):
            # PyTorch has no Float4 CPU cast kernel.  The pytest CPU golden
            # stores unpacked values, while the device input uses packed bytes.
            packed = source.view(torch.uint8)
            decode_values = MXFP4_DECODE_VALUES.to(device=source.device)
            low = decode_values[(packed & 0x0F).to(torch.long)]
            high = decode_values[(packed >> 4).to(torch.long)]
            source = torch.stack((low, high), dim=-1).reshape(*source.shape[:-1], source.shape[-1] * 2)
        tensor[selector].copy_(source.to(dtype=tensor.dtype, device=tensor.device))

    def query_selector(self, batch_index, sequence_slice):
        if self.layout_q == "BSND":
            start, stop = (0, self.q_lengths[batch_index])
            if sequence_slice is not None:
                start, stop = sequence_slice[:2]
            return (batch_index, slice(start, stop, 1)), stop - start
        token_start = self.q_prefix[batch_index]
        token_stop = self.q_prefix[batch_index + 1]
        if sequence_slice is not None:
            token_start += sequence_slice[0]
            token_stop = self.q_prefix[batch_index] + sequence_slice[1]
        return (slice(token_start, token_stop, 1),), token_stop - token_start

    def query_capacity(self, batch_index):
        """Return the physical q span used by the raw-byte output comparator."""
        if self.layout_q == "BSND":
            return int(self.query.shape[1])
        return self.q_prefix[batch_index + 1] - self.q_prefix[batch_index]

    def query_comparison_length(self, batch_index):
        """Match the q span that the output comparator will actually select."""
        if self.layout_q == "TND":
            return self.query_capacity(batch_index)
        return self.q_lengths[batch_index]

    def validate_relations(self, relations):
        mask_mode = int(self.attributes.get("mask_mode", self.attributes.get("sparse_mode", 0)))
        grouped_signatures = {}
        occupied = []
        for axes, slices, seed in relations:
            batch_start, batch_stop, _ = slices[0]
            if batch_stop > self.batch_size:
                raise ValueError(f"{self.operator_name} logical B slice exceeds B={self.batch_size}")
            sequence_slice = slices[1] if axes == (0, 1) else None
            if sequence_slice is not None and mask_mode != 0:
                raise ValueError(f"{self.operator_name} shifted logical S relations require mask_mode=0")
            if (
                sequence_slice is None
                and len({self.query_capacity(batch_index) for batch_index in range(batch_start, batch_stop)}) != 1
            ):
                raise ValueError(f"{self.operator_name} one B-only relation requires equal q output spans")
            signature = []
            for batch_index in range(batch_start, batch_stop):
                selector, q_count = self.query_selector(batch_index, sequence_slice)
                if sequence_slice is not None and sequence_slice[1] > self.q_lengths[batch_index]:
                    raise ValueError(f"{self.operator_name} logical S slice exceeds effective q length")
                occupied.append((selector, seed))
                signature.append(
                    (
                        q_count if sequence_slice is not None else self.query_comparison_length(batch_index),
                        self.k_lengths[batch_index],
                        self.residual[batch_index],
                    )
                )
            relation_size = tuple(stop - start for start, stop, _step in slices)
            key = (axes, seed, relation_size)
            value = tuple(signature)
            previous = grouped_signatures.setdefault(key, value)
            if previous != value:
                raise ValueError(
                    f"{self.operator_name} relation requires equal q output spans, K lengths and residuals"
                )

        for index, (left, left_seed) in enumerate(occupied):
            for right, right_seed in occupied[index + 1 :]:
                if left_seed == right_seed or len(left) != len(right):
                    continue
                if self.selectors_overlap(left, right):
                    raise ValueError(f"{self.operator_name} relations with different seeds overlap")

    @staticmethod
    def selectors_overlap(left, right):
        for left_item, right_item in zip(left, right):
            if isinstance(left_item, int) or isinstance(right_item, int):
                if left_item != right_item:
                    return False
                continue
            if left_item.stop <= right_item.start or right_item.stop <= left_item.start:
                return False
        return True

    def query_references(self, name):
        references = [self.data.get(name)]
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        references.append(state.get(name))
        return [value for value in references if value is not None]

    def fill_query_inputs(self, batch_index, sequence_slice, seed, relative_batch):
        selector, _count = self.query_selector(batch_index, sequence_slice)
        targets = (
            (self.query_references("query"), 0, False),
            (self.query_references("weights"), 1, False),
            (self.query_references("output_idx_offset"), 3, True),
        )
        if self.quant_mode != HIFLOAT8_QUANT_MODE:
            targets += ((self.query_references("query_dequant_scale"), 2, True),)
        for references, slot, positive in targets:
            if not references:
                continue
            value = self.random_tensor(
                tuple(references[0][selector].shape),
                references[0],
                seed,
                relative_batch,
                slot,
                positive,
            )
            for tensor in references:
                self.copy_selection(tensor, selector, value)

    def key_references(self, name):
        references = []
        if name == "key":
            references.append(self.data.get("cpu_key"))
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        references.append(state.get(name))
        return [value for value in references if value is not None]

    def input_references(self, name):
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        return [value for value in (self.data.get(name), state.get(name)) if value is not None]

    def fill_hifloat8_scales(self):
        """Use one stable global scale because mode 4 scales have shape ``(1,)``."""
        for name in ("query_dequant_scale", "key_dequant_scale"):
            for tensor in self.input_references(name):
                if torch.is_tensor(tensor):
                    tensor.fill_(1)
                else:
                    np.asarray(tensor).fill(1)

    def scatter_paged(self, tensor, batch_index, value, seed, relative_batch):
        table = self.tensor_values(self.block_table[batch_index])
        block_size = int(tensor.shape[1])
        copied = 0
        owner = (seed, relative_batch)
        for block_id in table:
            if block_id < 0 or copied >= value.shape[0]:
                continue
            count = min(block_size, int(value.shape[0]) - copied)
            assignment = self.assigned_blocks.setdefault(block_id, owner)
            if assignment != owner:
                raise ValueError(
                    f"{self.operator_name} paged relations share block {block_id} between different logical batches"
                )
            tensor[block_id, :count].copy_(value[copied : copied + count].to(tensor.device, tensor.dtype))
            copied += count
        if copied != value.shape[0]:
            raise ValueError(f"{self.operator_name} block table has insufficient capacity")

    def fill_key_tensor(self, name, batch_index, seed, relative_batch, slot, positive):
        tensor = self.data.get(name)
        if tensor is None:
            return
        key_length = self.k_lengths[batch_index]
        if self.layout_k == "BSND":
            selector = (batch_index, slice(0, key_length, 1))
            shape = tuple(tensor[selector].shape)
        elif self.layout_k == "TND":
            start, stop = self.k_prefix[batch_index : batch_index + 2]
            selector = (slice(start, stop, 1),)
            shape = tuple(tensor[selector].shape)
        elif self.layout_k == "PA_BBND":
            if self.block_table is None:
                raise ValueError(f"{self.operator_name} PA_BBND requires block_table")
            selector = None
            shape = (key_length, *tuple(tensor.shape[2:]))
        else:
            raise ValueError(f"{self.operator_name} unsupported key layout {self.layout_k!r}")
        value = self.random_tensor(shape, tensor, seed, relative_batch, slot, positive)
        if selector is None:
            self.scatter_paged(tensor, batch_index, value, seed, relative_batch)
        else:
            self.copy_selection(tensor, selector, value)

        for reference in self.key_references(name):
            if reference is tensor:
                continue
            if self.layout_k == "PA_BBND":
                permutation = (1, 0, *range(2, value.ndim))
                reference[batch_index, :, :key_length].copy_(
                    value.permute(permutation).to(reference.device, reference.dtype)
                )
            else:
                self.copy_selection(reference, selector, value)

    def apply(self, relations):
        self.validate_relations(relations)
        if self.quantized and self.quant_mode not in SUPPORTED_QUANT_MODES:
            raise ValueError(f"{self.operator_name} batch consistency supports quant_mode 1 through 5")
        for axes, slices, seed in relations:
            batch_start, batch_stop, _ = slices[0]
            sequence_slice = slices[1] if axes == (0, 1) else None
            for relative_batch, batch_index in enumerate(range(batch_start, batch_stop)):
                self.fill_query_inputs(batch_index, sequence_slice, seed, relative_batch)
                self.fill_key_tensor("key", batch_index, seed, relative_batch, 10, False)
                if self.quant_mode != HIFLOAT8_QUANT_MODE:
                    self.fill_key_tensor(
                        "key_dequant_scale",
                        batch_index,
                        seed,
                        relative_batch,
                        11,
                        True,
                    )
        if self.quant_mode == HIFLOAT8_QUANT_MODE:
            self.fill_hifloat8_scales()


class IndexerBatchOutputComparator:
    """Compare exact output-0 slices for relations inside one testcase."""

    def __init__(self, operator_name):
        self.operator_name = operator_name
        self.protocol = BatchRelationProtocol(operator_name)

    @staticmethod
    def storage_bytes(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return (
                tuple(tensor.shape),
                str(tensor.dtype),
                tensor.view(torch.uint8).numpy().tobytes(),
            )
        array = np.ascontiguousarray(np.asarray(value))
        return tuple(array.shape), array.dtype.str, array.view(np.uint8).tobytes()

    def output_selector(self, output, relation, attributes):
        axes, slices, _seed = relation
        batch_slice = slices[0]
        sequence_slice = slices[1] if axes == (0, 1) else None
        layout_q = attributes.get("layout_q", attributes.get("layout_query", "BSND"))
        batch_start, batch_stop, _ = batch_slice
        if layout_q == "BSND":
            if batch_stop > output.shape[0]:
                raise ValueError(f"{self.operator_name} logical B slice exceeds output")
            selector = [slice(*batch_slice)]
            if sequence_slice is not None:
                if sequence_slice[1] > output.shape[1]:
                    raise ValueError(f"{self.operator_name} logical S slice exceeds output")
                selector.append(slice(*sequence_slice))
            else:
                q_lengths = attributes.get("seqused_q_values")
                if q_lengths is not None:
                    selected_lengths = [int(value) for value in q_lengths[batch_start:batch_stop]]
                    if (
                        len(selected_lengths) != batch_stop - batch_start
                        or len(set(selected_lengths)) != 1
                        or selected_lengths[0] <= 0
                        or selected_lengths[0] > output.shape[1]
                    ):
                        raise ValueError(f"{self.operator_name} invalid effective q lengths for output")
                    selector.append(slice(0, selected_lengths[0], 1))
        elif layout_q == "TND":
            prefix = attributes.get("cu_seqlens_q_values")
            if prefix is None or prefix[0] != 0 or prefix[-1] != output.shape[0]:
                raise ValueError(f"{self.operator_name} TND output requires q prefix values")
            if batch_stop >= len(prefix):
                raise ValueError(f"{self.operator_name} logical B slice exceeds q prefix")
            if sequence_slice is None:
                token_start, token_stop = prefix[batch_start], prefix[batch_stop]
            else:
                token_start = prefix[batch_start] + sequence_slice[0]
                token_stop = prefix[batch_start] + sequence_slice[1]
                if token_stop > prefix[batch_start + 1]:
                    raise ValueError(f"{self.operator_name} logical S slice exceeds TND interval")
            selector = [slice(token_start, token_stop, 1)]
        else:
            raise ValueError(f"{self.operator_name} unsupported query layout {layout_q!r}")
        selector.extend([slice(None)] * (output.ndim - len(selector)))
        return tuple(selector)

    def compare(
        self,
        output,
        batch_consistency_id,
        batch_axis,
        batch_slice_info,
        batch_seed,
        compare_context,
    ):
        try:
            relations = self.protocol.parse(batch_axis, batch_slice_info, batch_seed)
            if relations is None:
                return None
            self.protocol.validate_id(batch_consistency_id, relations)
            if output is None:
                raise ValueError(f"{self.operator_name} batch output is None")
            value = output.detach().cpu() if torch.is_tensor(output) else np.asarray(output)
            attributes = dict(compare_context.attributes) if compare_context else {}
            groups = {}
            for relation in relations:
                selected = value[self.output_selector(value, relation, attributes)]
                axes, slices, seed = relation
                stored = self.storage_bytes(selected)
                relation_size = tuple(stop - start for start, stop, _step in slices)
                groups.setdefault((axes, seed, relation_size), []).append(stored)
            compared = 0
            for key, values in groups.items():
                if len(values) < 2:
                    continue
                compared += 1
                if any(values[0] != item for item in values[1:]):
                    return {
                        "pass": False,
                        "precision": "batch_intra=FAIL",
                        "error_info": f"{self.operator_name} relation {key} differs",
                    }
            if compared == 0:
                return {"pass": True, "precision": "batch_intra=NOT_APPLICABLE"}
            return {"pass": True, "precision": "batch_intra=PASS"}
        except (IndexError, TypeError, ValueError) as error:
            return {
                "pass": False,
                "precision": "batch_config=FAIL",
                "error_info": str(error),
            }


def normalize_indexer_inputs(data, attributes, operator_name, quantized=False):
    protocol = BatchRelationProtocol(operator_name)
    relations = protocol.parse(
        attributes.get("batch_axis"),
        attributes.get("batch_slice_info"),
        attributes.get("batch_seed"),
    )
    if relations is not None:
        IndexerBatchInputNormalizer(data, attributes, operator_name, quantized).apply(relations)
