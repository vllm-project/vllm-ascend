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
# This file is a part of the vllm-ascend project.

"""Static regressions for the recurrent KDA device-metadata contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/attention/recurrent_kda"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_recurrent_kda_uses_vllm_ascend_apache_headers():
    source_suffixes = {".cpp", ".h", ".py", ".txt"}
    sources = [OP_ROOT / "CMakeLists.txt", OP_ROOT / "op_host/CMakeLists.txt"]
    sources.extend(path for path in OP_ROOT.rglob("*") if path.suffix in source_suffixes)

    for path in sources:
        source = _read(path)
        assert "SPDX-License-Identifier: Apache-2.0" in source, path
        assert "CANN Open Software License Agreement" not in source, path


def test_aclnn_uses_device_cu_seqlens_and_aliased_state_output():
    header = _read(OP_ROOT / "op_host/op_api/aclnn_recurrent_kda.h")
    l0_source = _read(OP_ROOT / "op_host/op_api/recurrent_kda.cpp")

    assert "const aclTensor *initialStateRef" in header
    assert "const aclTensor *cuSeqlensOptional" in header
    assert "aclIntArray" not in header
    assert "const aclTensor *finalState" in header
    assert "OP_OUTPUT(attnOut, initialStateRef, finalState)" in l0_source


def test_tiling_processor_owns_context_and_supports_strided_state():
    source = _read(OP_ROOT / "op_host/recurrent_kda_tiling_processor.h")
    tiling = _read(OP_ROOT / "op_host/recurrent_kda_tiling.cpp")

    assert "RecurrentKdaTilingContext ctx_;" in source
    assert "const RecurrentKdaTilingContext &ctx_;" not in source
    assert "speculative [seq_num,max_step]" in source
    assert "ssmStateStride" in source
    assert "GetInputStride(STATE_INDEX)" in tiling
    assert "GetOutputStride(stateOutputIndex)" in tiling
    assert "stateInStrides" in source
    assert "stateOutStrides" in source
    assert "outer strides overlap state rows or heads" in source


def test_kernel_skips_empty_sequences_before_state_metadata_access():
    source = _read(OP_ROOT / "op_kernel/recurrent_kda.h")

    empty_skip = source.index("if (seqLen64 == 0)")
    slot_validation = source.index("ValidateStateSlots(batch_i, seq0, seqLen)")
    state_prefetch = source.index("PrefetchState(stateSlot, head_i, 0, nextSingleV)")
    assert empty_skip < slot_validation < state_prefetch
    assert "batchIdx * ssmStateStride_" in source
    assert "stateSlot >= static_cast<int64_t>(stateCapacity_)" in source


def test_cu_seqlens_uses_fla_prefix_sum_semantics():
    kernel_paths = [
        OP_ROOT / "op_kernel/recurrent_kda.h",
        OP_ROOT / "op_kernel/arch35/recurrent_kda.h",
    ]

    for kernel_path in kernel_paths:
        source = _read(kernel_path)
        assert "int64_t seq0 = SequenceStart(batch_i)" in source
        assert "int64_t seq1 = SequenceEnd(batch_i)" in source
        assert "int64_t seqLen64 = seq1 - seq0" in source
        assert "return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx)" in source
        assert "return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx + 1)" in source
        assert "if (seq0 != 0)" in source
        assert "return seq0 <= static_cast<int64_t>(T_)" in source
        assert "return seq0 == static_cast<int64_t>(T_)" not in source


def test_kernel_uses_real_state_strides_and_generated_dtype_macro():
    entry = _read(OP_ROOT / "op_kernel/recurrent_kda.cpp")
    assert "DTYPE_INITIAL_STATE" in entry
    assert "DTYPE_STATE" not in entry

    for kernel_path in (
        OP_ROOT / "op_kernel/recurrent_kda.h",
        OP_ROOT / "op_kernel/arch35/recurrent_kda.h",
    ):
        source = _read(kernel_path)
        assert "stateInStride0_" in source
        assert "stateInStride1_" in source
        assert "stateInStride2_" in source
        assert "stateInStride3_" in source
        assert "stateOutStride0_" in source
        assert "stateOutStride1_" in source
        assert "stateOutStride2_" in source
        assert "stateOutStride3_" in source


def test_arch35_kernel_has_dedicated_micro_api_implementation():
    source = _read(OP_ROOT / "op_kernel/arch35/recurrent_kda.h")

    assert '#include "../recurrent_kda.h"' not in source
    assert "using namespace AscendC::MicroAPI;" in source
    assert "__VEC_SCOPE__" in source
    assert "inline void MatVecMul" in source
    assert "inline void ProcessKQ" in source
    assert "inline void ReduceSumDispatch" in source


def test_torch_binding_preserves_mutation_and_accepts_tnd():
    adapter = _read(OP_ROOT / "recurrent_kda_torch_adpt.h")
    schema = _read(ROOT / "csrc/torch_binding.cpp")

    assert 'const char* layout = is_tnd ? "TND" : "BSND";' in adapter
    assert "        layout," in adapter
    assert "speculative [seq_num,max_step]" in adapter
    assert "at::Tensor final_state = at::empty_like(initial_state)" in adapter
    assert "bool inplace_final_state = true" in adapter
    assert "k_dim == 128 && (v_dim == 128 || v_dim == 256)" in adapter
    assert "        final_state);" in adapter
    assert "Tensor(a!) initial_state" in schema
    assert "Tensor cu_seqlens" in schema
    assert "-> Tensor output" in schema
    assert "Tensor(a!) final_state" not in schema
