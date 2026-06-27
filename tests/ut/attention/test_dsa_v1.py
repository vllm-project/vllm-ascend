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
#

import sys
from unittest.mock import MagicMock, patch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, AscendDSAMetadataBuilder


class TestAscendDSAImplUpdateGraphParams(TestBase):
    """Regression guard for `AscendDSAImpl.update_graph_params`.

    `acl_graph.update_full_graph_params` unconditionally invokes
    `attn_backend.get_impl_cls().update_graph_params(...)` whenever a
    forward context runs in `CUDAGraphMode.FULL`. If `AscendDSAImpl`
    forgets to define this method (as was the case before the
    accompanying fix), the very first inference request crashes with
    `AttributeError: type object 'AscendDSAImpl' has no attribute
    'update_graph_params'`. This test pins the method's existence and
    no-op contract.
    """

    def test_update_graph_params_is_noop(self):
        # The method is a staticmethod and intentionally does nothing
        # (DSA's NPU graph path does not need parameter refresh, mirroring
        # `AscendSFAImpl.update_graph_params`).
        update_stream = MagicMock()
        forward_context = MagicMock()
        # Should not raise, regardless of optional kwargs being supplied.
        AscendDSAImpl.update_graph_params(update_stream, forward_context, 100)
        AscendDSAImpl.update_graph_params(
            update_stream,
            forward_context,
            100,
            vllm_config=MagicMock(),
            speculative_config=MagicMock(),
            num_dcp_pcp_tokens=None,
            draft_attn_metadatas=None,
        )


class TestAscendDSAMetadataBuilderBuildForGraphCapture(TestBase):
    """Regression guard for the SAS-metadata kwargs contract.

    `AscendDSAMetadataBuilder.build()` asserts that
    `prefill_ratio_to_sas_metadata`, `decode_ratio_to_sas_metadata`, and
    `common_ratio_to_sas_metadata` are all non-None when invoked with
    `use_compress=True`. `build_for_graph_capture` is the entrypoint used
    by the MTP draft `dummy_run` path, so any caller that forgets to
    forward these kwargs (as `llm_base_proposer.dummy_run` did before
    the accompanying fix) silently breaks graph capture with an
    `AssertionError` deep inside `build()`.

    This test patches out `build` and only verifies that
    `build_for_graph_capture` faithfully forwards the kwargs it receives,
    without exercising any NPU-specific code paths.
    """

    def test_build_for_graph_capture_forwards_sas_metadata_kwargs(self):
        # Construct a builder instance without running the real __init__
        # (which touches scheduler config, scipy.linalg.hadamard and NPU
        # buffers — none of which are available in a unit-test sandbox).
        builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)

        common_attn_metadata = MagicMock()
        prefill_sas = {"sentinel": "prefill"}
        decode_sas = {"sentinel": "decode"}
        common_sas = {"sentinel": "common"}
        block_size = 128

        sentinel_metadata = MagicMock()
        with patch.object(AscendDSAMetadataBuilder, "build", return_value=sentinel_metadata) as mock_build:
            result = builder.build_for_graph_capture(
                common_attn_metadata=common_attn_metadata,
                attn_state=AscendAttentionState.SpecDecoding,
                prefill_ratio_to_sas_metadata=prefill_sas,
                decode_ratio_to_sas_metadata=decode_sas,
                common_ratio_to_sas_metadata=common_sas,
                block_size=block_size,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        self.assertIs(kwargs["common_attn_metadata"], common_attn_metadata)
        self.assertEqual(kwargs["common_prefix_len"], 0)
        self.assertIs(kwargs["prefill_ratio_to_sas_metadata"], prefill_sas)
        self.assertIs(kwargs["decode_ratio_to_sas_metadata"], decode_sas)
        self.assertIs(kwargs["common_ratio_to_sas_metadata"], common_sas)
        self.assertEqual(kwargs["block_size"], block_size)

        # Returned metadata should have its attn_state stamped by
        # build_for_graph_capture.
        self.assertIs(result, sentinel_metadata)
        self.assertEqual(sentinel_metadata.attn_state, AscendAttentionState.SpecDecoding)

    def test_build_for_graph_capture_rejects_unsupported_state(self):
        builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
        common_attn_metadata = MagicMock()

        with self.assertRaises(NotImplementedError):
            builder.build_for_graph_capture(
                common_attn_metadata=common_attn_metadata,
                attn_state=AscendAttentionState.ChunkedPrefill,
            )
