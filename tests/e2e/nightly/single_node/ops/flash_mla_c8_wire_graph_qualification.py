# SPDX-License-Identifier: Apache-2.0
"""Native T64 direct wire with empty sections and changing graph metadata.

Use the same candidate PYTHONPATH/OPP as qualify_npu.py. This is a separate
process and does not modify the original qualification or production files.
"""

import argparse
import json
import os
from pathlib import Path

_CUSTOM_OPP_PATH = os.environ.get("ASCEND_CUSTOM_OPP_PATH")

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401
from flash_mla_c8_qualification import C8Case

from vllm_ascend.ops.triton.sfa_dcp_exchange import pack_raw_dcp_output_lse

if _CUSTOM_OPP_PATH is not None:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH


def bit_equal(a, b):
    return torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


def physical_state(used, lengths):
    assert len(used) == len(lengths) == 16
    active = torch.zeros(64, dtype=torch.bool)
    for request, count in enumerate(used):
        assert 0 <= count <= 4
        if lengths[request] > 0:
            active[request * 4 : request * 4 + count] = True
    return active


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    result = {"status": "running", "physical_tokens": 64, "cu_step": 4, "states": []}

    def checkpoint():
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.json)

    checkpoint()
    try:
        case = C8Case([16384] * 16, 96, qlen=4)
        metadata = case.metadata()
        captured_metadata_ptr = metadata.data_ptr()
        used_ptr, length_ptr = case.used.data_ptr(), case.lengths.data_ptr()

        def run():
            output, lse = case.run(metadata, layout="NTD_DCP")
            wire = pack_raw_dcp_output_lse(output, lse.transpose(0, 1).unsqueeze(-1), 272)
            return output, lse, wire

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        graph_output_ptrs = [value.data_ptr() for value in captured]
        states = [
            ("valid", [4] * 16, [16384] * 16, False),
            ("empty_queries_empty_history", [0] * 16, [0] * 16, False),
            ("valid_again", [4] * 16, [16384] * 16, False),
            ("ragged", [0, 1, 2, 3, 4, 1, 0, 4] * 2, [16384, 127, 128, 129, 0, 4097, 0, 8192] * 2, False),
            ("empty_queries_nonempty_history", [0] * 16, [16384] * 16, False),
            ("zero_section_kernel_contract", [0] * 16, [0] * 16, True),
            ("valid_final", [4] * 16, [16384] * 16, False),
        ]
        for name, used, lengths, force_zero_sections in states:
            case.used.copy_(torch.tensor(used, dtype=torch.int32))
            case.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
            updated_metadata = case.metadata()
            assert updated_metadata.shape == metadata.shape
            if force_zero_sections:
                # An explicit legal no-work schedule exercises the native
                # zero-section contract even if the AICPU scheduler chooses
                # one empty section for fixed physical query dimensions.
                # Preserve all other header fields; no FA/FD records exist.
                updated_metadata[0:2].zero_()
                updated_metadata[16:].zero_()
            # Update the previously captured scheduling address. Constructing
            # a fresh schedule and leaving the captured pointer stale is invalid.
            metadata.copy_(updated_metadata)
            assert metadata.data_ptr() == captured_metadata_ptr
            assert case.used.data_ptr() == used_ptr and case.lengths.data_ptr() == length_ptr
            header = metadata[:7].cpu().tolist()  # Qualification diagnostics only.
            assert case.used.cpu().tolist() == used
            assert case.lengths.cpu().tolist() == lengths
            if force_zero_sections:
                assert header[0] == 0, (name, "schedule is not empty", header)
            elif any(used) and any(lengths):
                assert header[0] > 0, (name, "schedule unexpectedly empty", header)
            # Poison every physical word and the independent LSE so valid→zero
            # replay cannot pass by inheriting previously initialized storage.
            captured[2].fill_(0x5A5A5A5A)
            captured[1].fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            assert [value.data_ptr() for value in captured] == graph_output_ptrs
            output, lse, wire = (value.cpu() for value in captured)
            assert output.shape == (64, 96, 512)
            assert torch.count_nonzero(wire[..., 257:]) == 0, (name, "wire padding was not zeroed")
            embedded = wire[..., 256].reshape(96, 64)
            assert bit_equal(embedded, lse.contiguous().view(torch.int32)), (name, "two LSE copies differ")
            active = physical_state(used, lengths)
            inactive = ~active
            if inactive.any():
                assert torch.count_nonzero(output[inactive].contiguous().view(torch.int16)) == 0, (name, "inactive O not exact +0")
                assert torch.equal(lse[:, inactive].contiguous().view(torch.int32), torch.full((96, int(inactive.sum())), -8388608, dtype=torch.int32)), (name, "inactive LSE not -inf")
            if active.any():
                # Old TND invalid rows are not a golden. Compare only rows with
                # a live query and nonempty history, which it actually computes.
                normal, normal_lse = case.run(metadata, layout="TND")
                torch.npu.synchronize()
                assert bit_equal(output[active], normal.cpu()[active]), (name, "active O differs")
                assert bit_equal(lse[:, active], normal_lse.cpu()[:, active]), (name, "active LSE differs")
                assert torch.isfinite(output[active]).all()
            eager = run()
            assert bit_equal(captured[0], eager[0]) and bit_equal(captured[1], eager[1])
            assert bit_equal(captured[2], eager[2])
            result["states"].append({"name": name, "used": used, "lengths": lengths, "metadata_head": header, "metadata_source": "explicit zero-section kernel contract" if force_zero_sections else "actual AICPU scheduler", "active_tokens": int(active.sum()), "inactive_tokens": int(inactive.sum()), "same_metadata_address": True, "same_graph_output_addresses": True, "independent_zero_gold": "PASS", "embedded_lse_padding": "PASS", "active_vs_normal": "bitwise PASS" if active.any() else "not used", "graph_vs_eager": "bitwise PASS"})
            checkpoint()
            print(json.dumps(result["states"][-1]), flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        checkpoint()
        raise
    checkpoint()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
