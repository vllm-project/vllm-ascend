#!/usr/bin/env python3
"""Single-card qualification of the imported A5 MLA ABI.

Run in the isolated A5 build container, after its OPP environment is loaded:
  python test_npu_contract.py --device 0 --json mla-eager.json
  python test_npu_contract.py --device 0 --graph --json mla-graph.json

This executable fails on the first effective error and prints each gate before
dispatch. It never starts a service or changes a model. Capture is opt-in so an
AICPU capture failure is isolated from eager qualification. CPU references use
explicit logical page gather and right-aligned causal attention, independent of
the custom kernels. All host readbacks occur after dispatch/capture/replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import torch


def cpu_reference(q, cache, blocks, cu, used, lengths, causal, scale):
    """Return TND output and head-major LSE; invalid tokens are zero/+inf."""
    q, cache = q.float(), cache.float()
    out = torch.zeros((*q.shape[:2], 512), dtype=torch.float32)
    lse = torch.full((q.shape[1], q.shape[0]), float("inf"))
    live = torch.zeros(q.shape[0], dtype=torch.bool)
    page = cache.shape[1]
    for b, (uq, kvlen) in enumerate(zip(used, lengths)):
        if uq == 0 or kvlen == 0:
            continue
        assert 0 <= uq <= cu[b + 1] - cu[b]
        assert kvlen <= blocks.shape[1] * page
        keys = torch.cat([cache[p] for p in blocks[b].tolist()], dim=0)[:kvlen]
        query = q[cu[b] : cu[b] + uq]
        scores = torch.einsum("qhd,kd->hqk", query, keys) * scale
        if causal:
            visible = torch.arange(kvlen)[None, :] <= (kvlen - uq + torch.arange(uq)[:, None])
            scores.masked_fill_(~visible[None, :, :], float("-inf"))
        row_lse = scores.logsumexp(-1)
        probability = scores.softmax(-1).nan_to_num(0)
        out[cu[b] : cu[b] + uq] = torch.einsum("hqk,kd->qhd", probability, keys[:, :512])
        lse[:, cu[b] : cu[b] + uq] = row_lse
        live[cu[b] : cu[b] + uq] = torch.isfinite(row_lse).any(0)
    return out, lse, live


class Qualification:
    def __init__(self, args):
        import torch_npu  # noqa: F401
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        self.torch_npu = torch_npu
        self.args = args
        torch.npu.set_device(args.device)
        self.device = torch.device(f"npu:{args.device}")
        self.device_name = torch.npu.get_device_name(self.device)
        self.dtype = getattr(torch, args.dtype)
        self.aic = torch.npu.get_device_properties(self.device).cube_core_num
        self.aiv = torch.npu.get_device_properties(self.device).vector_core_num
        self.results = []
        self.started = time.time()
        self.generator = torch.Generator().manual_seed(16464)
        self.pages, self.page, self.heads = 12, args.page_size, args.heads
        # A layer-1 view within three interleaved layers; the 96-element tail
        # creates an additional gap after every physical allocator page.
        self.payload = self.page * 576
        self.stride = 3 * self.payload + 96
        self.offset = 128 + self.payload
        storage_size = 128 + self.pages * self.stride + 128
        self.expected_backing = torch.full((storage_size,), -13.0, dtype=self.dtype)
        self.expected_cache = self.expected_backing.as_strided(
            (self.pages, self.page, 576), (self.stride, 576, 1), self.offset
        )
        self.expected_cache.copy_(self.random(self.expected_cache.shape))
        self.backing = self.expected_backing.to(self.device)
        self.cache = self.backing.as_strided(self.expected_cache.shape, self.expected_cache.stride(), self.offset)
        self.cu_cpu = [0, 4, 7, 9]
        self.blocks_cpu = torch.tensor([[3, 0], [5, 8], [2, 1]], dtype=torch.int32)
        self.blocks = self.blocks_cpu.to(self.device)
        self.cu = torch.tensor(self.cu_cpu, dtype=torch.int32, device=self.device)
        self.used = torch.zeros(3, dtype=torch.int32, device=self.device)
        self.lengths = torch.zeros_like(self.used)
        self.q_cpu = self.random((9, self.heads, 576))
        self.q = self.q_cpu.to(self.device)
        self.scale = 1 / math.sqrt(576)
        self.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).to(self.device)
        self.metadata = torch.full(
            (((((self.aic + self.aiv) * 3 + 1) * 16 + 4095) // 4096 * 4096),), -211, dtype=torch.int32, device=self.device
        )
        self.graph_metadata = torch.empty_like(self.metadata)
        self.graph_cu = torch.empty_like(self.cu)
        self.graph_used = torch.empty_like(self.used)
        self.graph_lengths = torch.empty_like(self.lengths)
        self.graph_q = torch.empty_like(self.q)
        self.graph_out = torch.empty((self.heads, 9, 512), dtype=self.dtype, device=self.device)
        self.graph_lse = torch.empty((self.heads, 9), dtype=torch.float32, device=self.device)

    def random(self, shape):
        return (torch.randn(shape, generator=self.generator) * 0.5).to(self.dtype)

    def record(self, gate, **details):
        item = {"gate": gate, "status": "PASS", **details}
        self.results.append(item)
        print(json.dumps(item), flush=True)

    def begin(self, gate):
        print(json.dumps({"gate": gate, "status": "RUNNING"}), flush=True)

    def writer(self):
        self.begin("writer_nonzero_offset_layer_gap_sentinels")
        c = self.random((8, 1, 512))
        r = self.random((8, 1, 64))
        # Includes both slot integer widths and ignored negative slots. Distinct
        # slots avoid defining unspecified behavior for duplicate writes.
        slots = [
            0,
            self.page - 1,
            self.page,
            3 * self.page + 2,
            -1,
            5 * self.page + 1,
            -1,
            11 * self.page + self.page - 1,
        ]
        for integer in (torch.int64, torch.int32):
            c = (c.float() + 0.125).to(self.dtype)
            r = (r.float() - 0.125).to(self.dtype)
            c_npu, r_npu = c.to(self.device), r.to(self.device)
            slots_npu = torch.tensor(slots, dtype=integer, device=self.device)
            self.torch_npu.npu_scatter_pa_kv_cache(
                c_npu, r_npu, self.cache[..., :512].unsqueeze(2),
                self.cache[..., 512:].unsqueeze(2), slots_npu, cache_mode="Norm"
            )
            torch.npu.synchronize()
            for i, slot in enumerate(slots):
                if slot >= 0:
                    p, token = divmod(slot, self.page)
                    start = self.offset + p * self.stride + token * 576
                    self.expected_backing[start : start + 512] = c[i, 0]
                    self.expected_backing[start + 512 : start + 576] = r[i, 0]
            torch.testing.assert_close(self.backing.cpu(), self.expected_backing, rtol=0, atol=0)
        self.record(
            "writer_nonzero_offset_layer_gap_sentinels",
            shape=list(self.cache.shape),
            strides=list(self.cache.stride()),
            storage_offset=self.cache.storage_offset(),
            checked_backing_elements=self.backing.numel(),
        )

    def stage(self, used, lengths, q_cpu=None, cu_cpu=None):
        self.used.copy_(torch.tensor(used, dtype=torch.int32, device=self.device))
        self.lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=self.device))
        if q_cpu is not None:
            self.q_cpu = q_cpu
            self.q.copy_(q_cpu)
        if cu_cpu is not None:
            self.cu_cpu = cu_cpu
            self.cu.copy_(torch.tensor(cu_cpu, dtype=torch.int32, device=self.device))

    def dispatch(self, *, causal, layout="NTD", return_lse=True, out=None, lse=None, captured_inputs=False):
        cu, used, lengths, query, metadata = self.cu, self.used, self.lengths, self.q, self.metadata
        if captured_inputs:
            # Backend-style copies/derivation occur inside the captured region.
            # The staged input vectors keep stable addresses between epochs.
            self.graph_cu.copy_(cu)
            self.graph_used.copy_(used)
            self.graph_lengths.copy_(lengths)
            self.graph_q.copy_(query)
            cu, used, lengths, query, metadata = (
                self.graph_cu,
                self.graph_used,
                self.graph_lengths,
                self.graph_q,
                self.graph_metadata,
            )
        produced = torch.ops._C_ascend.flash_mla_with_kvcache_metadata(
            lengths, self.heads, 1, cu_seqlens_q=cu, seqused_q=used,
            max_seqlen_q=9, max_seqlen_kv=2 * self.page,
            head_dim_qk=576, head_dim_v=512,
            mask_mode=3 if causal else 0, layout_q="TND",
        )
        metadata.copy_(produced)
        result = torch.ops._C_ascend.flash_mla_with_kvcache(
            query, self.cache.unsqueeze(1), block_table=self.blocks,
            cache_seqlens=lengths, cu_seqlens_q=cu, seqused_q=used,
            attn_mask=self.mask if causal else None, metadata=metadata,
            head_dim_v=512, softmax_scale=self.scale,
            mask_mode=3 if causal else 0, max_seqlen_q=9,
            max_seqlen_kv=2 * self.page, layout_q="TND", layout_kv="PA_BNBD",
            layout_out=layout, return_softmax_lse=return_lse,
        )
        if out is not None:
            out.copy_(result[0])
            lse.copy_(result[1])
            return out, lse
        return result

    def verify(self, out, lse, used, lengths, *, causal, layout, gate):
        torch.npu.synchronize()
        observed = out.cpu().float()
        if layout == "NTD":
            observed = observed.permute(1, 0, 2)
        expected, expected_lse, live = cpu_reference(
            self.q_cpu, self.expected_cache, self.blocks_cpu, self.cu_cpu, used, lengths, causal, self.scale
        )
        torch.testing.assert_close(observed, expected, rtol=self.args.rtol, atol=self.args.atol)
        # Padding must be overwritten every epoch, including all-idle replays.
        torch.testing.assert_close(observed[~live], torch.zeros_like(observed[~live]), rtol=0, atol=0)
        if lse.numel():
            observed_lse = lse.cpu()
            torch.testing.assert_close(observed_lse[:, live], expected_lse[:, live], rtol=0.01, atol=0.01)
            assert torch.isposinf(observed_lse[:, ~live]).all(), "invalid LSE must be +inf"
        # Reader must not mutate adjacent layers, page gaps or its KV payload.
        torch.testing.assert_close(self.backing.cpu(), self.expected_backing, rtol=0, atol=0)
        self.record(
            gate,
            max_abs_error=float((observed - expected).abs().max()),
            used_q=used,
            cache_seqlens=lengths,
            cu_seqlens_q=self.cu_cpu,
            layout=layout,
            lse_elements=lse.numel(),
        )

    def eager(self):
        # All cases share the same metadata and device length buffers.
        cases = [
            ("partial_used", [3, 2, 0], [self.page + 5, 7, 0]),
            ("zero_used_positive_kv", [0, 2, 0], [self.page + 5, 7, 3]),
            ("positive_q_zero_kv", [4, 3, 2], [self.page + 5, 7, 0]),
            ("all_idle", [0, 0, 0], [0, 0, 0]),
            ("all_zero_kv_positive_capacity", [4, 3, 2], [0, 0, 0]),
            ("active_after_idle", [1, 3, 1], [3, self.page + 7, 4]),
        ]
        if self.args.smoke:
            cases = cases[:1]
        for causal in (False, True):
            for layout in ("NTD",) if self.args.smoke else ("NTD", "TND"):
                shape = (self.heads, 9, 512) if layout == "NTD" else (9, self.heads, 512)
                for return_lse in (False,) if self.args.smoke else (False, True):
                    out = torch.empty(shape, dtype=self.dtype, device=self.device)
                    lse = torch.empty((self.heads, 9) if return_lse else (0,), dtype=torch.float32, device=self.device)
                    for label, used, lengths in cases:
                        gate = f"eager_{label}_causal{int(causal)}_{layout}_lse{int(return_lse)}"
                        self.begin(gate)
                        self.stage(used, lengths)
                        out.fill_(19)
                        lse.fill_(-29)
                        result = self.dispatch(causal=causal, layout=layout, return_lse=return_lse, out=out, lse=lse)
                        self.verify(*result, used, lengths, causal=causal, layout=layout, gate=gate)

    def graph(self):
        self.graph_writer()
        for causal in (False, True):
            gate = f"capture_causal{int(causal)}"
            self.begin(gate)
            initial_used, initial_lengths = [3, 2, 0], [self.page + 5, 7, 0]
            self.stage(initial_used, initial_lengths)
            kwargs = dict(causal=causal, out=self.graph_out, lse=self.graph_lse, captured_inputs=True)
            for _ in range(3):
                self.dispatch(**kwargs)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                self.dispatch(**kwargs)
            self.record(gate)
            ptrs = tuple(
                t.data_ptr()
                for t in (self.q, self.cu, self.used, self.lengths, self.graph_metadata, self.graph_out, self.graph_lse)
            )
            epochs = [
                ("active", [3, 2, 0], [self.page + 5, 7, 0], [0, 4, 7, 9]),
                ("idle", [0, 0, 0], [0, 0, 0], [0, 4, 7, 9]),
                ("changed_offsets", [1, 4, 0], [9, self.page + 3, 0], [0, 2, 8, 9]),
                ("idle_again", [0, 0, 0], [0, 0, 0], [0, 0, 0, 9]),
                ("active_again", [2, 1, 1], [3, 7, 5], [0, 4, 7, 9]),
            ]
            for label, used, lengths, cu in epochs:
                gate = f"graph_{label}_causal{int(causal)}"
                self.begin(gate)
                self.stage(used, lengths, self.random(self.q_cpu.shape), cu)
                self.graph_out.fill_(19)
                self.graph_lse.fill_(-29)
                graph.replay()
                self.verify(self.graph_out, self.graph_lse, used, lengths, causal=causal, layout="NTD", gate=gate)
                assert ptrs == tuple(
                    t.data_ptr()
                    for t in (
                        self.q,
                        self.cu,
                        self.used,
                        self.lengths,
                        self.graph_metadata,
                        self.graph_out,
                        self.graph_lse,
                    )
                )
            del graph
            torch.npu.synchronize()

    def graph_writer(self):
        self.begin("capture_writer")
        c_cpu, r_cpu = self.random((3, 1, 512)), self.random((3, 1, 64))
        c, r = c_cpu.to(self.device), r_cpu.to(self.device)
        # Warmup/capture starts with no valid slot, preserving the eager cache.
        slots = torch.full((3,), -1, dtype=torch.int64, device=self.device)
        for _ in range(3):
            self.mla.write_mla_cache(c, r, slots, self.cache)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.mla.write_mla_cache(c, r, slots, self.cache)
        self.record("capture_writer")
        for epoch, mapping in enumerate(
            ([3 * self.page + 1, 5 * self.page + 2, -1], [-1, -1, -1], [3 * self.page + 2, -1, 5 * self.page + 1])
        ):
            self.begin(f"graph_writer_epoch{epoch}")
            c_cpu, r_cpu = self.random(c.shape), self.random(r.shape)
            c.copy_(c_cpu)
            r.copy_(r_cpu)
            slots.copy_(torch.tensor(mapping, dtype=torch.int64, device=self.device))
            graph.replay()
            torch.npu.synchronize()
            for token, slot in enumerate(mapping):
                if slot >= 0:
                    page, offset = divmod(slot, self.page)
                    start = self.offset + page * self.stride + offset * 576
                    self.expected_backing[start : start + 512] = c_cpu[token, 0]
                    self.expected_backing[start + 512 : start + 576] = r_cpu[token, 0]
            torch.testing.assert_close(self.backing.cpu(), self.expected_backing, rtol=0, atol=0)
            self.record(
                f"graph_writer_epoch{epoch}", slot_mapping=mapping, checked_backing_elements=self.backing.numel()
            )
        del graph
        torch.npu.synchronize()

    def report(self, status, error=None):
        mapped_libraries = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ("libcust_opapi", "libcust_opmaster", "vllm_ascend_C"))
        }
        return {
            "status": status,
            "error": error,
            "seconds": time.time() - self.started,
            "torch": torch.__version__,
            "device": str(self.device),
            "device_name": self.device_name,
            "dtype": self.args.dtype,
            "heads": self.heads,
            "page_size": self.page,
            "opp_paths": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "loaded_libraries": sorted(mapped_libraries),
            "aic_cores": self.aic,
            "aiv_cores": self.aiv,
            "graph_requested": self.args.graph or self.args.graph_only,
            "smoke": self.args.smoke,
            "gates": self.results,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Run writer plus causal/noncausal reader before model startup"
    )
    parser.add_argument(
        "--graph-only", action="store_true", help="Skip the eager matrix after it has passed separately"
    )
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    qualifier = None
    try:
        qualifier = Qualification(args)
        qualifier.writer()
        if not args.graph_only:
            qualifier.eager()
        if args.graph or args.graph_only:
            qualifier.graph()
        report = qualifier.report("PASS")
    except Exception:
        error = traceback.format_exc()
        report = qualifier.report("FAIL", error) if qualifier else {"status": "FAIL", "error": error, "gates": []}
        print(error, flush=True)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        raise
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "gates_passed": len(report["gates"]), "seconds": report["seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
