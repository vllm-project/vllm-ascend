# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import hashlib
import json
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.kimi_k3 import prune_experts as p


class ExpertPruneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ep256-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_fixture(self):
        src = self.root / "src"
        dst = self.root / "dst"
        src.mkdir()
        dst.mkdir()
        ids, remap = p.mapping(896, 256)
        base = "language_model.model.layers.1.block_sparse_moe."
        records = [
            (base + "experts.0.w1.weight", "U8", [2, 4], bytes(range(8))),
            (base + "experts.1.w1.weight", "U8", [2, 4], b"ABCDEFGH"),
            (base + "experts.3.w1.weight", "U8", [2, 4], b"12345678"),
            (base + "gate.weight", "BF16", [896, 7168], b"".join(struct.pack("<H", i) * 7168 for i in range(896))),
            (base + "gate.e_score_correction_bias", "BF16", [896], b"".join(struct.pack("<H", i) for i in range(896))),
            ("unmodified", "F32", [2], struct.pack("<ff", 1.25, -2.0)),
        ]
        h = {}
        offset = 0
        payload = b""
        expected = {}
        items = []
        for key, dtype, shape, data in records:
            m = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
            h[key] = m
            payload += data
            offset += len(data)
            tr = p.transform(key, m, remap, 896)
            if tr:
                nk, nm, op = tr
                if op == "rows":
                    rb = len(data) // 896
                    expected[nk] = b"".join(data[i * rb : (i + 1) * rb] for i in ids)
                else:
                    expected[nk] = data
                items.append(
                    {
                        "source_key": key,
                        "key": nk,
                        "meta": nm,
                        "op": op,
                        "source_offsets": m["data_offsets"],
                        "size": len(expected[nk]),
                    }
                )
        raw = json.dumps(h, separators=(",", ":")).encode()
        raw += b" " * ((-len(raw)) % 8)
        file = src / "input.safetensors"
        file.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        hn, _, hh = p.read_header(file)
        shard = {
            "source": file.name,
            "output": "output.safetensors",
            "source_header_len": hn,
            "metadata": None,
            "identity": p.fingerprint(file, hh),
            "items": items,
        }
        return src, dst, ids, shard, expected

    def test_mapping(self):
        ids, remap = p.mapping(896, 256)
        self.assertEqual(ids[:4], [0, 3, 7, 10])
        self.assertEqual(ids[-1], 892)
        self.assertEqual(len(set(ids)), 256)
        self.assertEqual(set(remap.values()), set(range(256)))
        self.assertNotIn(895, remap)

    def test_copy_remove_rename_router_and_verify(self):
        src, dst, ids, sh, expected = self.make_fixture()
        receipt = p.write_shard(src, dst, sh, ids, lambda _: None)
        out = dst / sh["output"]
        hn, h, _ = p.read_header(out)
        self.assertNotIn(
            "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight",
            {x["source_key"]: 1 for x in sh["items"]},
        )
        with open(out, "rb") as f:
            for key, data in expected.items():
                a, b = h[key]["data_offsets"]
                f.seek(8 + hn + a)
                self.assertEqual(f.read(b - a), data)
                self.assertEqual(receipt["tensor_hashes"][key], hashlib.sha256(data).hexdigest())
        p.verify_shard(out, sh, receipt["tensor_hashes"])
        self.assertEqual(h["language_model.model.layers.1.block_sparse_moe.gate.weight"]["shape"], [256, 7168])

    def test_payload_corruption_detected(self):
        src, dst, ids, sh, _ = self.make_fixture()
        r = p.write_shard(src, dst, sh, ids, lambda _: None)
        out = dst / sh["output"]
        with open(out, "r+b") as f:
            f.seek(-1, 2)
            old = f.read(1)
            f.seek(-1, 2)
            f.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            p.verify_shard(out, sh, r["tensor_hashes"])

    def test_four_independent_shards_in_parallel(self):
        src, dst, ids, sh, _ = self.make_fixture()
        shards = []
        for i in range(4):
            item = copy.deepcopy(sh)
            item["output"] = f"parallel-{i}.safetensors"
            shards.append(item)
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(lambda s: p.write_shard(src, dst, s, ids, lambda _: None), shards))
        for shard, receipt in zip(shards, receipts):
            p.verify_shard(dst / shard["output"], shard, receipt["tensor_hashes"])

    def test_source_changed_rejected(self):
        src, dst, ids, sh, _ = self.make_fixture()
        with open(src / sh["source"], "ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ValueError, "source changed"):
            p.write_shard(src, dst, sh, ids, lambda _: None)

    def test_assets_config_quant_remap_and_optional(self):
        src = self.root / "src"
        dst = self.root / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "optional").mkdir()
        cfg = {"text_config": {"num_experts": 896, "num_hidden_layers": 93, "num_experts_per_token": 16}}
        base = "language_model.model.layers.1.block_sparse_moe.experts."
        q = {"group_size": 32, base + "1.w1.weight": "W4A8_MXFP", base + "3.w1.weight": "W4A8_MXFP"}
        (src / "config.json").write_text(json.dumps(cfg))
        (src / "quant_model_description.json").write_text(json.dumps(q))
        (src / "optional/rotation.bin").write_bytes(b"rotation-unchanged")
        assets = [
            {"name": n, "sha256": p.file_hash(src / n)}
            for n in ["config.json", "quant_model_description.json", "optional/rotation.bin"]
        ]
        ids, _ = p.mapping(896, 256)
        p.write_assets(src, dst, {"assets": assets, "keep_ids": ids, "keep_experts": 256})
        c = p.read_json(dst / "config.json")
        self.assertEqual(c["text_config"]["num_experts"], 256)
        self.assertEqual(c["text_config"]["num_hidden_layers"], 93)
        outq = p.read_json(dst / "quant_model_description.json")
        self.assertEqual(outq, {"group_size": 32, base + "1.w1.weight": "W4A8_MXFP"})
        self.assertEqual((dst / "optional/rotation.bin").read_bytes(), b"rotation-unchanged")

    def test_router_wrong_shape_rejected(self):
        _, remap = p.mapping(896, 256)
        with self.assertRaisesRegex(ValueError, "unexpected router format"):
            p.transform(
                "language_model.model.layers.1.block_sparse_moe.gate.weight",
                {"dtype": "BF16", "shape": [895, 7168], "data_offsets": [0, 0]},
                remap,
                896,
            )

    def test_malformed_offsets_rejected(self):
        file = self.root / "bad.safetensors"
        h = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}
        raw = json.dumps(h).encode()
        file.write_bytes(struct.pack("<Q", len(raw)) + raw + b"ab")
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            p.read_header(file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
