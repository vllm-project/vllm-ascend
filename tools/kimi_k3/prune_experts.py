# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#!/usr/bin/env python3
"""Byte-preserving routed-expert pruning; no model execution or requantization."""

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import regex as re

EXPERT = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$")
SIX = {f"{w}.{s}" for w in ("w1", "w2", "w3") for s in ("weight", "weight_scale")}
DTYPE_BYTES = {
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}
CHUNK = 8 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def unique_pairs(items):
    d = {}
    for k, v in items:
        require(k not in d, "duplicate JSON key: " + k)
        d[k] = v
    return d


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def read_header(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, "truncated prefix: " + str(path))
        n = struct.unpack("<Q", raw)[0]
        require(2 <= n <= min(path.stat().st_size - 8, 512 * 1024**2), "invalid header length")
        raw = f.read(n)
        h = json.loads(raw, object_pairs_hook=unique_pairs)
    cursor = 0
    intervals = []
    for k, m in h.items():
        if k == "__metadata__":
            require(
                isinstance(m, dict) and all(isinstance(x, str) and isinstance(y, str) for x, y in m.items()),
                "invalid metadata",
            )
            continue
        a, b = m["data_offsets"]
        require(type(a) is int and type(b) is int and 0 <= a <= b, "invalid tensor offsets")
        require(m["dtype"] in DTYPE_BYTES, "unsupported dtype: " + m["dtype"])
        require(all(type(x) is int and x >= 0 for x in m["shape"]), "invalid shape")
        require(math.prod(m["shape"]) * DTYPE_BYTES[m["dtype"]] == b - a, "shape/byte mismatch: " + k)
        intervals.append((a, b))
    for a, b in sorted(intervals):
        require(a == cursor, "payload gap or overlap: " + str(path))
        cursor = b
    require(8 + n + cursor == path.stat().st_size, "payload length mismatch: " + str(path))
    return n, h, digest(raw)


def source_file(src, name):
    p = src / name
    require(not Path(name).is_absolute() and p.is_file() and not p.is_symlink(), "invalid source: " + name)
    require(src in p.resolve().parents, "source path escape: " + name)
    return p


def mapping(n, keep):
    ids = [j * n // keep for j in range(keep)]
    require(0 < keep <= n and len(set(ids)) == keep, "invalid retained expert count")
    return ids, {old: new for new, old in enumerate(ids)}


def transform(k, meta, remap, n):
    meta = copy.deepcopy(meta)
    m = EXPERT.match(k)
    if m:
        old = int(m[2])
        if old not in remap:
            return None
        newkey = f"language_model.model.layers.{m[1]}.block_sparse_moe.experts.{remap[old]}.{m[3]}"
        return newkey, meta, "copy"
    r = ROUTER.match(k)
    if r:
        expected = [n, 7168] if r[2] == "weight" else [n]
        require(meta["shape"] == expected and meta["dtype"] == "BF16", "unexpected router format: " + k)
        meta["shape"][0] = len(remap)
        return k, meta, "rows"
    return k, meta, "copy"


def fingerprint(path, header_hash=None):
    st = path.stat()
    d = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if header_hash:
        d["header_sha256"] = header_hash
    return d


def build_plan(src, keep):
    cfg = read_json(src / "config.json")
    tc = cfg["text_config"]
    require(tc["num_hidden_layers"] == 93 and tc["num_experts"] == 896, "unexpected source architecture")
    require(tc["num_experts_per_token"] == 16 and tc["num_shared_experts"] == 2, "unexpected topk/shared experts")
    require(keep >= tc["num_experts_per_token"] and keep % 8 == 0, "invalid target for topk/EP8")
    ids, remap = mapping(tc["num_experts"], keep)
    idx = read_json(src / "quant_model_weights.safetensors.index.json")
    q = read_json(src / "quant_model_description.json")
    require(q.get("group_size") == 32, "unexpected quantization group size")
    wm = idx["weight_map"]
    seen = set()
    ex = collections.defaultdict(lambda: collections.defaultdict(set))
    routers = collections.Counter()
    total = 0
    old_total = 0
    output_keys = set()
    shards = []
    for sn in sorted(set(wm.values())):
        p = source_file(src, sn)
        hn, h, hh = read_header(p)
        items = []
        for k, m in h.items():
            if k == "__metadata__":
                continue
            require(k not in seen and wm.get(k) == sn, "index/header conflict: " + k)
            seen.add(k)
            old_total += m["data_offsets"][1] - m["data_offsets"][0]
            match = EXPERT.match(k)
            if match:
                layer_id, e, suffix = match.groups()
                require(suffix in SIX and q.get(k) == "W4A8_MXFP", "unexpected expert entry: " + k)
                ex[int(layer_id)][int(e)].add(suffix)
                expected_shapes = {
                    "w1.weight": [3072, 1792],
                    "w3.weight": [3072, 1792],
                    "w2.weight": [3584, 1536],
                    "w1.weight_scale": [3072, 112],
                    "w3.weight_scale": [3072, 112],
                    "w2.weight_scale": [3584, 96],
                }
                require(m["shape"] == expected_shapes[suffix] and m["dtype"] == "U8", "unexpected expert tensor format")
            if ROUTER.match(k):
                routers[int(ROUTER.match(k)[1])] += 1
            tr = transform(k, m, remap, tc["num_experts"])
            if tr is None:
                continue
            nk, nm, op = tr
            require(nk not in output_keys, "output key collision")
            output_keys.add(nk)
            size = math.prod(nm["shape"]) * DTYPE_BYTES[nm["dtype"]]
            total += size
            items.append(
                {"source_key": k, "key": nk, "meta": nm, "op": op, "source_offsets": m["data_offsets"], "size": size}
            )
        if items:
            items.sort(key=lambda x: x["source_offsets"][0])
            shards.append(
                {
                    "source": sn,
                    "source_header_len": hn,
                    "identity": fingerprint(p, hh),
                    "metadata": h.get("__metadata__"),
                    "items": items,
                }
            )
    require(seen == set(wm), "index key set mismatch")
    require(set(ex) == set(range(1, 93)), "unexpected MoE layer IDs")
    for layer_id, experts in ex.items():
        require(set(experts) == set(range(896)), "missing expert IDs in layer " + str(layer_id))
        require(all(s == SIX for s in experts.values()), "missing expert weight or scale")
    require(routers == collections.Counter({layer_id: 2 for layer_id in range(1, 93)}), "incomplete routers")
    require(idx["metadata"]["total_size"] == old_total, "source total_size mismatch")
    for i, s in enumerate(shards, 1):
        s["output"] = f"quant_model_weights_ep256-{i:05d}-of-{len(shards):05d}.safetensors"
    # Only explicit top-level sidecars and optional assets; exclude previous job caches.
    assets = []
    for p in sorted(src.iterdir()):
        if (
            p.is_file()
            and not p.name.endswith(".safetensors")
            and p.name != "quant_model_weights.safetensors.index.json"
        ):
            source_file(src, p.name)
            assets.append({"name": p.name, **fingerprint(p), "sha256": digest(p.read_bytes())})
    if (src / "optional").exists():
        for p in sorted((src / "optional").rglob("*")):
            require(not p.is_symlink(), "optional symlink: " + str(p))
            if p.is_file():
                rel = p.relative_to(src).as_posix()
                assets.append({"name": rel, **fingerprint(p), "sha256": file_hash(p)})
    # All JSON optional/ references must refer to a copied asset.
    asset_names = {a["name"] for a in assets}

    def refs(x):
        if isinstance(x, dict):
            for v in x.values():
                refs(v)
        elif isinstance(x, list):
            for v in x:
                refs(v)
        elif isinstance(x, str) and x.startswith("optional/"):
            require(x in asset_names, "missing referenced asset: " + x)

    for a in assets:
        if a["name"].endswith(".json"):
            refs(read_json(src / a["name"]))
    plan = {
        "version": 1,
        "source": str(src),
        "keep_experts": keep,
        "keep_ids": ids,
        "source_config_sha256": digest((src / "config.json").read_bytes()),
        "source_index_sha256": digest((src / "quant_model_weights.safetensors.index.json").read_bytes()),
        "source_payload_bytes": old_total,
        "output_payload_bytes": total,
        "output_tensors": len(output_keys),
        "shards": shards,
        "assets": assets,
    }
    plan["plan_sha256"] = digest(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    return plan


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def new_header(shard):
    h = {}
    if shard["metadata"] is not None:
        h["__metadata__"] = shard["metadata"]
    offset = 0
    for item in shard["items"]:
        meta = copy.deepcopy(item["meta"])
        meta["data_offsets"] = [offset, offset + item["size"]]
        h[item["key"]] = meta
        offset += item["size"]
    raw = json.dumps(h, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * ((-len(raw)) % 8)
    return h, raw


def selected_data(source, shard, item, keep_ids):
    base = 8 + shard["source_header_len"] + item["source_offsets"][0]
    if item["op"] == "copy":
        source.seek(base)
        left = item["size"]
        while left:
            b = source.read(min(CHUNK, left))
            require(bool(b), "unexpected source EOF")
            left -= len(b)
            yield b
    else:
        source_rows = 896
        row_bytes = (item["source_offsets"][1] - item["source_offsets"][0]) // source_rows
        for old in keep_ids:
            source.seek(base + old * row_bytes)
            b = source.read(row_bytes)
            require(len(b) == row_bytes, "truncated router row")
            yield b


def verify_shard(path, shard, tensor_hashes, safe=True):
    n, h, _ = read_header(path)
    expected, _ = new_header(shard)
    require(h == expected, "output header differs from plan: " + str(path))
    with open(path, "rb") as f:
        for item in shard["items"]:
            a, b = h[item["key"]]["data_offsets"]
            f.seek(8 + n + a)
            hh = hashlib.sha256()
            left = b - a
            while left:
                data = f.read(min(CHUNK, left))
                require(bool(data), "output EOF")
                left -= len(data)
                hh.update(data)
            require(hh.hexdigest() == tensor_hashes[item["key"]], "payload hash mismatch: " + item["key"])
    if safe:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as sf:
            require(set(sf.keys()) == set(h) - {"__metadata__"}, "safetensors reader mismatch")


def write_shard(src, dst, shard, ids, progress):
    sp = source_file(src, shard["source"])
    require(fingerprint(sp) == {k: v for k, v in shard["identity"].items() if k != "header_sha256"}, "source changed")
    hn, _, hh = read_header(sp)
    require(hn == shard["source_header_len"] and hh == shard["identity"]["header_sha256"], "source header changed")
    final = dst / shard["output"]
    temp = final.with_name(final.name + ".writing")
    if temp.exists():
        temp.unlink()  # Exact in-task temporary file only; final files are never overwritten.
    require(not final.exists(), "unregistered output already exists")
    _, raw = new_header(shard)
    hashes = {}
    written = 0
    last = time.monotonic()
    with open(sp, "rb") as source, open(temp, "xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for item in shard["items"]:
            h = hashlib.sha256()
            count = 0
            for data in selected_data(source, shard, item, ids):
                output.write(data)
                h.update(data)
                count += len(data)
                written += len(data)
                if time.monotonic() - last > 20:
                    progress({"phase": "copy", "current_shard": shard["output"], "current_shard_bytes": written})
                    last = time.monotonic()
            require(count == item["size"], "copied tensor size mismatch")
            hashes[item["key"]] = h.hexdigest()
        output.flush()
        os.fsync(output.fileno())
    require(
        fingerprint(sp) == {k: v for k, v in shard["identity"].items() if k != "header_sha256"},
        "source changed during copy",
    )
    progress({"phase": "verify_shard", "current_shard": shard["output"], "current_shard_bytes": written})
    verify_shard(temp, shard, hashes)
    os.replace(temp, final)
    return {
        "output": shard["output"],
        "source": shard["source"],
        "payload_bytes": written,
        "tensor_hashes": hashes,
        "file_size": final.stat().st_size,
    }


def write_assets(src, dst, plan):
    remap = {old: new for new, old in enumerate(plan["keep_ids"])}
    expected_hashes = {}
    for asset in plan["assets"]:
        name = asset["name"]
        sp = source_file(src, name)
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        require(file_hash(sp) == asset["sha256"], "source sidecar changed: " + name)
        if name == "config.json":
            c = read_json(sp)
            c["text_config"]["num_experts"] = plan["keep_experts"]
            atomic_json(target, c)
        elif name == "quant_model_description.json":
            q = read_json(sp)
            out = {}
            for k, v in q.items():
                m = EXPERT.match(k)
                if m:
                    if int(m[2]) not in remap:
                        continue
                    k = f"language_model.model.layers.{m[1]}.block_sparse_moe.experts.{remap[int(m[2])]}.{m[3]}"
                require(k not in out, "quantization key collision")
                out[k] = v
            atomic_json(target, out)
        else:
            shutil.copyfile(sp, target)
            require(file_hash(target) == asset["sha256"], "sidecar copy mismatch: " + name)
        expected_hashes[name] = file_hash(target)
    return expected_hashes


def validate_final(dst, src, plan, asset_hashes):
    cfg = read_json(dst / "config.json")
    original = read_json(src / "config.json")
    original["text_config"]["num_experts"] = plan["keep_experts"]
    require(cfg == original, "unexpected config modification")
    q = read_json(dst / "quant_model_description.json")
    index = read_json(dst / "quant_model_weights.safetensors.index.json")
    actual = {}
    total = 0
    experts = collections.defaultdict(lambda: collections.defaultdict(set))
    for sh in plan["shards"]:
        _, h, _ = read_header(dst / sh["output"])
        for k, m in h.items():
            if k == "__metadata__":
                continue
            require(k not in actual, "duplicate output tensor")
            actual[k] = sh["output"]
            total += m["data_offsets"][1] - m["data_offsets"][0]
            ex = EXPERT.match(k)
            if ex:
                experts[int(ex[1])][int(ex[2])].add(ex[3])
                require(q.get(k) == "W4A8_MXFP", "missing quant metadata")
    require(actual == index["weight_map"], "final index mismatch")
    require(total == plan["output_payload_bytes"] == index["metadata"]["total_size"], "final total_size mismatch")
    require(len(actual) == plan["output_tensors"], "final tensor count mismatch")
    require(set(experts) == set(range(1, 93)), "final MoE layers mismatch")
    for ex in experts.values():
        require(
            set(ex) == set(range(plan["keep_experts"])) and all(v == SIX for v in ex.values()),
            "final experts incomplete",
        )
    for k in q:
        ex = EXPERT.match(k)
        if ex:
            require(int(ex[2]) < plan["keep_experts"] and k in actual, "stale quant entry")
    for name, expected in asset_hashes.items():
        require(file_hash(dst / name) == expected, "changed output sidecar")
    return {
        "status": "passed",
        "validation": "all_retained_tensor_sha256_verified_after_write",
        "layers": 93,
        "moe_layers": 92,
        "experts_per_layer": plan["keep_experts"],
        "topk": 16,
        "shared_experts_per_layer": 2,
        "payload_bytes": total,
        "payload_GiB": total / 1024**3,
        "tensors": len(actual),
        "shards": len(plan["shards"]),
        "model_loading": "not_tested",
        "accuracy": "not_tested",
        "capacity": "not_tested",
    }


def run(args):
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    work = Path(args.work).resolve()
    require(src.is_dir() and src != dst and src not in dst.parents and dst not in src.parents, "invalid source/output")
    require(
        not any(p.is_symlink() for p in (Path(args.src), Path(args.dst), Path(args.work))), "symlink endpoint forbidden"
    )
    work.mkdir(parents=True, exist_ok=True)
    (work / "manifests/shards").mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = open(work / "job.lock", "a")  # noqa: SIM115 -- held until the function finally block
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time()
    base = {"pid": os.getpid(), "source": str(src), "output": str(dst), "started_unix": start}
    progress_lock = threading.Lock()
    state_extra = {}

    def progress(extra):
        with progress_lock:
            state_extra.update(extra)
            state = {**base, **state_extra, "updated_unix": time.time()}
            atomic_json(work / "status.json", state)
            print(json.dumps(state), flush=True)

    try:
        progress({"phase": "audit_source"})
        plan = build_plan(src, args.keep_experts)
        pp = work / "manifests/plan.json"
        if pp.exists():
            require(read_json(pp)["plan_sha256"] == plan["plan_sha256"], "resume source/plan changed")
        else:
            atomic_json(pp, plan)
        atomic_json(
            work / "manifests/expert_map.json",
            {
                "rule": "floor(j*896/256)",
                "keep_ids": plan["keep_ids"],
                "new_to_old": {str(n): o for n, o in enumerate(plan["keep_ids"])},
                "layers": list(range(1, 93)),
            },
        )
        if args.mode == "plan":
            progress(
                {
                    "phase": "planned",
                    "payload_bytes": plan["output_payload_bytes"],
                    "tensors": plan["output_tensors"],
                    "shards": len(plan["shards"]),
                }
            )
            return
        require(args.keep_experts == 256, "production writer currently restricted to agreed 256 experts")
        dst.mkdir(parents=True, exist_ok=True)
        marker = dst / "PRUNING_INCOMPLETE"
        if not marker.exists():
            require(not any(dst.iterdir()), "non-empty output without ownership marker")
            marker.write_text(plan["plan_sha256"] + "\n")
        require(marker.read_text().strip() == plan["plan_sha256"], "output belongs to another plan")
        expected_files = {s["output"] for s in plan["shards"]}
        require(
            not any(
                x.name not in expected_files
                and x.name not in {"PRUNING_INCOMPLETE"}
                and not x.name.endswith(".writing")
                for x in dst.glob("*.safetensors")
            ),
            "unexpected shard in output",
        )
        asset_hashes = write_assets(src, dst, plan)
        atomic_json(work / "manifests/output_asset_hashes.json", asset_hashes)
        new_wm = {}
        done = 0

        def process_shard(pos, shard):
            receipt_path = work / "manifests/shards" / f"{pos:05d}.json"
            if receipt_path.exists():
                r = read_json(receipt_path)
                require(
                    r["plan_sha256"] == plan["plan_sha256"] and r["output"] == shard["output"], "bad resume receipt"
                )
                progress({"phase": "verify_resume", "current_shard": shard["output"]})
                verify_shard(dst / shard["output"], shard, r["tensor_hashes"])
            else:
                progress({"phase": "copy", "current_shard": shard["output"]})
                # Recover a fully renamed but unregistered file only by rechecking every byte against source.
                if (dst / shard["output"]).exists():
                    hashes = {}
                    with open(src / shard["source"], "rb") as f:
                        for item in shard["items"]:
                            h = hashlib.sha256()
                            for b in selected_data(f, shard, item, plan["keep_ids"]):
                                h.update(b)
                            hashes[item["key"]] = h.hexdigest()
                    verify_shard(dst / shard["output"], shard, hashes)
                    r = {
                        "output": shard["output"],
                        "source": shard["source"],
                        "payload_bytes": sum(x["size"] for x in shard["items"]),
                        "tensor_hashes": hashes,
                    }
                else:
                    r = write_shard(src, dst, shard, plan["keep_ids"], progress)
                r["plan_sha256"] = plan["plan_sha256"]
                atomic_json(receipt_path, r)
            return shard, r

        workers = getattr(args, "workers", 1)
        require(1 <= workers <= 4, "workers must be in 1..4")
        progress(
            {
                "phase": "copy",
                "completed_shards": 0,
                "total_shards": len(plan["shards"]),
                "completed_bytes": 0,
                "total_bytes": plan["output_payload_bytes"],
                "workers": workers,
            }
        )
        # Each worker owns a distinct source/output shard and receipt. Status updates are serialized.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_shard, pos, shard) for pos, shard in enumerate(plan["shards"], 1)]
            for completed, future in enumerate(as_completed(futures), 1):
                shard, r = future.result()
                done += r["payload_bytes"]
                for item in shard["items"]:
                    new_wm[item["key"]] = shard["output"]
                progress(
                    {
                        "phase": "shard_complete",
                        "completed_shards": completed,
                        "total_shards": len(plan["shards"]),
                        "completed_bytes": done,
                        "total_bytes": plan["output_payload_bytes"],
                        "elapsed_seconds": time.time() - start,
                    }
                )
        for sh in plan["shards"]:
            require(
                fingerprint(src / sh["source"]) == {k: v for k, v in sh["identity"].items() if k != "header_sha256"},
                "source changed before completion",
            )
        atomic_json(
            dst / "quant_model_weights.safetensors.index.json", {"metadata": {"total_size": done}, "weight_map": new_wm}
        )
        progress({"phase": "final_validation", "completed_bytes": done})
        result = validate_final(dst, src, plan, asset_hashes)
        result.update(
            {
                "source": str(src),
                "output": str(dst),
                "plan_sha256": plan["plan_sha256"],
                "elapsed_seconds": time.time() - start,
            }
        )
        atomic_json(work / "reports/validation.json", result)
        atomic_json(
            dst / "expert_pruning_manifest.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "source": str(src),
                "keep_ids": plan["keep_ids"],
                "selection": "deterministic_uniform_capacity_only",
                "validation": result,
            },
        )
        marker.unlink()
        atomic_json(dst / "PRUNING_COMPLETE.json", result)
        progress({"phase": "completed", **result})
    except BaseException as e:
        progress({"phase": "failed", "error": repr(e)})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep-experts", type=int, default=256)
    ap.add_argument("--mode", choices=["plan", "run"], default="plan")
    ap.add_argument("--workers", type=int, default=1)
    run(ap.parse_args())
