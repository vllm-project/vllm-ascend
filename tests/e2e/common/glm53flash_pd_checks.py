# SPDX-License-Identifier: Apache-2.0
"""Output validity and state-transfer assertions for the Flash PD test."""

import json
import math
from collections import defaultdict


def check_checkpoint(model):
    """Fail before starting workers if this runner lacks the real cached crop."""
    required = ("config.json", "quant_model_description.json", "model.safetensors.index.json", "tokenizer.json")
    for name in required:
        assert (model / name).is_file(), f"Missing real Flash checkpoint file: {model / name}; sync the A5 model cache"
    config = json.loads((model / "config.json").read_text())["text_config"]
    expected = {
        "num_hidden_layers": 9,
        "num_nextn_predict_layers": 1,
        "hidden_size": 4096,
        "n_routed_experts": 288,
        "vocab_size": 154880,
    }
    for key, value in expected.items():
        assert config[key] == value, f"Unexpected real Flash crop {key}: {config[key]}"
    assert config["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4, 5, 6, 8]
    assert config["linear_attn_config"]["full_attn_layers"] == [3, 7]
    quant = json.loads((model / "quant_model_description.json").read_text())
    assert "W8A8_MXFP8" in quant.values(), "Expected the real MXFP8 checkpoint"
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert index["weight_map"] and index["metadata"]["total_size"] > 0
    for shard in set(index["weight_map"].values()):
        path = model / shard
        assert path.is_file() and path.stat().st_size > 0, f"Missing or empty checkpoint shard: {path}"


def check_completion(result, output_tokens):
    assert "error" not in result, result
    assert result["usage"]["completion_tokens"] == output_tokens
    choice = result["choices"][0]
    assert choice["finish_reason"] == "length"
    logs = choice["logprobs"]
    assert len(logs["tokens"]) == len(logs["token_logprobs"]) == output_tokens, "Unexpected token/logprob count"
    assert all(math.isfinite(value) for value in logs["token_logprobs"]), "Non-finite token log probabilities"
    return logs


def observe_state_copy(
    original, receiver, spec, remote_pcp_rank, remote_tp_rank, remote_tp_size, entries, metadata, local, remote, lengths
):
    start = len(lengths)
    result = original(
        receiver, spec, remote_pcp_rank, remote_tp_rank, remote_tp_size, entries, metadata, local, remote, lengths
    )
    remote_rank = remote_tp_rank * len(metadata.metadata_by_pcp_rank) + remote_pcp_rank
    rm = metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
    targets = []
    for (li, ri), requests in entries.items():
        for request, lbs, rbs in requests:
            for lb, rb in zip(lbs, rbs):
                for ci in range(len(receiver.block_lens[li])):
                    targets.append(
                        dict(
                            request=request,
                            layer=receiver.layer_names[li],
                            cache=ci,
                            block=lb,
                            local=receiver.kv_caches_base_addr[li][ci] + lb * receiver.block_strides[li][ci],
                            remote=rm.kv_caches_base_addr[ri][ci] + rb * metadata.block_strides[ri][ci],
                            local_bytes=receiver.block_lens[li][ci],
                            remote_bytes=metadata.block_lens[ri][ci],
                        )
                    )
    record = dict(
        remote_rank=remote_rank,
        remote_size=len(metadata.metadata_by_pcp_rank) * remote_tp_size,
        local_size=receiver.pcp_size * receiver.tp_size,
        replicated=getattr(spec, "tp_replicated", False),
        targets=targets,
        ranges=list(zip(local[start:], remote[start:], lengths[start:])),
    )
    return result, record


def check_state_pulls(workers):
    assert len(workers) == 2 and all(w["copies"] for w in workers), "Both DP workers must have copy records"
    summary = []
    for worker in workers:
        groups = defaultdict(lambda: {"spans": [], "remote_ranks": set(), "bytes": None})
        for record in worker["copies"]:
            assert record["local_size"] == 1 and record["remote_size"] == 2
            assert not record["replicated"], "KDA states must be head-sharded"
            by_target = defaultdict(list)
            for local, remote, n in record["ranges"]:
                assert n > 0
                matches = [
                    t
                    for t in record["targets"]
                    if t["local"] <= local
                    and local + n <= t["local"] + t["local_bytes"]
                    and t["remote"] <= remote
                    and remote + n <= t["remote"] + t["remote_bytes"]
                ]
                assert len(matches) == 1, "Each copied range must belong to one request/layer/state"
                t = matches[0]
                key = (t["request"], t["layer"], t["cache"], t["block"])
                group = groups[key]
                assert group["bytes"] in (None, t["local_bytes"])
                group["bytes"] = t["local_bytes"]
                group["spans"].append((local - t["local"], local - t["local"] + n))
                group["remote_ranks"].add(record["remote_rank"])
                by_target[key].append((remote - t["remote"], remote - t["remote"] + n))
            for t in record["targets"]:
                key = (t["request"], t["layer"], t["cache"], t["block"])
                spans = sorted(by_target[key])
                cursor = 0
                assert spans, "Missing state copy"
                for lo, hi in spans:
                    assert lo == cursor, "Remote shard copy has a gap or overlap"
                    cursor = hi
                assert cursor == t["remote_bytes"] == t["local_bytes"] // 2
        assert groups
        for group in groups.values():
            assert group["remote_ranks"] == {0, 1}, "Both producer head shards required"
            cursor = 0
            for lo, hi in sorted(group["spans"]):
                assert lo == cursor, "Local assembled state has a gap or overlap"
                cursor = hi
            assert cursor == group["bytes"], "Incomplete assembled state"
        summary.append(
            {
                "state_blocks_checked": len(groups),
                "requests": len({k[0] for k in groups}),
                "bytes_checked": sum(g["bytes"] for g in groups.values()),
            }
        )
    return summary
