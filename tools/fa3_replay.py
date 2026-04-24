import argparse
from pathlib import Path
from typing import Any

import torch
import torch_npu


def _load_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"{path} is not a valid FA3 dump file")
    return payload


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


def _take_first_dim(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return tensor[:0].clone()
    if len(indices) == 1:
        return tensor.narrow(0, indices[0], 1).contiguous()
    return torch.cat([tensor.narrow(0, idx, 1) for idx in indices], dim=0).contiguous()


def _canonicalize_output(output: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
    if output.dim() == 4:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.squeeze(2).contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2, 3).squeeze(2).contiguous()
    if output.dim() == 3:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2).contiguous()
    raise ValueError(
        f"Unexpected FA3 output shape {tuple(output.shape)} for batch_size={batch_size}, num_heads={num_heads}"
    )


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_flat = lhs.reshape(lhs.shape[0], -1).float()
    rhs_flat = rhs.reshape(rhs.shape[0], -1).float()
    return torch.nn.functional.cosine_similarity(lhs_flat, rhs_flat, dim=-1)


def _run_fa3_operator(payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    inputs = payload["inputs"]
    op_kwargs = payload["op_kwargs"]
    query = _move_to_device(inputs["query"], device)
    key_cache = _move_to_device(inputs["key_cache"], device)
    query_rope = _move_to_device(inputs["query_rope"], device)
    key_rope = _move_to_device(inputs["key_rope"], device)
    block_table = _move_to_device(inputs["block_table"], device)
    dequant_scale_query = _move_to_device(inputs["dequant_scale_query"], device)
    dequant_scale_key = _move_to_device(inputs["dequant_scale_key"], device)
    dequant_scale_value = _move_to_device(inputs["dequant_scale_value"], device)

    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key_cache,
        key_cache,
        query_rope=query_rope,
        key_rope=key_rope,
        num_query_heads=op_kwargs["num_query_heads"],
        num_key_value_heads=op_kwargs["num_key_value_heads"],
        input_layout=op_kwargs["input_layout"],
        atten_mask=None,
        sparse_mode=op_kwargs["sparse_mode"],
        softmax_scale=op_kwargs["softmax_scale"],
        query_quant_mode=3,
        key_quant_mode=0,
        value_quant_mode=0,
        dequant_scale_query=dequant_scale_query,
        dequant_scale_key=dequant_scale_key,
        dequant_scale_value=dequant_scale_value,
        block_table=block_table,
        block_size=op_kwargs["block_size"],
        actual_seq_qlen=inputs["actual_seq_qlen"],
        actual_seq_kvlen=inputs["actual_seq_kvlen"],
        return_softmax_lse=True,
    )
    return _canonicalize_output(output, query.shape[0], op_kwargs["num_query_heads"])


def _build_single_req_payload(payload: dict[str, Any], req_idx: int) -> dict[str, Any]:
    single = {
        "meta": dict(payload["meta"]),
        "inputs": {},
        "op_kwargs": dict(payload["op_kwargs"]),
        "debug": {},
    }
    inputs = payload["inputs"]
    local_block_ids = payload["debug"]["per_req_local_block_ids"][req_idx]

    prev_qlen = 0 if req_idx == 0 else inputs["actual_seq_qlen"][req_idx - 1]
    cur_qlen = inputs["actual_seq_qlen"][req_idx]
    single["inputs"]["query"] = inputs["query"][req_idx : req_idx + 1].contiguous()
    single["inputs"]["query_rope"] = inputs["query_rope"][req_idx : req_idx + 1].contiguous()
    single["inputs"]["key_cache"] = _take_first_dim(inputs["key_cache"], local_block_ids)
    single["inputs"]["key_rope"] = _take_first_dim(inputs["key_rope"], local_block_ids)
    single["inputs"]["block_table"] = torch.arange(len(local_block_ids), dtype=torch.int32).unsqueeze(0)
    single["inputs"]["actual_seq_qlen"] = [int(cur_qlen - prev_qlen)]
    single["inputs"]["actual_seq_kvlen"] = [int(inputs["actual_seq_kvlen"][req_idx])]
    single["inputs"]["dequant_scale_query"] = inputs["dequant_scale_query"][req_idx : req_idx + 1].contiguous()
    single["inputs"]["dequant_scale_key"] = inputs["dequant_scale_key"].contiguous()
    single["inputs"]["dequant_scale_value"] = inputs["dequant_scale_value"].contiguous()
    single["debug"]["per_req_local_block_ids"] = [list(range(len(local_block_ids)))]
    return single


def cmd_summary(args: argparse.Namespace) -> None:
    payload = _load_dump(Path(args.path))
    meta = payload["meta"]
    print(f"kind={meta.get('kind')} stage={meta.get('stage')} status={meta.get('status')}")
    print(f"layer={meta.get('layer_name')} pid={meta.get('pid')} dp={meta.get('dp_rank')} tp={meta.get('tp_rank')}")
    if meta.get("error"):
        print(f"error={meta['error']}")

    inputs = payload.get("inputs", {})
    if "query" in inputs:
        print(f"query={tuple(inputs['query'].shape)} dtype={inputs['query'].dtype}")
        print(f"query_rope={tuple(inputs['query_rope'].shape)} dtype={inputs['query_rope'].dtype}")
        print(f"key_cache={tuple(inputs['key_cache'].shape)} dtype={inputs['key_cache'].dtype}")
        print(f"key_rope={tuple(inputs['key_rope'].shape)} dtype={inputs['key_rope'].dtype}")
        print(f"block_table={tuple(inputs['block_table'].shape)}")
        print(f"actual_seq_qlen={inputs['actual_seq_qlen']}")
        print(f"actual_seq_kvlen={inputs['actual_seq_kvlen']}")
    elif "kv_no_split" in inputs:
        print(f"kv_no_split={tuple(inputs['kv_no_split'].shape)} dtype={inputs['kv_no_split'].dtype}")
        print(f"slots={tuple(inputs['slots'].shape)} touched_blocks={inputs['touched_global_blocks']}")


def cmd_replay(args: argparse.Namespace) -> None:
    payload = _load_dump(Path(args.path))
    if payload["meta"].get("kind") != "fa3_forward_decode":
        raise ValueError("Only fa3_forward_decode dumps support replay.")

    device = torch.device(args.device)
    torch.npu.set_device(device)

    batched_output = _run_fa3_operator(payload, device).cpu()
    dumped_output = payload.get("outputs", {}).get("attn_output")

    single_outputs = []
    num_reqs = payload["inputs"]["query"].shape[0]
    for req_idx in range(num_reqs):
        single_payload = _build_single_req_payload(payload, req_idx)
        single_output = _run_fa3_operator(single_payload, device).cpu()
        single_outputs.append(single_output.squeeze(0))
    single_outputs_tensor = torch.stack(single_outputs, dim=0)

    print(f"batched_output_shape={tuple(batched_output.shape)}")
    if dumped_output is not None:
        dumped_output = dumped_output.cpu()
        diff_dump = (batched_output.float() - dumped_output.float()).abs()
        cos_dump = _cosine_similarity(batched_output, dumped_output)
        print(f"replay_vs_dump max_abs_diff={diff_dump.max().item():.6f} cos={cos_dump.tolist()}")

    diff_single = (batched_output.float() - single_outputs_tensor.float()).abs()
    cos_single = _cosine_similarity(batched_output, single_outputs_tensor)
    print(f"replay_vs_single max_abs_diff={diff_single.max().item():.6f} cos={cos_single.tolist()}")

    for req_idx in range(num_reqs):
        req_diff = (batched_output[req_idx].float() - single_outputs_tensor[req_idx].float()).abs().max().item()
        req_cos = _cosine_similarity(
            batched_output[req_idx : req_idx + 1], single_outputs_tensor[req_idx : req_idx + 1]
        ).item()
        print(f"req[{req_idx}] batched_vs_single max_abs_diff={req_diff:.6f} cos={req_cos:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay vLLM Ascend FA3 debug dumps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Print dump metadata and tensor shapes.")
    summary_parser.add_argument("path", help="Path to a .pt FA3 dump file.")
    summary_parser.set_defaults(func=cmd_summary)

    replay_parser = subparsers.add_parser("replay", help="Replay a forward decode dump and compare outputs.")
    replay_parser.add_argument("path", help="Path to a .pt FA3 forward dump file.")
    replay_parser.add_argument("--device", default="npu:0", help="Replay target device. Default: npu:0")
    replay_parser.set_defaults(func=cmd_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
