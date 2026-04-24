import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "meta" not in data:
        raise ValueError(f"{path} is not a valid identical-request dump")
    return data


def _tensor_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    lhs_flat = lhs.reshape(-1).float()
    rhs_flat = rhs.reshape(-1).float()
    cosine = torch.nn.functional.cosine_similarity(lhs_flat.unsqueeze(0), rhs_flat.unsqueeze(0)).item()
    maxdiff = (lhs_flat - rhs_flat).abs().max().item()
    return cosine, maxdiff


def _fmt_meta(meta: dict[str, Any]) -> str:
    return (
        f"kind={meta.get('kind')} stage={meta.get('stage')} step={meta.get('decode_step')} "
        f"dp={meta.get('dp_rank')} tp={meta.get('tp_rank')} name={meta.get('name')}"
    )


def _compare_python_values(lhs: Any, rhs: Any) -> str:
    return f"equal={lhs == rhs} lhs={lhs} rhs={rhs}"


def _compare_tensor_group(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float] | None:
    if name in {"query_lens", "logits_indices"}:
        return None
    if lhs.shape != rhs.shape:
        return None
    return _tensor_stats(lhs, rhs)


def _compare_same_file_req01(name: str, tensor: torch.Tensor) -> tuple[float, float] | None:
    if name in {"query_lens", "logits_indices"}:
        return None
    if tensor.ndim == 0 or tensor.shape[0] < 2:
        return None
    return _tensor_stats(tensor[0], tensor[1])


def _segment_by_query_lens(tensor: torch.Tensor, query_lens: torch.Tensor) -> list[torch.Tensor]:
    query_lens_list = [int(x) for x in query_lens.cpu().tolist()]
    if len(query_lens_list) < 2:
        return []
    total = sum(query_lens_list)
    if tensor.shape[0] < total:
        return []
    segments = []
    start = 0
    for length in query_lens_list:
        end = start + length
        segments.append(tensor[start:end].contiguous())
        start = end
    return segments


def _compare_req01_with_lens(tensor: torch.Tensor, query_lens: torch.Tensor) -> tuple[float, float] | None:
    segments = _segment_by_query_lens(tensor, query_lens)
    if len(segments) < 2:
        return None
    if segments[0].shape != segments[1].shape:
        return _tensor_stats(segments[0][-1], segments[1][-1])
    return _tensor_stats(segments[0], segments[1])


def _select_query_lens_for_tensor(tensors: dict[str, Any], tensor_name: str) -> torch.Tensor | None:
    if tensor_name in {"q_nope", "q_pe", "attn_output"}:
        value = tensors.get("decode_query_lens")
        return value if isinstance(value, torch.Tensor) else None
    if tensor_name in {"attn_out", "block_out"}:
        value = tensors.get("query_lens")
        return value if isinstance(value, torch.Tensor) else None
    if tensor_name in {"output_prefill", "o_proj_input_prefill_slice"}:
        value = tensors.get("prefill_query_lens")
        return value if isinstance(value, torch.Tensor) else None
    if tensor_name in {"output_decode", "o_proj_input_decode_slice"}:
        value = tensors.get("decode_query_lens")
        return value if isinstance(value, torch.Tensor) else None
    return None


def _group_tensor_files(paths: list[tuple[Path, dict[str, Any]]]) -> dict[tuple, dict[int, tuple[Path, dict[str, Any]]]]:
    groups: dict[tuple, dict[int, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    for path, data in paths:
        meta = data.get("meta", {})
        key = (
            meta.get("stage"),
            meta.get("decode_step"),
            meta.get("tp_rank"),
            meta.get("name"),
        )
        groups[key][meta.get("dp_rank", -1)] = (path, data)
    return groups


def _print_tensor_comparison(group_key: tuple, pair: dict[int, tuple[Path, dict[str, Any]]]) -> dict[str, tuple[float, float]]:
    path0, data0 = pair[0]
    path1, data1 = pair[1]
    print("=" * 140)
    print(f"stage={group_key[0]} step={group_key[1]} tp={group_key[2]} name={group_key[3]}")
    print("dp0:", path0.name)
    print("dp1:", path1.name)
    results = {}
    for name, lhs in data0.get("tensors", {}).items():
        rhs = data1.get("tensors", {}).get(name)
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            if lhs is not None:
                print(f"  {name}: {_compare_python_values(lhs, rhs)}")
            continue
        stats = _compare_tensor_group(name, lhs, rhs)
        if stats is None:
            if lhs.shape != rhs.shape:
                print(
                    f"  {name}: shape mismatch dp0={tuple(lhs.shape)} dp1={tuple(rhs.shape)} "
                    f"dtype0={lhs.dtype} dtype1={rhs.dtype}"
                )
            else:
                print(f"  {name}: shape={tuple(lhs.shape)} dtype={lhs.dtype}")
            continue
        cosine, maxdiff = stats
        results[name] = (cosine, maxdiff)
        print(f"  {name}: cosine(dp0,dp1)={cosine:.6f} maxdiff={maxdiff:.6f}")
    return results


def _print_same_file_req_comparison(path: Path, data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    print("=" * 140)
    print(path.name)
    print(_fmt_meta(data.get("meta", {})))
    tensors = data.get("tensors", {})
    results = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, list) and len(tensor) >= 2:
                print(f"  {name}: req0={tensor[0]} req1={tensor[1]} equal={tensor[0] == tensor[1]}")
            else:
                print(f"  {name}: {tensor}")
            continue
        query_lens = _select_query_lens_for_tensor(tensors, name)
        if query_lens is not None:
            stats = _compare_req01_with_lens(tensor, query_lens)
        else:
            stats = _compare_same_file_req01(name, tensor)
        if stats is None:
            print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            continue
        cosine, maxdiff = stats
        results[name] = (cosine, maxdiff)
        print(f"  {name}: cosine(req0,req1)={cosine:.6f} maxdiff={maxdiff:.6f}")
    num_decodes = tensors.get("num_decodes")
    num_prefills = tensors.get("num_prefills")
    if isinstance(num_decodes, int) and isinstance(num_prefills, int):
        if num_decodes > 0 and num_prefills > 0:
            print("  phase_mismatch=True (mixed file contains both decode and prefill requests)")
    return results


def _collect_tensor_results(paths: list[tuple[Path, dict[str, Any]]]) -> dict[str, list[tuple[float, float]]]:
    grouped = _group_tensor_files(paths)
    aggregated: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for group_key in sorted(grouped):
        pair = grouped[group_key]
        if 0 not in pair or 1 not in pair:
            continue
        results = _print_tensor_comparison(group_key, pair)
        for tensor_name, stats in results.items():
            aggregated[group_key[3] + "::" + tensor_name].append(stats)
    return aggregated


def _collect_same_file_results(paths: list[tuple[Path, dict[str, Any]]]) -> dict[str, list[tuple[float, float]]]:
    aggregated: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path, data in sorted(paths, key=lambda item: item[0].name):
        results = _print_same_file_req_comparison(path, data)
        meta = data.get("meta", {})
        for tensor_name, stats in results.items():
            aggregated[str(meta.get("name")) + "::" + tensor_name].append(stats)
    return aggregated


def _load_logits(paths: list[Path], stage: str | None, step: int | None) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in paths:
        data = _load(path)
        meta = data.get("meta", {})
        if meta.get("kind") != "logits":
            continue
        if stage and meta.get("stage") != stage:
            continue
        if step is not None and meta.get("decode_step") != step:
            continue
        result.append((path, data))
    return result


def _print_logits(logit_paths: list[tuple[Path, dict[str, Any]]]) -> None:
    for path, data in sorted(logit_paths, key=lambda item: item[0].name):
        meta = data.get("meta", {})
        summary = data.get("summary", {})
        print("=" * 140)
        print(path.name)
        print(_fmt_meta(meta))
        print("  argmax_ids =", summary.get("argmax_ids"))
        print("  cosine_to_req0 =", summary.get("cosine_to_req0"))
        print("  max_abs_diff_to_req0 =", summary.get("max_abs_diff_to_req0"))


def _print_verdict(aggregated: dict[str, list[tuple[float, float]]]) -> None:
    print("=" * 140)
    print("VERDICT")

    def worst(key_prefix: str) -> tuple[float, float] | None:
        stats = []
        for key, values in aggregated.items():
            if key_prefix in key:
                stats.extend(values)
        if not stats:
            return None
        worst_cos = min(cos for cos, _ in stats)
        worst_diff = max(diff for _, diff in stats)
        return worst_cos, worst_diff

    decode_attn = worst("decode_attention::attn_output")
    prefill_attn = worst("prefill_attention::attn_output")
    output_decode = worst("output_decode_tensor::output_decode")
    decode_assign = worst("o_proj_input_after_decode_assign::o_proj_input_decode_slice")
    output_prefill = worst("output_prefill_tensor::output_prefill")
    prefill_assign = worst("o_proj_input_after_prefill_assign::o_proj_input_prefill_slice")
    prepared_input_ids = worst("input_ids::input_ids")
    prepared_positions = worst("input_ids::positions")
    sampled_token_ids = worst("sampled_token_ids::sampled_token_ids")
    attn_out = worst("attn_out_tensor::attn_out")
    block_out = worst("block_out_tensor::block_out")
    sample_hidden = worst("sample_hidden_states::sample_hidden_states")

    print("  decode_attention.attn_output =", decode_attn)
    print("  prefill_attention.attn_output =", prefill_attn)
    print("  output_decode_tensor.output_decode =", output_decode)
    print("  o_proj_input_after_decode_assign.o_proj_input_decode_slice =", decode_assign)
    print("  output_prefill_tensor.output_prefill =", output_prefill)
    print("  o_proj_input_after_prefill_assign.o_proj_input_prefill_slice =", prefill_assign)
    print("  prepare_inputs.input_ids =", prepared_input_ids)
    print("  prepare_inputs.positions =", prepared_positions)
    print("  sample.sampled_token_ids =", sampled_token_ids)
    print("  attn_out_tensor.attn_out =", attn_out)
    print("  block_out_tensor.block_out =", block_out)
    print("  sample_hidden_states =", sample_hidden)

    if decode_attn and decode_attn[1] > 1e-4:
        print("  likely first bad point: decode attention")
    elif prefill_attn and prefill_attn[1] > 1e-4:
        print("  likely first bad point: prefill attention")
    elif prepared_input_ids and prepared_input_ids[1] > 1e-4:
        print("  likely first bad point: next-step input_ids preparation")
    elif prepared_positions and prepared_positions[1] > 1e-4:
        print("  likely first bad point: next-step positions preparation")
    elif sampled_token_ids and sampled_token_ids[1] > 1e-4:
        print("  likely first bad point: sampled token ids")
    elif output_decode and output_decode[1] > 1e-4:
        print("  likely first bad point: decode branch output before o_proj_input assignment")
    elif decode_assign and decode_assign[1] > 1e-4:
        print("  likely first bad point: assigning decode branch into o_proj_input")
    elif output_prefill and output_prefill[1] > 1e-4:
        print("  likely first bad point: prefill branch output before o_proj_input assignment")
    elif prefill_assign and prefill_assign[1] > 1e-4:
        print("  likely first bad point: assigning prefill branch into o_proj_input")
    elif attn_out and attn_out[1] > 1e-4:
        print("  likely first bad point: attention output aggregation / o_proj_input")
    elif block_out and block_out[1] > 1e-4:
        print("  likely first bad point: o_proj or post-attention block output")
    elif sample_hidden and sample_hidden[1] > 1e-4:
        print("  likely first bad point: hidden state selection before compute_logits")
    else:
        print("  no divergence detected in compared tensor packs")


def _print_req_verdict(aggregated: dict[str, list[tuple[float, float]]]) -> None:
    print("=" * 140)
    print("REQ0 VS REQ1 VERDICT")

    def worst(key_prefix: str) -> tuple[float, float] | None:
        stats = []
        for key, values in aggregated.items():
            if key_prefix in key:
                stats.extend(values)
        if not stats:
            return None
        worst_cos = min(cos for cos, _ in stats)
        worst_diff = max(diff for _, diff in stats)
        return worst_cos, worst_diff

    decode_q = worst("decode_attention::q_nope")
    decode_q_rope = worst("decode_attention::q_pe")
    decode_attn = worst("decode_attention::attn_output")
    output_decode = worst("output_decode_tensor::output_decode")
    decode_assign = worst("o_proj_input_after_decode_assign::o_proj_input_decode_slice")
    output_prefill = worst("output_prefill_tensor::output_prefill")
    prefill_assign = worst("o_proj_input_after_prefill_assign::o_proj_input_prefill_slice")
    prepared_input_ids = worst("input_ids::input_ids")
    prepared_positions = worst("input_ids::positions")
    sampled_token_ids = worst("sampled_token_ids::sampled_token_ids")
    attn_out = worst("attn_out_tensor::attn_out")
    block_out = worst("block_out_tensor::block_out")
    sample_hidden = worst("sample_hidden_states::sample_hidden_states")

    print("  decode_attention.q_nope =", decode_q)
    print("  decode_attention.q_pe =", decode_q_rope)
    print("  decode_attention.attn_output =", decode_attn)
    print("  output_decode_tensor.output_decode =", output_decode)
    print("  o_proj_input_after_decode_assign.o_proj_input_decode_slice =", decode_assign)
    print("  output_prefill_tensor.output_prefill =", output_prefill)
    print("  o_proj_input_after_prefill_assign.o_proj_input_prefill_slice =", prefill_assign)
    print("  prepare_inputs.input_ids =", prepared_input_ids)
    print("  prepare_inputs.positions =", prepared_positions)
    print("  sample.sampled_token_ids =", sampled_token_ids)
    print("  attn_out_tensor.attn_out =", attn_out)
    print("  block_out_tensor.block_out =", block_out)
    print("  sample_hidden_states =", sample_hidden)

    if sampled_token_ids and sampled_token_ids[1] > 1e-4:
        print("  likely first bad point inside one DP replica: sampled token ids")
    elif prepared_input_ids and prepared_input_ids[1] > 1e-4:
        print("  likely first bad point inside one DP replica: next-step input_ids preparation")
    elif prepared_positions and prepared_positions[1] > 1e-4:
        print("  likely first bad point inside one DP replica: next-step positions preparation")
    elif decode_q and decode_q[1] > 1e-4:
        print("  likely first bad point inside one DP replica: decode q_nope / preprocess")
    elif decode_q_rope and decode_q_rope[1] > 1e-4:
        print("  likely first bad point inside one DP replica: decode q_pe / rope preprocess")
    elif decode_attn and decode_attn[1] > 1e-4:
        print("  likely first bad point inside one DP replica: decode attention output")
    elif output_decode and output_decode[1] > 1e-4:
        print("  likely first bad point inside one DP replica: decode branch output before o_proj_input assignment")
    elif decode_assign and decode_assign[1] > 1e-4:
        print("  likely first bad point inside one DP replica: assigning decode branch into o_proj_input")
    elif output_prefill and output_prefill[1] > 1e-4:
        print("  likely first bad point inside one DP replica: prefill branch output before o_proj_input assignment")
    elif prefill_assign and prefill_assign[1] > 1e-4:
        print("  likely first bad point inside one DP replica: assigning prefill branch into o_proj_input")
    elif attn_out and attn_out[1] > 1e-4:
        print("  likely first bad point inside one DP replica: attention output aggregation / o_proj_input")
    elif block_out and block_out[1] > 1e-4:
        print("  likely first bad point inside one DP replica: o_proj or post-attention block output")
    elif sample_hidden and sample_hidden[1] > 1e-4:
        print("  likely first bad point inside one DP replica: hidden state selection before compute_logits")
    else:
        print("  no req0-vs-req1 divergence detected in compared tensor packs")


def cmd_analyze(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir)
    paths = sorted(dump_dir.glob("identical_req_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No identical-request dumps found under {dump_dir}")

    tensor_paths = []
    skipped_files: list[tuple[str, str]] = []
    for path in paths:
        try:
            data = _load(path)
        except EOFError:
            skipped_files.append((path.name, "EOFError"))
            continue
        except Exception as exc:
            skipped_files.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue
        meta = data.get("meta", {})
        if meta.get("kind") != "tensor":
            continue
        if args.stage and meta.get("stage") != args.stage:
            continue
        if args.step is not None and meta.get("decode_step") != args.step:
            continue
        if args.name_contains and args.name_contains not in str(meta.get("name", "")):
            continue
        tensor_paths.append((path, data))

    if skipped_files:
        print("=" * 140)
        print("SKIPPED FILES")
        for file_name, reason in skipped_files:
            print(f"  {file_name}: {reason}")

    print("DP0 VS DP1")
    aggregated = _collect_tensor_results(tensor_paths)
    print("REQ0 VS REQ1")
    req_aggregated = _collect_same_file_results(tensor_paths)
    logit_paths = _load_logits(paths, args.stage, args.step)
    if logit_paths:
        print("LOGITS")
        _print_logits(logit_paths)
    _print_verdict(aggregated)
    _print_req_verdict(req_aggregated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze identical-request debug dumps by dp/tp grouping.")
    parser.add_argument("dump_dir", help="Directory containing identical-request .pt dumps.")
    parser.add_argument("--stage", choices=["prefill", "decode", "mixed"], help="Optional stage filter.")
    parser.add_argument("--step", type=int, help="Optional decode step filter.")
    parser.add_argument("--name-contains", help="Optional substring filter on meta.name.")
    parser.set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
