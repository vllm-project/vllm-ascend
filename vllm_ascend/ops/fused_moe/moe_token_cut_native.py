# SPDX-License-Identifier: Apache-2.0
"""Contracts for the user's in-place native operators; no kernel substitutes."""

from collections.abc import Mapping

import torch

NATIVE_TOKEN_CUT_CAPACITY = 96

# Match the supplied ops.def, including names, order and mutation annotations.
MOE_TOKEN_CUT_SCHEMA = (
    "moe_token_cut(Tensor topk_ids, Tensor topk_weights, Tensor hidden, Tensor! cutidx, "
    "Tensor! new_topk, Tensor! new_topkweight, Tensor! new_hidden, Tensor! new_residuals, "
    "Tensor! actual_bs, Tensor! prefix_index, Tensor! no_cut_rank_num, Tensor! active_mask, "
    "Tensor local_cut_count, Tensor batch_size, Tensor enable_cut, Tensor! residuals, "
    "Tensor! discard_label, Tensor discard_tokens, Tensor discard_moe_req_id, Tensor! req_id_list, "
    "Tensor discard_residuals, Tensor discard_topk, Tensor discard_topk_weight, Tensor! moe_cnt, "
    "Tensor! history_bs, Tensor original_bs, str group_ep, int topk, int rank_id, int layer_idx, "
    "int rank_num, int expert_num, int expert_per_rank, int level_limit, int cut_count_threshold, "
    "int small_expert_token_threshold, int send_aiv_num, int recv_aiv_num) -> ()"
)
ATTN_METADATA_UPDATE_SCHEMA = (
    "attn_metadata_update(Tensor req_ids, Tensor discard_label, Tensor positions, Tensor slot_mapping, "
    "Tensor block_table, Tensor seq_lens, Tensor discard_positions, Tensor discard_slot_mapping, "
    "Tensor discard_context_lens, Tensor discard_block_tables, Tensor discard_attn_cnt, "
    "Tensor discard_req_ids, Tensor original_bs, int max_bs, int layer_idx, Tensor! out_positions, "
    "Tensor! out_slots, Tensor! out_seq_lens, Tensor! out_block_table) -> ()"
)
COMBINE_AND_DISPATCH_REBATCH_SCHEMA = (
    "combine_and_dispatch_rebatch(Tensor! batch_before_dispatch, Tensor! residuals_before_dispatch, "
    "Tensor cut_idx, Tensor! discard_label, Tensor! discard_tokens, Tensor! discard_moe_req_id_tensor, "
    "Tensor! req_id_list, Tensor! discard_residuals, Tensor! running_topk, Tensor! discard_topk, "
    "Tensor! running_topk_weight, Tensor! discard_topk_weight, Tensor! moe_cnt_input, Tensor actual_bs, "
    "Tensor! history_bs, Tensor! running_bs, Tensor original_bs, int is_combine_rebatch, int max_bs, "
    "int k) -> ()"
)
REBATCH_AND_UPDATE_SCHEMA = (
    "rebatch_and_update(Tensor! batch_before_dispatch, Tensor! residuals_before_dispatch, "
    "Tensor cut_idx, Tensor prefix_index, Tensor! discard_label, Tensor! discard_tokens, "
    "Tensor! discard_moe_req_id_tensor, Tensor! req_id_list, Tensor! discard_residuals, "
    "Tensor! running_topk, Tensor! discard_topk, Tensor! running_topk_weight, Tensor! discard_topk_weight, "
    "Tensor! moe_cnt_input, Tensor actual_bs, Tensor! history_bs, Tensor! running_bs, Tensor original_bs, "
    "Tensor positions, Tensor slot_mapping, Tensor block_table, Tensor seq_lens, Tensor discard_positions, "
    "Tensor discard_slot_mapping, Tensor discard_context_lens, Tensor discard_block_tables, "
    "Tensor! discard_attn_cnt, Tensor! discard_req_ids, int is_combine_rebatch, int max_bs, int k, "
    "int layer_idx, Tensor! out_rebatch_residuals, Tensor! out_positions, Tensor! out_slots, "
    "Tensor! out_seq_lens, Tensor! out_block_table) -> ()"
)


def validate_native_schema(actual, expected=MOE_TOKEN_CUT_SCHEMA):
    """Reject mismatched ABI before launching a kernel, not just by a substring."""
    reference = torch._C.parse_schema(expected)

    def arguments(schema):
        return tuple(
            (arg.name, str(arg.type), bool(arg.alias_info and arg.alias_info.is_write), arg.kwarg_only)
            for arg in schema.arguments
        )

    if actual.name.rsplit("::", 1)[-1] != reference.name or arguments(actual) != arguments(reference) or actual.returns:
        raise RuntimeError(f"native token-cut binding mismatch; expected {reference}; loaded {actual}")


def invoke_native_op(schema: str, values: Mapping, *, op=None) -> None:
    """Pass caller-owned buffers in binding order. Does not allocate state."""
    contract = torch._C.parse_schema(schema)
    expected_names = {arg.name for arg in contract.arguments}
    if set(values) != expected_names:
        raise ValueError(
            f"{contract.name} arguments: missing={sorted(expected_names - set(values))}, "
            f"unexpected={sorted(set(values) - expected_names)}"
        )
    for arg in contract.arguments:
        value = values[arg.name]
        kind = str(arg.type)
        valid = (
            isinstance(value, torch.Tensor)
            if kind == "Tensor"
            else type(value) is int
            if kind == "int"
            else isinstance(value, str)
        )
        if not valid:
            raise TypeError(f"{contract.name}.{arg.name} must be {kind}")
    if op is None:
        op = getattr(torch.ops._C_ascend, contract.name).default
    if hasattr(op, "_schema"):
        validate_native_schema(op._schema, schema)
    result = op(*(values[arg.name] for arg in contract.arguments))
    if result is not None:
        raise RuntimeError(f"{contract.name} must return None; outputs are in-place tensors")
