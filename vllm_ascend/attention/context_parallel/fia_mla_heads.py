import torch

_SUPPORTED_NUM_HEADS = (1, 2, 4, 8, 16, 32, 64, 128)


def pad_fia_mla_query_heads(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    input_layout: str,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad MLA query heads to a shape accepted by FIA with D=512 and RoPE."""
    if num_heads in _SUPPORTED_NUM_HEADS:
        return q_nope, q_pe, num_heads

    head_axis = {"BNSD": 1, "BSND": 2}.get(input_layout)
    if head_axis is None:
        raise ValueError(f"Cannot pad FIA MLA query heads for input layout {input_layout}")
    padded_num_heads = next((value for value in _SUPPORTED_NUM_HEADS if value > num_heads), None)
    if padded_num_heads is None:
        raise ValueError(f"FIA MLA query head count {num_heads} exceeds the supported range")

    pad_shape = list(q_nope.shape)
    pad_shape[head_axis] = padded_num_heads - num_heads
    q_nope = torch.cat((q_nope, q_nope.new_zeros(pad_shape)), dim=head_axis)
    pad_shape[-1] = q_pe.shape[-1]
    q_pe = torch.cat((q_pe, q_pe.new_zeros(pad_shape)), dim=head_axis)
    return q_nope, q_pe, padded_num_heads


def trim_fia_mla_query_heads(
    attn_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    input_layout: str,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_head_axis = {"BNSD": 1, "BSND": 2}.get(input_layout)
    if output_head_axis is None:
        raise ValueError(f"Cannot trim FIA MLA query heads for input layout {input_layout}")
    return (
        attn_output.narrow(output_head_axis, 0, num_heads),
        softmax_lse.narrow(1, 0, num_heads),
    )
