from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FusedCompressorRowInfo:
    expected_rows: int
    valid_rows: int
    safe: bool
    reason: str = ""


def get_fused_compressor_row_info(
    x_shape: Sequence[int],
    rope_shape: Sequence[int],
    slot_mapping_shape: Sequence[int],
    cmp_ratio: int,
) -> FusedCompressorRowInfo:
    """Return the padded compressor rows and the rows safe for scatter."""
    if cmp_ratio <= 0:
        return FusedCompressorRowInfo(0, 0, False, "cmp_ratio must be positive")
    if len(x_shape) == 2:
        if not rope_shape:
            return FusedCompressorRowInfo(0, 0, False, "rope shape is empty")
        expected_rows = int(rope_shape[0])
    elif len(x_shape) == 3:
        if len(x_shape) < 2:
            return FusedCompressorRowInfo(0, 0, False, "x shape is incomplete")
        expected_rows = int(x_shape[0]) * ((int(x_shape[1]) + cmp_ratio - 1) // cmp_ratio)
    else:
        return FusedCompressorRowInfo(0, 0, False, "x must be rank 2 or 3")

    if len(slot_mapping_shape) != 2 or int(slot_mapping_shape[1]) != 2:
        return FusedCompressorRowInfo(expected_rows, 0, False, "slot_mapping must have shape [rows, 2]")

    valid_rows = int(slot_mapping_shape[0])
    if valid_rows > expected_rows:
        return FusedCompressorRowInfo(
            expected_rows,
            valid_rows,
            False,
            "slot_mapping rows exceed padded compressor output rows",
        )
    if valid_rows == 0:
        return FusedCompressorRowInfo(expected_rows, valid_rows, False, "no valid rows to scatter")
    return FusedCompressorRowInfo(expected_rows, valid_rows, True)
