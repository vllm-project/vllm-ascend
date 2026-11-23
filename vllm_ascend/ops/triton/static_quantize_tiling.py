# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core partition and UB tile selection for static QINT8 quantization."""

# Instruction alignment in bytes on Ascend ND DMA.
DMA_ALIGNMENT_BYTES = 32


def _cann_ub_bytes(rows: int, columns: int, element_bytes: int, parameter_bytes: int, per_tensor: bool) -> int:
    """CANN v9.1's explicit buffers, NOT a Triton allocation measurement.

    FP16/BF16 retain a batch of raw input/output and one FP32 work row.
    FP32 modifies its input batch in place. Channel parameters remain live
    across all row batches; only low-precision parameters need cast buffers.
    The Triton lowering must independently demonstrate this buffer reuse.
    """

    def aligned(size):
        return (size + DMA_ALIGNMENT_BYTES - 1) // DMA_ALIGNMENT_BYTES * DMA_ALIGNMENT_BYTES

    data = aligned(rows * columns * element_bytes) + aligned(rows * columns)
    if element_bytes < 4:
        data += aligned(columns * 4)
    if per_tensor:
        # FP16/BF16 use a DMA block and GetValue. FP32 reads a GM scalar.
        parameters = 2 * aligned(parameter_bytes) if parameter_bytes < 4 else 0
    else:
        parameters = 2 * aligned(columns * parameter_bytes)
        if parameter_bytes < 4:
            parameters += 2 * aligned(columns * 4)
    return data + parameters


def _cann_ub_factors(
    row_factor: int,
    column_factor: int,
    element_bytes: int,
    per_tensor: bool,
    double_cut: bool,
    ub_bytes: int,
    cache_line_bytes: int,
) -> tuple[int, int]:
    """Translate CalcMaxBaseLen/CalcMaxN/CalcUBFactor, including its reserve.

    These factors describe the original implementation. They are separate
    from Triton's power-of-two carriers and compiler-owned temporary storage.
    """
    reserve_bytes = 2048  # AscendQuantV2::reserveUb_, not a tuned tile size.
    available = ub_bytes - reserve_bytes
    cast_bytes = 4 if element_bytes < 4 else 0
    input_count = 1 if per_tensor else 3  # x, reciprocal, offset.
    bytes_per_column = input_count * (element_bytes + cast_bytes) + 1
    line_elements = cache_line_bytes // element_bytes
    max_columns = available // bytes_per_column // line_elements * line_elements
    if max_columns < line_elements:
        raise RuntimeError("UB cannot hold one CANN-aligned static quantization tile")
    block_columns = (column_factor + line_elements - 1) // line_elements * line_elements
    if not double_cut and block_columns <= max_columns // 2:
        parameter_count = 0 if per_tensor else 2
        remaining = available - block_columns * (parameter_count * (element_bytes + cast_bytes) + cast_bytes)
        max_rows = max(1, remaining // (element_bytes + 1) // block_columns)
        return min(row_factor, max_rows), block_columns
    return 1, min(block_columns, max_columns)


def _quantize_ub_bytes(
    rows: int,
    columns: int,
    element_bytes: int,
    parameter_bytes: int,
    contiguous: bool,
    parameter_contiguous: bool,
    per_tensor: bool,
) -> int:
    """Conservative carrier working set; verify against each compiled kernel.

    CANN's buffer formula is only the starting point. Triton additionally
    needs row arithmetic/select temporaries, addressing and masks. A multirow
    result uses a FP16 carrier before the final INT8 cast: A3's INT8
    insert_slice otherwise recasts the whole batch twice for every row.
    Strided DMA stages one 32B block per lane in the observed A3 lowering.
    This source bound is not a compiler-reported allocation or proof of reuse.
    """

    def aligned(size):
        return (size + DMA_ALIGNMENT_BYTES - 1) // DMA_ALIGNMENT_BYTES * DMA_ALIGNMENT_BYTES

    data = _cann_ub_bytes(rows, columns, element_bytes, parameter_bytes, per_tensor)
    if rows == 1 and contiguous and (parameter_contiguous or per_tensor):
        # The affine row path lowers addresses/tail lengths to scalars. Its
        # previous physical IR reused the FP32 work row through arithmetic
        # and casts. Keep its extra row/predicate allowance as a conservative
        # upper bound after removing the explicit NaN select; do not shrink
        # the budget without checking the new compiler's buffer allocation.
        return data + aligned(columns * 4) + aligned((columns + 7) // 8)
    # Keep up to three FP32 row buffers; only low-precision CANN input has
    # already contributed one. The actual reuse is checked in physical IR.
    data += (2 if element_bytes < 4 else 3) * aligned(columns * 4)
    data += 2 * aligned(columns * 8) + 3 * aligned(columns) + aligned(rows * 8)
    if rows > 1:
        data += aligned(rows * columns * 2)
    if not contiguous:
        data += 3 * aligned(columns * 8) + aligned(rows * columns * DMA_ALIGNMENT_BYTES)
    if not per_tensor and not parameter_contiguous:
        data += 2 * aligned(columns * (8 + DMA_ALIGNMENT_BYTES))
    return data


def _quantize_tiling(
    rows: int,
    width: int,
    element_bytes: int,
    parameter_bytes: int,
    contiguous: bool,
    parameter_contiguous: bool,
    per_tensor: bool,
    vector_cores: int,
    ub_bytes: int,
    cache_line_bytes: int,
) -> dict[str, int]:
    """Resource-sized row tasks, then CANN partitioning and UB factors.

    Source: CANN v9.1 AscendQuantV2 CalcTiling/GetCoreNum/CalcBlockFactor.
    Scalar parameters use logical [1, numel]; channel parameters use
    [prod(shape[:-1]), shape[-1]]. For contiguous channel input with fewer
    rows than cores, avoid splitting each UB-sized row piece into many small
    programs. This is a Triton scheduling choice, not CANN's hardware-pool
    policy. At least one row per hardware core, strided layouts and scalar
    parameters retain the original pool and its established batch behavior.
    UB factors follow CalcUBFactor/CalcUBFactorDoubleCut. Power-of-two
    carriers are accounted separately, without changing core ownership.
    Original buffer sizes do not prove the compiler reuses Triton buffers.
    """
    core_pool = vector_cores
    if contiguous and parameter_contiguous and not per_tensor and rows < vector_cores:
        _, row_columns = _cann_ub_factors(1, width, element_bytes, False, False, ub_bytes, cache_line_bytes)
        row_columns = 1 << (row_columns.bit_length() - 1)
        while _quantize_ub_bytes(1, row_columns, element_bytes, parameter_bytes, True, True, False) > ub_bytes - 2048:
            if row_columns == DMA_ALIGNMENT_BYTES:
                raise RuntimeError("UB cannot hold one aligned static quantization tile")
            row_columns //= 2
        row_pieces = (width + row_columns - 1) // row_columns
        core_pool = min(vector_cores, rows * row_pieces)

    def active(parts, pool):
        work = (parts + pool - 1) // pool
        return (parts + work - 1) // work

    line_elements = cache_line_bytes // element_bytes
    column_units = (width + line_elements - 1) // line_elements
    # CANN GetNewShape0 binds short output rows on non-V220 architectures.
    row_bind = (
        (DMA_ALIGNMENT_BYTES + width - 1) // width if cache_line_bytes != 512 and width < DMA_ALIGNMENT_BYTES else 1
    )
    partition_rows = (rows + row_bind - 1) // row_bind
    row_cores, column_cores = active(partition_rows, core_pool), active(column_units, core_pool)
    if row_cores >= column_cores:
        cores, row_factor = row_cores, (partition_rows + row_cores - 1) // row_cores * row_bind
        column_factor, column_cores = width, 1
    else:
        cores, row_factor = column_cores, rows
        column_factor = (column_units + column_cores - 1) // column_cores * line_elements
    # The width guard is CANN's g_BlockSize (32), not a preferred tile size.
    double_cut = False
    if cores < core_pool and rows > 1 and width > DMA_ALIGNMENT_BYTES and core_pool // rows:
        double_columns = active(column_units, core_pool // rows)
        if rows * double_columns > cores:
            cores, row_factor, column_cores = rows * double_columns, 1, double_columns
            column_factor = (column_units + column_cores - 1) // column_cores * line_elements
            double_cut = True

    base_rows, base_columns = _cann_ub_factors(
        row_factor, column_factor, element_bytes, per_tensor, double_cut, ub_bytes, cache_line_bytes
    )
    # A smaller power-of-two column carrier never exceeds the original
    # column buffer. Rows may have a masked tail, so account their carrier.
    block_columns = 1 << (base_columns.bit_length() - 1)
    block_rows = 1 << (base_rows - 1).bit_length()
    while (
        _quantize_ub_bytes(
            block_rows, block_columns, element_bytes, parameter_bytes, contiguous, parameter_contiguous, per_tensor
        )
        > ub_bytes - 2048
    ):
        if block_rows > 1:
            block_rows //= 2
            base_rows = min(base_rows, block_rows)
        elif block_columns > DMA_ALIGNMENT_BYTES:
            block_columns //= 2
        else:
            raise RuntimeError("UB cannot hold one aligned static quantization tile")
    return {
        "NUM_CORES": cores,
        "ROW_FACTOR": row_factor,
        "COLUMN_FACTOR": column_factor,
        "COLUMN_CORES": column_cores,
        "BASE_N": base_rows,
        "BLOCK_ROWS": block_rows,
        "BLOCK_COLS": block_columns,
    }
