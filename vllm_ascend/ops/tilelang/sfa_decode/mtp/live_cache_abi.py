# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]

from ..shared.carrier import keep_parameters, mixed_signature
from . import INDEX_HEAD_PANEL_SHAPE, mtp_projection_rows


def abi_program(
    requests,
    tokens,
    pages,
    columns,
    main_pages,
    main_cols,
    ring_pages,
    ring_table_stride,
    index_width=2176,
    index_cols=None,
    native_output=False,
    native_head_weights=False,
    native_qkv_a=False,
    native_q_b=False,
):
    index_cols = columns if index_cols is None else index_cols
    uv_shape = (64, 512, 256) if native_head_weights else (64, 256, 512)
    uk_shape = (64, 256, 512) if native_head_weights else (64, 512, 256)
    output_weight_shape = (16384, 4096) if native_output else (4096, 16384)
    qa_shape = (4096, 2048) if native_qkv_a else (2048, 4096)
    qb_shape = (1536, 16384) if native_q_b else (16384, 1536)
    nr = mtp_projection_rows(tokens)
    output_rows = nr
    pools = columns * 32
    padded = (tokens + 5) * 32

    @T.prim_func
    def main(
        Q: T.Tensor((padded, 128), T.float8_e4m3fn),
        QS: T.Tensor((padded,), T.float32),
        K: T.Tensor((pages, 34, 128), T.float8_e4m3fn),
        KS: T.StridedTensor((pages, 32), (1088, 1), T.float32),
        W: T.Tensor((nr * 32,), T.bfloat16),
        Table: T.Tensor((requests, index_cols), T.int32),
        Length: T.Tensor((requests,), T.int32),
        Ends: T.Tensor((requests,), T.int64),
        Runs: T.Tensor((requests, columns // 8), T.int32),
        Score: T.Tensor((tokens, pools), T.bfloat16),
        Raw: T.Tensor((nr, 4096), T.bfloat16),
        Pos: T.Tensor((tokens,), T.int64),
        Packed: T.Tensor((tokens, index_width), T.int32),
        MQ: T.Tensor((nr, 64, 512), T.bfloat16),
        MK: T.Tensor((main_pages, 128, 132), T.int32),
        MT: T.Tensor((requests, main_cols), T.int32),
        MS: T.Tensor((requests + 1,), T.int32),
        MY: T.Tensor((output_rows, 64, 512), T.bfloat16),
        WV: T.Tensor(uv_shape, T.bfloat16),
        V: T.Tensor((output_rows, 16384), T.bfloat16),
        OC: T.Tensor((output_rows, 16384), T.float8_e4m3fn),
        XS: T.Tensor((256, output_rows), T.int16),
        WO: T.Tensor(output_weight_shape, T.float8_e4m3fn),
        OS: T.Tensor((256, 4096), T.int16),
        Final: T.Tensor((tokens, 4096), T.bfloat16),
        X: T.Tensor((tokens, 4096), T.bfloat16),
        AC: T.Tensor((nr, 4096), T.float8_e4m3fn),
        AX: T.Tensor((64, nr), T.int16),
        AW: T.Tensor(qa_shape, T.float8_e4m3fn),
        AWX: T.Tensor((64, 2048), T.int16),
        QKV: T.Tensor((nr, 2048), T.bfloat16),
        GQ: T.Tensor((1536,), T.bfloat16),
        GK: T.Tensor((512,), T.bfloat16),
        Slot: T.Tensor((tokens, 1), T.int32),
        Norm: T.Tensor((nr, 1536), T.bfloat16),
        BC: T.Tensor((nr, 1536), T.float8_e4m3fn),
        BX: T.Tensor((24, nr), T.int16),
        BW: T.Tensor(qb_shape, T.float8_e4m3fn),
        BWX: T.Tensor((24, 16384), T.int16),
        QB: T.Tensor((nr, 16384), T.bfloat16),
        UK: T.Tensor(uk_shape, T.bfloat16),
        IQuery: T.Tensor((1, 4096, 1536), T.bfloat16),
        IWeight: T.Tensor((32, 4096), T.bfloat16),
        Panel: T.Tensor(INDEX_HEAD_PANEL_SHAPE, T.float32),
        Parts: T.Tensor((nr // 32, 32, 32, 32), T.float32),
        GateWeight: T.Tensor((1, 256, 4096), T.bfloat16),
        KG: T.Tensor((nr, 256), T.bfloat16),
        Gamma: T.Tensor((128,), T.float32),
        Beta: T.Tensor((128,), T.float32),
        Projected: T.Tensor((tokens, 256), T.float32),
        Ape: T.Tensor((4, 128), T.float32),
        RingPage: T.StridedTensor((requests,), (ring_table_stride,), T.int32),
        Ring: T.StridedTensor((ring_pages, 9, 256), (289 * 256, 256, 1), T.float32),
        IndexSlots: T.Tensor((tokens,), T.int64),
        Writes: T.Tensor((requests, 6), T.int64),
    ):
        mixed_signature(Q[0:32, :128], K[0, :32, :], Final)

    return keep_parameters(main)
