/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

class GenericBlockSparseAttention : public OpDef {
public:
    explicit GenericBlockSparseAttention(const char* name) : OpDef(name)
    {
        // dtype column mapping:
        //   col0: FP16 input  -> FP16 output
        //   col1: BF16 input  -> BF16 output
        //   col2: FP8  input  -> FP16 output (attention_dtype=half)
        //   col3: FP8  input  -> BF16 output (attention_dtype=bfloat16)
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("sparseBlockIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("sparseBlockCount")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("metaData")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("attenMaskOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND});
        this->Input("qDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("kDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("vDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("pQuantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("cuSeqLengthsQOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cuSeqLengthsKvOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sequsedQOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sequsedKvOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("blockTableOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmaxLse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        // Regular path requires blockShapeX=1; keep default aligned with supported config.
        this->Attr("blockShape").AttrType(OPTIONAL).ListInt({1, 128});
        // Regular path / kernel only support packed GQA (isPackedGQA=1); host rejects other values.
        this->Attr("isPackedGQA").AttrType(OPTIONAL).Int(1);
        // Design doc: layoutQ/layoutKv are String (unlike SAS). Values e.g. "TND", "PAGED_BBND".
        this->Attr("layoutQ").AttrType(OPTIONAL).String("TND");
        this->Attr("layoutKv").AttrType(OPTIONAL).String("TND");
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("maskType").AttrType(OPTIONAL).Int(0);
        this->Attr("quantType").AttrType(OPTIONAL).Int(0);      // 0=none, 1=per-block FP8
        this->Attr("dstTypeMax").AttrType(OPTIONAL).Float(0.0);
        this->Attr("softmaxPrecision").AttrType(OPTIONAL).Int(0);
        this->Attr("winLeft").AttrType(OPTIONAL).Int(-1);
        this->Attr("winRight").AttrType(OPTIONAL).Int(-1);
        this->Attr("returnSoftmaxlse").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(GenericBlockSparseAttention);

}  // namespace ops
