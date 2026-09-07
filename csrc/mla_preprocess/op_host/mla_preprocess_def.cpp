/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class MlaPreprocess : public OpDef {
public:
    explicit MlaPreprocess(const char *name) : OpDef(name)
    {
        this->Input("hiddenState")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantScale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantOffset1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wdqkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("bias1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantScale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quantOffset2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("keycache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("slotMapping")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wuq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("bias2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wuk")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale2")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ctkvScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qnopeScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("keycacheOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("q2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("keycacheOut2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("innerOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("cacheMode").AttrType(OPTIONAL).Int(1);
        this->Attr("quantMode").AttrType(OPTIONAL).Int(1);
        this->Attr("enableInnerOut").AttrType(OPTIONAL).Bool(false);
        this->Attr("enableRope").AttrType(OPTIONAL).Bool(true);
        this->Attr("kvCacheStride0").AttrType(REQUIRED).Int(0);
        this->Attr("kvCacheRopeStride0").AttrType(REQUIRED).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "mla_preprocess");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(MlaPreprocess);
}  // namespace ops
