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

#ifndef MLA_PREPROCESS_VALIDATION_H
#define MLA_PREPROCESS_VALIDATION_H

#include <cmath>
#include <c10/util/Exception.h>

namespace vllm_ascend {
namespace mla_preprocess_detail {

inline void ValidateRmsNormEpsilon(double epsilon, const char *name)
{
    const float kernelEpsilon = static_cast<float>(epsilon);
    TORCH_CHECK(std::isfinite(kernelEpsilon) && kernelEpsilon > 0.0f,
                name, " must be positive and finite in float32.");
}

}  // namespace mla_preprocess_detail
}  // namespace vllm_ascend

#endif  // MLA_PREPROCESS_VALIDATION_H
