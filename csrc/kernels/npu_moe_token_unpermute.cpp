// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/include/ops.h>
#include "inc/aclnn_common.h"

using namespace at_npu::native;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace {
    const static int DIMS = 2;
    const static int MIN_DIMS = 1;
    const static int64_t DEFAULT_TOPK = 1;

    void CheckMoeTokenUnpermuteForward(
        const at::Tensor& permuted_tokens,
        const at::Tensor& sorted_indices,
        c10::optional<at::Tensor>& probs,
        bool padded_mode = false
    )
    {
        if (padded_mode) {
            throw std::runtime_error("current version only support padded_mode is false");
        }
        // current version permuted_tokens only support bfloat16
        TORCH_CHECK(permuted_tokens.scalar_type() == at::ScalarType::BFloat16,
                    "Input tensor permuted_tokens dtype [", permuted_tokens.scalar_type(),
                    "] is invalid, should be bfloat16");
        // current version sorted_indices only support at::ScalarType::Int
        TORCH_CHECK(sorted_indices.scalar_type() == at::ScalarType::Int,
                    "Input tensor sorted_indices dtype [", sorted_indices.scalar_type(),
                    "] is invalid, should be int32");
        if (probs.has_value()) {
            TORCH_CHECK(probs.value().scalar_type() == at::ScalarType::BFloat16,
                        "Input tensor probs dtype [", probs.value().scalar_type(),
                        "] is invalid, should be bfloat16");
        }
        TORCH_CHECK(permuted_tokens.dim() == DIMS,
                    "The dims of input permuted_tokens should be 2 dimensional, but got ", permuted_tokens.dim(), "-dimensional.");
        TORCH_CHECK(sorted_indices.dim() == MIN_DIMS,
                    "The dims of input sorted_indices should be 1 dimensional, but got ", sorted_indices.dim(), "-dimensional.");
        if (probs.has_value()) {
            TORCH_CHECK(probs.value().dim() == DIMS,
                        "The dims of input probs should be 2 dimensional, but got ", probs.value().dim(), "-dimensional.");
        }
    }

    class NPUMoeTokenUnpermute {
    public:
        static at::Tensor forward(
            const at::Tensor& permuted_tokens,
            const at::Tensor& sorted_indices,
            c10::optional<at::Tensor>& probs,
            c10::optional<bool> padded_mode,
            c10::optional<at::IntArrayRef>& restore_shape
        )
        {
            at::AutoDispatchBelowADInplaceOrView guard;
            bool padded_mode_vale = padded_mode.value_or(false);
            auto restore_shape_vale = restore_shape.value_or(at::IntArrayRef{1});
            CheckMoeTokenUnpermuteForward(permuted_tokens, sorted_indices, probs, padded_mode_vale);
            int64_t topk = probs.has_value() ? probs.value().size(1) : DEFAULT_TOPK;
            // The sorted_indices actually implemented by the aclnn operator are different from the sorted_indices
            // output by the permute function of the megatron source code.
            // The actual sorted_indices implemented by the aclnn operator are not sliced.
            // so, num_unpermuted_tokens is obtained by dividing sorted_indices.size(0) by topk
            int64_t num_unpermuted_tokens = sorted_indices.size(0) / topk;
            at::Tensor unpermuted_tokens = at::empty({num_unpermuted_tokens, permuted_tokens.size(-1)}, permuted_tokens.options());
            at::Tensor probs_value = probs.has_value() ? probs.value() : at::Tensor();
            ACLNN_CMD(aclnnMoeTokenUnpermute, permuted_tokens, sorted_indices, probs_value, padded_mode_vale, restore_shape_vale, unpermuted_tokens);

            return unpermuted_tokens;
        }

    };
} // namespace

at::Tensor npu_moe_token_unpermute(
    const at::Tensor& permuted_tokens,
    const at::Tensor& sorted_indices,
    c10::optional<at::Tensor>& probs,
    c10::optional<bool> padded_mode,
    c10::optional<at::IntArrayRef>& restore_shape
)
{
    return NPUMoeTokenUnpermute::forward(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_token_unpermute", &npu_moe_token_unpermute,
          "npu moe token unpermute",
          pybind11::arg("permuted_tokens"),
          pybind11::arg("sorted_indices"),
          pybind11::arg("probs") = pybind11::none(),
          pybind11::arg("padded_mode") = false,
          pybind11::arg("restore_shape") = pybind11::none());
}
