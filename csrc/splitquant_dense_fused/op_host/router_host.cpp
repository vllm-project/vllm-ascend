// SPDX-License-Identifier: BSD-3-Clause
#include "torch_kernel_helper.h"
#include "../ops.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_moe_router_finish_bf16.h"
namespace splitquant {
std::tuple<at::Tensor,at::Tensor> router_finish(const at::Tensor& weights,const at::Tensor& ids,bool normalize){
 TORCH_CHECK(weights.device().type()==c10::DeviceType::PrivateUse1 && weights.scalar_type()==at::kFloat && weights.dim()==2 && weights.size(0)>=1 && weights.size(0)<=32 && weights.size(1)==8 && weights.is_contiguous()," router requires FP32 contiguous NPU weights[M,8], M1..32");
 TORCH_CHECK(ids.device()==weights.device() && ids.sizes()==weights.sizes() && ids.is_contiguous() && (ids.scalar_type()==at::kInt||ids.scalar_type()==at::kLong)," router requires same-shape INT32/INT64 indices");
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform," router platform unavailable");
 auto out=at::empty(weights.sizes(),weights.options().dtype(at::kBFloat16));auto outids=at::empty(ids.sizes(),ids.options().dtype(at::kLong));uint32_t m=weights.size(0),i64=ids.scalar_type()==at::kLong,norm=normalize;
 uint32_t blocks=std::min<uint32_t>(platform->GetCoreNumAiv(),(m+1)/2);EXEC_KERNEL_CMD(moe_router_finish_bf16,blocks,weights,ids,out,outids,m,i64,norm);return {out,outids};
}
}
