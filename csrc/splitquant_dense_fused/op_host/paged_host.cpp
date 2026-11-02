// SPDX-License-Identifier: BSD-3-Clause
#include "torch_kernel_helper.h"
#include "../ops.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_moe_paged_silu_debug.h"
#include "aclrtlaunch_moe_scalar_round_debug.h"
#include <array>
#include <cstring>
#include <map>
#include <tuple>
namespace splitquant {
at::Tensor paged_lut(const at::Tensor& lut){
 using Key=std::tuple<const void*,uint32_t,int,c10::StreamId>;struct Entry{at::Tensor source,value;};static thread_local std::map<Key,Entry> cache;
 Key key{lut.unsafeGetTensorImpl(),lut.is_inference()?0:lut.unsafeGetTensorImpl()->version_counter().current_version(),lut.device().index(),c10_npu::getCurrentNPUStream(lut.device().index()).id()};auto found=cache.find(key);if(found!=cache.end())return found->second.value;
 auto original=lut.to(at::kCPU);auto* bits=reinterpret_cast<const uint16_t*>(original.const_data_ptr());auto storage=at::zeros({32768},at::TensorOptions().dtype(at::kByte).device(at::kCPU));auto* data=storage.data_ptr<uint8_t>();auto* offsets=reinterpret_cast<uint32_t*>(data);auto* bases=reinterpret_cast<uint16_t*>(data+2048);std::map<std::array<uint16_t,128>,uint32_t> dict;
 for(uint32_t page=0;page<512;++page){
  bases[page]=bits[page*128];std::array<uint16_t,128> delta{};for(uint32_t j=0;j<128;++j)delta[j]=uint16_t(bits[page*128+j]-bases[page]);auto it=dict.find(delta);
  if(it==dict.end()){if(dict.size()>=116){cache.emplace(key,Entry{lut,{}});return {};}
   uint32_t offset=dict.size()*256;std::memcpy(data+3072+offset,delta.data(),256);it=dict.emplace(delta,offset).first;}
  offsets[page]=it->second;
 }
 auto table=storage.slice(0,0,3072+dict.size()*256).to(lut.device(),false,true);cache.emplace(key,Entry{lut,table});return table;
}
at::Tensor paged_silu_debug(const at::Tensor& x,const at::Tensor& lut){
 TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.numel()>0&&x.numel()<=1048576,"paged SiLU requires contiguous BF16 NPU x, 1..1048576 elements");TORCH_CHECK(lut.device()==x.device()&&lut.scalar_type()==at::kBFloat16&&lut.is_contiguous()&&lut.dim()==1&&lut.numel()==65536,"paged SiLU LUT shape");auto table=paged_lut(lut);TORCH_CHECK(table.defined(),"paged SiLU dictionary exceeds capacity; production falls back to original LUT");auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"paged SiLU platform unavailable");auto out=at::empty_like(x);uint32_t n=x.numel(),bytes=table.numel(),blocks=std::min<uint32_t>(platform->GetCoreNumAiv(),(n+127)/128);EXEC_KERNEL_CMD(moe_paged_silu_debug,blocks,x,table,out,n,bytes);return out;
}
}

namespace splitquant {
at::Tensor scalar_round_debug(const at::Tensor& x){
 TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1&&x.scalar_type()==at::kFloat&&x.is_contiguous()&&x.numel()>0&&x.numel()<=1048576,"scalar round debug requires contiguous NPU FP32 x, 1..1048576 elements");auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"scalar round platform unavailable");auto out=at::empty_like(x);uint32_t n=x.numel(),blocks=std::min<uint32_t>(platform->GetCoreNumAiv(),(n+127)/128);EXEC_KERNEL_CMD(moe_scalar_round_debug,blocks,x,out,n);return out;
}
}
