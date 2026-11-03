#include "torch_kernel_helper.h"
#include "../ops.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_moe_compact_silu_debug.h"
#include <map>
#include <tuple>
namespace splitquant {
at::Tensor compact_lut(const at::Tensor& lut){
 using Key=std::tuple<const void*,uint32_t,int,c10::StreamId>;struct Entry{at::Tensor source,value;};static thread_local std::map<Key,Entry> cache;
 Key key{lut.unsafeGetTensorImpl(),lut.is_inference()?0:lut.unsafeGetTensorImpl()->version_counter().current_version(),lut.device().index(),c10_npu::getCurrentNPUStream(lut.device().index()).id()};auto it=cache.find(key);
 if(it==cache.end()){auto small=at::cat({lut.slice(0,14976,17152),lut.slice(0,47744,49920),lut.slice(0,0,256),lut.slice(0,32768,33024),lut.slice(0,32640,32768),lut.slice(0,65408,65536)},0).contiguous();it=cache.emplace(key,Entry{lut,small}).first;}return it->second.value;
}
at::Tensor compact_silu_debug(const at::Tensor& x,const at::Tensor& lut){
 TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.numel()>0&&x.numel()<=1048576,"compact SiLU debug requires contiguous BF16 NPU x, 1..1048576 elements");TORCH_CHECK(lut.device()==x.device()&&lut.scalar_type()==at::kBFloat16&&lut.is_contiguous()&&lut.dim()==1&&lut.numel()==65536,"compact SiLU debug LUT shape");auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"compact SiLU platform unavailable");auto table=compact_lut(lut);auto out=at::empty_like(x);uint32_t n=x.numel(),blocks=std::min<uint32_t>(platform->GetCoreNumAiv(),(n+127)/128);EXEC_KERNEL_CMD(moe_compact_silu_debug,blocks,x,table,out,n);return out;
}
}
