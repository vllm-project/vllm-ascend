#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_splitquant_decode_bf16.h"
#include "aclrtlaunch_splitquant_decode_fp16.h"
#include <map>
#include <tuple>
namespace splitquant {
at::Tensor decode(const at::Tensor& a,const at::Tensor& w,const at::Tensor& ws,const at::Tensor& wo,const at::Tensor& s,const at::Tensor& sum,const c10::optional<at::Tensor>& bias,const at::Tensor& x){
 const int64_t n=w.size(0),g=a.size(0);
 using Key=std::tuple<const void*,uint32_t,int,c10::StreamId>;
 struct Packed{at::Tensor source,value;};
 static thread_local std::map<Key,Packed> cache;
 uint32_t version=w.is_inference()?0:w.unsafeGetTensorImpl()->version_counter().current_version();
 Key key{w.unsafeGetTensorImpl(),version,w.device().index(),c10_npu::getCurrentNPUStream(w.device().index()).id()};
 auto it=cache.find(key);
 if(it==cache.end())it=cache.emplace(key,Packed{w,w.reshape({n/16,16,g,64}).permute({0,2,1,3}).contiguous()}).first;
 auto packed=it->second.value;auto y=at::empty({1,n},x.options());auto b=bias?*bias:at::empty({1},x.options());
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"cannot query platform");
 uint32_t nu=n,gu=g,hb=bias?1:0,blocks=std::min<int64_t>(platform->GetCoreNumAiv(),n/16);
 if(x.scalar_type()==at::kBFloat16){EXEC_KERNEL_CMD(splitquant_decode_bf16,blocks,a,packed,ws,wo,s,sum,b,y,nu,gu,hb);}
 else {EXEC_KERNEL_CMD(splitquant_decode_fp16,blocks,a,packed,ws,wo,s,sum,b,y,nu,gu,hb);}
 return y;
}
}
