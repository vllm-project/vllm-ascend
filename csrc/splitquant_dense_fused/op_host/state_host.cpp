// BSD-3-Clause.
#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_splitquant_state_copy.h"
#include "aclrtlaunch_splitquant_state_index_copy.h"
namespace splitquant {
namespace {
uint64_t Check(const at::Tensor& cache,int64_t row){
 TORCH_CHECK(cache.device().type()==c10::DeviceType::PrivateUse1&&cache.dim()>=2," state needs NPU cache with contiguous inner rows");
 TORCH_CHECK(cache.scalar_type()==at::kFloat||cache.scalar_type()==at::kHalf||cache.scalar_type()==at::kBFloat16," state dtype");
 int64_t stride=1;for(int64_t d=cache.dim()-1;d>=1;--d){TORCH_CHECK(cache.size(d)==1||cache.stride(d)==stride," cache inner rows must be contiguous");stride*=cache.size(d);}
 TORCH_CHECK(cache.stride(0)>=stride," cache rows must not overlap");
 TORCH_CHECK(row>=0&&row<cache.size(0)," state row out of bounds");return cache.numel()/cache.size(0)*cache.element_size();
}
void Copy(const at::Tensor& src,at::Tensor& dst,uint64_t bytes,uint64_t so,uint64_t dest){
 if(!bytes)return;auto* p=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(p," platform missing");uint64_t ub=0;p->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);TORCH_CHECK(ub>=16384," state UB");
 uint32_t blocks=std::min<uint64_t>(p->GetCoreNumAiv(),(bytes+8191)/8192);
 EXEC_KERNEL_CMD(splitquant_state_copy,blocks,src,dst,bytes,so,dest);
}
}
at::Tensor state_read(const at::Tensor& cache,int64_t row){
 uint64_t bytes=Check(cache,row);auto shape=cache.sizes().vec();shape[0]=1;auto out=at::empty(shape,cache.options());Copy(cache,out,bytes,uint64_t(row)*cache.stride(0)*cache.element_size(),0);return out;
}
at::Tensor state_write(at::Tensor cache,const at::Tensor& source,int64_t row){
 uint64_t bytes=Check(cache,row);TORCH_CHECK(source.device()==cache.device()&&source.scalar_type()==cache.scalar_type()&&source.is_contiguous()&&uint64_t(source.numel()*source.element_size())==bytes," state source mismatch");
 Copy(source,cache,bytes,0,uint64_t(row)*cache.stride(0)*cache.element_size());return cache;
}
}

namespace splitquant {
at::Tensor indexed_copy(at::Tensor cache,const at::Tensor& source,const at::Tensor& indices,bool write){
 Check(cache,0);TORCH_CHECK(indices.device()==cache.device()&&indices.is_contiguous()&&indices.dim()==1&&(indices.scalar_type()==at::kLong||indices.scalar_type()==at::kInt)," indices must be same-device contiguous 1D INT32/INT64");
 int64_t count=indices.numel();uint64_t bytes=cache.numel()/cache.size(0)*cache.element_size(),stride=cache.stride(0)*cache.element_size(),rows=cache.size(0);auto shape=cache.sizes().vec();shape[0]=count;
 auto out=write?cache:at::empty(shape,cache.options());
 if(write)TORCH_CHECK(source.device()==cache.device()&&source.scalar_type()==cache.scalar_type()&&source.is_contiguous()&&source.numel()==count*cache.numel()/cache.size(0)," scatter source mismatch");
 if(!count||!bytes)return out;
 auto* p=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(p," platform missing");uint64_t ub=0;p->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);TORCH_CHECK(ub>=16384," state UB");
 auto ix=indices.to(at::kLong);uint32_t blocks=std::min<uint64_t>(p->GetCoreNumAiv(),count*((bytes+8191)/8192)),writing=write?1:0;auto input=write?source:cache;uint64_t batches=count;
 EXEC_KERNEL_CMD(splitquant_state_index_copy,blocks,input,out,ix,bytes,stride,rows,batches,writing);return out;
}
at::Tensor state_gather(const at::Tensor& cache,const at::Tensor& indices){return indexed_copy(cache,cache,indices,false);}
at::Tensor state_scatter(at::Tensor cache,const at::Tensor& source,const at::Tensor& indices){return indexed_copy(cache,source,indices,true);}
}
