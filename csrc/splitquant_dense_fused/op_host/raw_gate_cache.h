#pragma once
#include "torch_kernel_helper.h"
#include "weight_scale_cache.h"
#include <map>
#include <tuple>
namespace splitquant {
// Immutable cache is shared across streams after synchronous preparation.
// This prevents duplicating the production 9 GiB weight layout per graph stream.
inline at::Tensor raw_gate_weight(const at::Tensor& source,int64_t h,int64_t n){
 using Key=std::tuple<const void*,uint32_t,int>;struct Entry{at::Tensor source,value;};
 static thread_local std::map<Key,Entry> cache;
 Key key{source.unsafeGetTensorImpl(),source.is_inference()?0:source.unsafeGetTensorImpl()->version_counter().current_version(),source.device().index()};
 auto it=cache.find(key);if(it!=cache.end())return it->second.value;
 const int64_t e=source.size(0);
 auto cpu=source.to(at::kCPU).to(at::kByte);
 auto pairs=cpu.permute({0,2,3,1,4}).contiguous().reshape({e,h,n/2});
 auto codes=at::stack({at::bitwise_and(pairs,15),at::bitwise_right_shift(pairs,4)},-1).reshape({e,h,n}).transpose(1,2).contiguous();
 auto rows=at::bitwise_or(codes.slice(2,0,h,2),at::bitwise_left_shift(codes.slice(2,1,h,2),4));
 auto prepared=rows.reshape({e,n/64,4,16,h/64,32}).permute({0,1,4,2,3,5}).contiguous();
 auto value=nd_storage(prepared.to(source.device(),false,true));
 c10_npu::getCurrentNPUStream(source.device().index()).synchronize();
 return cache.emplace(key,Entry{source,value}).first->second.value;
}
}
