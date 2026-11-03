#pragma once
#include "torch_kernel_helper.h"
#include <map>
#include <tuple>
namespace splitquant {
at::Tensor nd_storage(const at::Tensor&);
inline at::Tensor rounded_weight_scale(const at::Tensor& source,at::ScalarType dtype){
 using Key=std::tuple<const void*,uint32_t,int,int,c10::StreamId>;struct Entry{at::Tensor source,value;};static thread_local std::map<Key,Entry> cache;
 Key key{source.unsafeGetTensorImpl(),source.is_inference()?0:source.unsafeGetTensorImpl()->version_counter().current_version(),int(dtype),source.device().index(),c10_npu::getCurrentNPUStream(source.device().index()).id()};auto it=cache.find(key);
 if(it==cache.end())it=cache.emplace(key,Entry{source,nd_storage(source.to(dtype))}).first;return it->second.value;
}
}
