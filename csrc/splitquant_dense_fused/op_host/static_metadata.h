// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <map>
#include <tuple>
#include <cstdlib>
namespace splitquant {
inline bool use_zero_epilogue(const at::Tensor& offsets,int64_t m){
 if(m!=1 || !std::getenv("SPLITQUANT_ZERO_EPI"))return false;
 using Key=std::tuple<const void*,uint32_t,int,c10::StreamId>;
 struct Entry{at::Tensor source;bool zero;};static thread_local std::map<Key,Entry> cache;
 Key key{offsets.unsafeGetTensorImpl(),offsets.is_inference()?0:offsets.unsafeGetTensorImpl()->version_counter().current_version(),offsets.device().index(),c10_npu::getCurrentNPUStream(offsets.device().index()).id()};
 auto it=cache.find(key);if(it==cache.end())it=cache.emplace(key,Entry{offsets,at::count_nonzero(offsets).item<int64_t>()==0}).first;
 return it->second.zero;
}
}
