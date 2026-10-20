// SPDX-License-Identifier: Apache-2.0
#include "mla_preprocess_serial.h"
#include "aclnn_torch_adapter/op_api_common.h"
namespace vllm_ascend {
static void CheckTensor(const at::Tensor& x)
{
    TORCH_CHECK(x.scalar_type()==at::kBFloat16,"MLA Vector operators require BF16");
    TORCH_CHECK(x.device().type()==at::kPrivateUse1,"MLA Vector operators require NPU");
    TORCH_CHECK(x.is_contiguous(),"MLA Vector source must be contiguous");
}
static void CheckRope(const std::optional<at::Tensor>& cos,const std::optional<at::Tensor>& sin,
                      const at::Tensor& x)
{
    TORCH_CHECK(cos.has_value()==sin.has_value(),"cos/sin must be paired");
    if(cos) {
        CheckTensor(*cos); CheckTensor(*sin);
        TORCH_CHECK(cos->sizes()==sin->sizes() && cos->dim()==2 && cos->size(0)==x.size(0) &&
                    cos->size(1)==64,"RoPE coefficients must have shape [T,64]");
        TORCH_CHECK(cos->device()==x.device() && sin->device()==x.device(),"device mismatch");
    }
}
at::Tensor mla_serial_norm_cache(const at::Tensor& down,const at::Tensor& gamma_q,const at::Tensor& gamma,
    const std::optional<at::Tensor>& cos,const std::optional<at::Tensor>& sin,
    const at::Tensor& slots,at::Tensor& cache,at::Tensor& cache_rope)
{
    CheckTensor(down); CheckTensor(gamma); CheckTensor(gamma_q); CheckRope(cos,sin,down);
    TORCH_CHECK(gamma_q.dim()==1 && down.dim()==2 && down.size(1)==gamma_q.numel()+576 &&
                gamma_q.device()==down.device(),"invalid gamma_q/down");
    TORCH_CHECK(down.dim()==2 && gamma.dim()==1 && gamma.numel()==512,"invalid down/gamma");
    TORCH_CHECK(slots.dim()==1 && slots.numel()==down.size(0) && slots.scalar_type()==at::kInt &&
                slots.is_contiguous(),"slots must be int32 [T]");
    TORCH_CHECK(cache.dim()==4 && cache_rope.dim()==4 && cache.size(0)==cache_rope.size(0) &&
                cache.size(1)==128 && cache.size(2)==1 && cache.size(3)==512 &&
                cache_rope.size(1)==128 && cache_rope.size(2)==1 && cache_rope.size(3)==64,
                "cache must be split ND [blocks,128,1,512/64]");
    TORCH_CHECK(cache.stride(1)==512 && cache.stride(2)==512 && cache.stride(3)==1 &&
                cache_rope.stride(1)==64 && cache_rope.stride(2)==64 && cache_rope.stride(3)==1,
                "only cache block stride may be noncontiguous");
    TORCH_CHECK(cache.scalar_type()==at::kBFloat16 && cache_rope.scalar_type()==at::kBFloat16 &&
                down.device()==gamma.device() && down.device()==slots.device() &&
                down.device()==cache.device() && down.device()==cache_rope.device(),"dtype/device mismatch");
    at::Tensor c=cos.value_or(at::Tensor()),s=sin.value_or(at::Tensor());
    int64_t cache_stride=cache.stride(0),rope_stride=cache_rope.stride(0),blocks=cache.size(0);
    int64_t use_rope=cos.has_value();
    auto q=at::empty({down.size(0),gamma_q.numel()},down.options());
    EXEC_NPU_CMD(aclnnMlaSerialNormCache,down,gamma,c,s,slots,gamma_q,cache_stride,rope_stride,blocks,use_rope,cache,cache_rope,q);
    return q;
}
std::tuple<at::Tensor,at::Tensor> mla_serial_q_prepare(const at::Tensor& up,int64_t heads,
    const std::optional<at::Tensor>& cos,const std::optional<at::Tensor>& sin)
{
    CheckTensor(up); CheckRope(cos,sin,up);
    TORCH_CHECK(up.dim()==2 && heads>0 && heads<=128 && up.size(1)%heads==0,
                "up must be [T,H*(P+64)] with 1<=H<=128");
    int64_t nope=up.size(1)/heads-64;
    TORCH_CHECK(nope>=16 && nope<=256 && nope%16==0,"P must be aligned and within [16,256]");
    auto packed=at::empty({up.size(0),heads,nope},up.options());
    auto qr=at::empty({up.size(0),heads,64},up.options());
    at::Tensor c=cos.value_or(at::Tensor()),s=sin.value_or(at::Tensor());
    int64_t use_rope=cos.has_value();
    EXEC_NPU_CMD(aclnnMlaSerialQPrepare,up,c,s,heads,use_rope,packed,qr);
    return {packed,qr};
}
at::Tensor mla_serial_q_pack(const at::Tensor& up,int64_t heads)
{
    CheckTensor(up);
    TORCH_CHECK(up.dim()==2 && heads>0 && heads<=128 && up.size(1)%heads==0,"invalid up/heads");
    int64_t nope=up.size(1)/heads-64;
    TORCH_CHECK(nope>=16 && nope<=256 && nope%16==0,"invalid P");
    auto packed=at::empty({heads,up.size(0),nope},up.options());
    EXEC_NPU_CMD(aclnnMlaSerialQPack,up,heads,packed);
    return packed;
}
std::tuple<at::Tensor,at::Tensor> mla_serial_q_finalize(const at::Tensor& up,const at::Tensor& key,
    const std::optional<at::Tensor>& cos,const std::optional<at::Tensor>& sin)
{
    CheckTensor(up); CheckTensor(key); CheckRope(cos,sin,up);
    TORCH_CHECK(up.dim()==2 && key.dim()==3 && key.size(1)==up.size(0) && key.size(2)==512,
                "up/key must be [T,H*(P+64)]/[H,T,512]");
    TORCH_CHECK(up.device()==key.device(),"device mismatch");
    int64_t heads=key.size(0),use_rope=cos.has_value();
    TORCH_CHECK(heads>0 && heads<=128 && up.size(1)%heads==0,"invalid heads");
    int64_t nope=up.size(1)/heads-64;
    TORCH_CHECK(nope>=16 && nope<=256 && nope%16==0,"invalid P");
    auto qn=at::empty({up.size(0),heads,512},up.options());
    auto qr=at::empty({up.size(0),heads,64},up.options());
    at::Tensor c=cos.value_or(at::Tensor()),s=sin.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnMlaSerialQFinalize,up,key,c,s,heads,use_rope,qn,qr);
    return {qn,qr};
}
namespace meta {
at::Tensor mla_serial_norm_cache(const at::Tensor& down,const at::Tensor& gamma_q,const at::Tensor&,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const at::Tensor&,at::Tensor&,at::Tensor&)
{ return at::empty({down.size(0),gamma_q.numel()},down.options()); }

at::Tensor mla_serial_q_pack(const at::Tensor& up,int64_t heads)
{ return at::empty({heads,up.size(0),up.size(1)/heads-64},up.options()); }
std::tuple<at::Tensor,at::Tensor> mla_serial_q_finalize(const at::Tensor& up,const at::Tensor& key,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&)
{ return {at::empty({up.size(0),key.size(0),512},up.options()),
          at::empty({up.size(0),key.size(0),64},up.options())}; }

std::tuple<at::Tensor,at::Tensor> mla_serial_q_prepare(const at::Tensor& up,int64_t heads,
    const std::optional<at::Tensor>&,const std::optional<at::Tensor>&)
{ return {at::empty({up.size(0),heads,up.size(1)/heads-64},up.options()),
          at::empty({up.size(0),heads,64},up.options())}; }
}
}
