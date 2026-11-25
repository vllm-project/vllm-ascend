"""TileLang projection stages with BF16 weights packed outside the inference path."""
import tilelang.ascend.language as T
from tvm import tirx
from tilelang.language.builtin import access_ptr

PRELUDE = r"""
struct compact_nz_weight { unsigned short data; };
struct global_nz_weight { unsigned short data; };
__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ compact_nz_weight* dst, __gm__ compact_nz_weight* src, uint32_t bytes, asc_load_l2_cache_mode mode, uint32_t, uint32_t, uint32_t, uint32_t) {
    asc_copy_gm2l1_align((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 1, bytes, 0, 0, false, mode, bytes, bytes);
}
__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ global_nz_weight* dst, __gm__ global_nz_weight* src, uint32_t src_stride, asc_load_l2_cache_mode mode, uint32_t groups, uint32_t bytes_per_group, uint32_t dst_stride, uint32_t) {
    asc_copy_gm2l1_align((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, groups, bytes_per_group, 0, 0, false, mode, src_stride, dst_stride);
}
"""

def load_compact(dst, src, n, k):
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_gm_to_cbuf"),
                            access_ptr(dst,"w"), access_ptr(src,"r"),
                            0, n*k*2, 0, 1, n*k, 0, 0, 0, 1, "compact_nz_weight")

def load_global(dst, src, out_width, n, k, valid):
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_gm_to_cbuf"),
                            access_ptr(dst,"w"), access_ptr(src,"r"),
                            0, out_width*32, 0, k//16, valid*32, n*32, 0, 0, 1, "global_nz_weight")

def pack_weight(weight,n,k,layout="global"):
    import torch
    o,i=weight.shape
    if layout=="global":
        return weight.reshape(o,i//16,16).transpose(0,1).contiguous()
    p=torch.nn.functional.pad(weight,(0,0,0,(-o)%n))
    return p.reshape(-1,n,i//k,k//16,16).permute(2,0,3,1,4).contiguous().reshape(i//k,-1,n*k//16,16)

def program(input_size,output_size,m,n,k,layout="global",cores=32,cache_hint=0):
    B=T.dynamic("B")
    tiles=(output_size+n-1)//n
    prelude = PRELUDE.replace("false, mode, src_stride", f"false, static_cast<asc_load_l2_cache_mode>({cache_hint}), src_stride")
    ws=(input_size//16,output_size,16) if layout=="global" else (input_size//k,tiles,n*k//16,16)
    @T.prim_func
    def main(X:T.Tensor((B*6,input_size),"bfloat16"), W:T.Tensor(ws,"bfloat16"),
             Y:T.Tensor((T.ceildiv(B*6,m)*m,output_size),"bfloat16"), Meta:T.Tensor((B,),"int32")):
        with T.Kernel(cores,prelude=prelude) as (bx,):
            a=T.alloc_l1((m,k),"bfloat16")
            b=T.alloc_l1((n,k),"bfloat16")
            acc=T.alloc_l0c((m,n),"float32")
            for it in T.Serial(T.ceildiv(T.ceildiv(B*6,m)*tiles-bx,cores)):
                j=(it*cores+bx)%tiles
                r=(it*cores+bx)//tiles*m
                rows=T.min(m,B*6-r)
                cols=T.min(n,output_size-j*n)
                for kk in T.Pipelined(input_size//k,num_stages=2,annotations={"enable_offset":True}):
                    T.copy(X[r:r+rows,kk*k:(kk+1)*k],a[:rows,:])
                    if layout=="global":
                        load_global(b[0,0],W[kk*(k//16),j*n,0],output_size,n,k,cols)
                    else:
                        load_compact(b[0,0],W[kk,j,0,0],n,k)
                    T.gemm(a,b,acc,transpose_B=True,clear_accum=(kk==0))
                T.copy(acc[:,:cols],Y[r:r+m,j*n:j*n+cols])
    return main
