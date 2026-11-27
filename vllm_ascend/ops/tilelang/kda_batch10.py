# SPDX-License-Identifier: Apache-2.0
"""Schedule ten six-token requests on 32 paired Atlas 850E Cube/Vector cores."""

PROJECTION = r"""
template<int N>
__aicore__ inline void limited_q(__gm__ bfloat16_t* W, __gm__ bfloat16_t* X, __gm__ bfloat16_t* Y, int32_t B, int first, int count, int out_col, int pitch) {

  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *acc = (__cc__ float *)0;
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
  for (int32_t it = 0; it < (((B * 3) + 31) >> 5); ++it) {
    for (int32_t kk = 0; kk < 32; ++kk) {
      asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>((kk & 1)));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(1), 0);
      asc_copy_gm2l1_nd2nz((__cbuf__ q_global_nz_weight*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + (kk&1)*N*128), (__gm__ q_global_nz_weight*)(W + kk*3186688 + first*16), 796672, static_cast<asc_load_l2_cache_mode>(0), 8, count*32, N*32, 0);
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
      asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((kk & 1) * 8192) + (2*N*128))), (&(X[((((int64_t)it) * (int64_t)262144) + (((int64_t)kk) * (int64_t)128))])), 8192, static_cast<asc_load_l2_cache_mode>(0), min(64, ((B * 6) - (it * 64))), 128, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>((kk & 1)));
      if (kk == 0) {
        asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
      }
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>((kk & 1)));
      ascend_gemm_l1<64, 128, N, 16, true, bfloat16_t>(((__cc__ float*)acc), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((kk & 1) * 8192) + (2*N*128))), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((kk & 1) * N*128)), (kk == 0), 0, 0);
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>((kk & 1)));
      if (kk == 31) {
        asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
      }
    }
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm(Y + it*64*pitch + out_col, acc, count, 60, pitch, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(16), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
  }
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
}

__aicore__ inline void gate_blocks(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* low,
                                   int pitch, __gm__ bfloat16_t* weights) {
    __cbuf__ bfloat16_t* l1=(__cbuf__ bfloat16_t*)0;
    __cc__ float* acc=(__cc__ float*)0;
    asc_set_gm2l1_nz_para(1,1,64,0);
    asc_copy_gm2l1_nd2nz(l1+131072,low,2*pitch,static_cast<asc_load_l2_cache_mode>(0),60,128,0,0);
    for(int i=0;i<2;++i){
        asc_sync_notify(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(i));
        asc_sync_notify(PIPE_FIX,PIPE_M,static_cast<event_t>(i));
    }
    for(int i=0;i<=16;++i){
        if(i<16){
            int slot=i&1;
            asc_sync_wait(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(slot));
            asc_set_gm2l1_nz_para(1,1,512,0);
            asc_copy_gm2l1_nd2nz(l1+slot*65536,weights+i*65536,256,static_cast<asc_load_l2_cache_mode>(0),512,128,0,0);
            asc_sync_notify(PIPE_MTE2,PIPE_MTE1,static_cast<event_t>(slot));
        }
        if(i>0){
            int slot=(i-1)&1;
            asc_sync_wait(PIPE_MTE2,PIPE_MTE1,static_cast<event_t>(slot));
            asc_sync_wait(PIPE_FIX,PIPE_M,static_cast<event_t>(slot));
            ascend_gemm_l1<64,128,512,32,true,bfloat16_t>(acc+slot*32768,l1+131072,l1+slot*65536,true,0,0);
            asc_sync_notify(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(slot));
            asc_sync_notify(PIPE_M,PIPE_FIX,static_cast<event_t>(slot));
            asc_sync_wait(PIPE_M,PIPE_FIX,static_cast<event_t>(slot));
            asc_set_l0c_copy_nz_para(1,0,0);
            asc_copy_l0c2gm(dst+(i-1)*512,acc+slot*32768,512,60,8192,64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(16), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
            asc_sync_notify(PIPE_FIX,PIPE_M,static_cast<event_t>(slot));
        }
    }
    for(int i=0;i<2;++i){
        asc_sync_wait(PIPE_FIX,PIPE_M,static_cast<event_t>(i));
        asc_sync_wait(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(i));
    }
}
"""

DMA = r"""
__aicore__ inline __cbuf__ float* state_buf(int r) {
    return (__cbuf__ float*)((r & 1)*131072);
}
__aicore__ inline void queue_state(__gm__ float* state,__gm__ int32_t* meta,int b,int rows,int r) {
    int first=meta[b+r]*STATE_STRIDE+(int)block_idx*256;
    int valid=min(256,max(0,rows-first));
    __cbuf__ float* dest=state_buf(r);
    auto event=static_cast<event_t>(4+r%2);
    if(r>=2) asc_sync_wait(PIPE_MTE1,PIPE_MTE2,event);
    if(valid>0) {
        asc_set_gm2l1_nz_para(1,1,256,0);
        asc_copy_gm2l1_nd2nz(dest,state+(int64_t)first*128,512,static_cast<asc_load_l2_cache_mode>(0),valid,128,0,0);
    }
    if(valid<256) {
        asc_sync_pipe(PIPE_MTE2);
        asc_fill_l1((__cbuf__ uint32_t*)(dest+valid*8),(uint32_t)0,{.repeat=16,.blk_num=static_cast<uint64_t>(256-valid),.dst_gap=static_cast<uint64_t>(valid)});
    }
    asc_sync_notify(PIPE_MTE2,PIPE_MTE1,event);
}

__aicore__ inline void queue_output(__gm__ bfloat16_t* w,int k,int offset) {
    asc_copy_gm2l1_align((__cbuf__ uint8_t*)0+offset,(__gm__ uint8_t*)(w+k*1048576+(int)block_idx*2048),16,4096,0,0,false,static_cast<asc_load_l2_cache_mode>(0),131072,4096);
}
"""


CUBE_START = """
const int bid = (int)block_idx;
// Place the dedicated F and G producers on physical cores 0 and 2.
if (bid == 0 || bid == 2) {
    bool g = bid == 2;
    if (g) limited_q<128>(Wi,X,P,B,24768,128,24768,24896);
    else limited_q<192>(Wi,X,P,B,24576,192,24576,24896);
    asc_sync_pipe(PIPE_ALL);
    gate_blocks(g ? GG : GF, P+(g ? 24768 : 24640), 24896, g ? WG : WF);
} else {
    int first = (bid == 1 ? 0 : bid-2)*832;
    limited_q<832>(Wi,X,P,B,first,min(832,24576-first),first,24896);
}
asc_sync_pipe(PIPE_ALL);
asc_sync_intra_arrive(PIPE_FIX,10);
asc_sync_intra_arrive(PIPE_FIX,26);
queue_state(State,Accepted,B,state_rows,0);
"""


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unrecognized TileLang KDA instruction sequence")
    return source.replace(old, new)


def _line(source, *parts):
    line, = (s for s in source.splitlines() if all(part in s for part in parts))
    return line


def schedule_batch10(helpers, calls, layout):
    """Return rewritten phases and entry bodies, retaining the lowered Vector arithmetic."""
    projection, _, middle, output = helpers
    projection = projection[:projection.index("__aicore__ inline void piece_q(")] + PROJECTION
    pos = middle.index("__aicore__ inline void piece_mid(")
    middle = middle[:pos] + DMA.replace("STATE_STRIDE", str(layout[2])) + middle[pos:]
    middle = _replace_once(middle, "int32_t state_rows) {", "int32_t state_rows, __gm__ bfloat16_t* Wo) {")
    a = middle.index("    asc_sync_notify(PIPE_MTE1, PIPE_MTE2", pos)
    z = middle.index("    asc_disable_hf32();", a)
    middle = middle[:a] + middle[z:]
    a = middle.index("      int32_t condval;", a)
    z = middle.index("        asc_sync_intra_wait(PIPE_MTE1", a)
    middle = middle[:a] + """      int r=it-1;
      if(r>=0 && r+1<B) queue_state(State,Accepted,B,state_rows,r+1);
      if(it==1) queue_output(Wo,2,294912);
      if(r>=0 && r<B) {
        asc_sync_wait(PIPE_MTE2,PIPE_MTE1,static_cast<event_t>(4+r%2));
""" + middle[z:]
    middle = _replace_once(middle, "((__cbuf__ float*)buf_dyn_l1 + (((it + 1) & 1) * 32768))", "state_buf(r)")
    a = middle.index("        asc_sync_notify(PIPE_MTE1, PIPE_MTE2", a)
    z = middle.index("      if (2 <= it)", a)
    # The last state reader releases the low L1 slots for the first two Wo panels.
    middle = middle[:a] + """        if(r+2<B) asc_sync_notify(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(4+r%2));
      }
      if(it==4) queue_output(Wo,3,360448);
      if(it==6) queue_output(Wo,4,425984);
      if(it==B) {
        asc_sync_notify(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(7));
        asc_sync_wait(PIPE_MTE1,PIPE_MTE2,static_cast<event_t>(7));
        queue_output(Wo,0,0);
        queue_output(Wo,1,65536);
      }
""" + middle[z:]
    middle = _replace_once(middle, "    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));\n"
                           "    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));", "")
    qkv = _line(middle, "asc_copy_gm2ub_align", "(&(P[", "q_1")
    wait = """          if(it_1==0 && q_1==0) {
            asc_sync_intra_wait(PIPE_MTE2,10);
            asc_sync_inter_arrive(PIPE_MTE2,15);
            asc_sync_inter_wait(PIPE_MTE2,15);
          }
"""
    wg = _line(middle, "asc_copy_gm2ub_align", "(&(WG[")
    wf = _line(middle, "asc_copy_gm2ub_align", "(&(WF[")
    middle = _replace_once(middle, wg+"\n", "")
    middle = _replace_once(middle, wf, wg+"\n"+wf)
    middle = _replace_once(middle, qkv, wait+qkv)
    weight = _line(output, "(&(W[")
    output = _replace_once(output, weight, "        if (kk>=5) "+weight.strip())
    gemm = _line(output, "ascend_gemm_l1<")
    ptr = "((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((kk + 1) & 1) * 32768))"
    resident = "(__cbuf__ bfloat16_t*)(buf_dyn_l1+294912+(kk-3)*65536)"
    output = _replace_once(output, gemm, _replace_once(gemm, ptr, f"((kk>2 && kk<=5) ? {resident} : {ptr})"))
    x = _line(output, "asc_copy_gm2l1_nd2nz", "(&(X[")
    output = _replace_once(output, x, "        if (it==0 && kk==0) { asc_sync_intra_wait(PIPE_MTE2,11); "
                           "asc_sync_intra_wait(PIPE_MTE2,27); }\n"+x)
    mid_call = calls["mid"].replace(");", ", Wo);")
    fence = "asc_sync_pipe(PIPE_ALL);\n"
    cube = CUBE_START + mid_call + fence + calls["o"] + fence
    vector = mid_call + fence + "asc_sync_inter_arrive(PIPE_MTE3,13); asc_sync_inter_wait(PIPE_MTE3,13);\n" + (
        "asc_sync_intra_arrive(PIPE_MTE3,11);\n"
    )
    return [projection, middle, output], cube, vector
