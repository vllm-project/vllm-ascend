// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Row-strided fusion follows sgl-kernel-npu/norm/add_rmsnorm_bias.py:
// round the residual addition, reduce in FP32, then fuse affine and bias.
#ifndef ADD_RMS_NORM_BIAS_A5_H_
#define ADD_RMS_NORM_BIAS_A5_H_
#include "kernel_operator.h"
using namespace AscendC;

template <typename T> class KernelAddRmsNormBiasA5 {
public:
    __aicore__ inline explicit KernelAddRmsNormBiasA5(TPipe* p) : pipe(p) {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
        GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
        const AddRMSNormBiasTilingData* t) {
        rows=t->num_row; cols=t->num_col; eps=t->epsilon;
        hasBias=!t->nullptr_beta;
        in1.SetGlobalBuffer((__gm__ T*)x1);
        in2.SetGlobalBuffer((__gm__ T*)x2);
        weight.SetGlobalBuffer((__gm__ T*)gamma);
        bias.SetGlobalBuffer((__gm__ T*)beta);
        out.SetGlobalBuffer((__gm__ T*)y);
        residual.SetGlobalBuffer((__gm__ T*)x);
        inv.SetGlobalBuffer((__gm__ float*)rstd);
        pipe->InitBuffer(fp, cols*6*sizeof(float));
        pipe->InitBuffer(raw, cols*2*sizeof(T));
    }
    template<HardEvent E> __aicore__ inline void Sync() {
        auto e=static_cast<event_t>(pipe->FetchEventID(E));
        SetFlag<E>(e); WaitFlag<E>(e);
    }
    __aicore__ inline void ToFloat(LocalTensor<float> d, LocalTensor<T> s) {
        if constexpr (std::is_same<T,float>::value) {
            Adds(d,s,0.0f,cols);
        } else {
            Cast(d,s,RoundMode::CAST_NONE,cols);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FromFloat(LocalTensor<T> d, LocalTensor<float> s) {
        if constexpr (std::is_same<T,float>::value) {
            Adds(d,s,0.0f,cols);
        } else {
            Cast(d,s,RoundMode::CAST_RINT,cols);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        auto sum=fp.Get<float>(); auto other=sum[cols];
        auto w=sum[2*cols]; auto b=sum[3*cols];
        auto square=sum[4*cols]; auto tmp=sum[5*cols];
        auto a=raw.Get<T>(); auto r=a[cols];
        DataCopy(a,weight,cols); Sync<HardEvent::MTE2_V>();
        ToFloat(w,a); Sync<HardEvent::V_MTE2>();
        if(hasBias) {
            DataCopy(a,bias,cols); Sync<HardEvent::MTE2_V>();
            ToFloat(b,a); Sync<HardEvent::V_MTE2>();
        }
        for(uint32_t row=GetBlockIdx(); row<rows; row+=GetBlockNum()) {
            uint64_t offset=static_cast<uint64_t>(row)*cols;
            DataCopy(a,in1[offset],cols); DataCopy(r,in2[offset],cols);
            Sync<HardEvent::MTE2_V>();
            ToFloat(sum,a); ToFloat(other,r);
            Add(sum,sum,other,cols); PipeBarrier<PIPE_V>();
            FromFloat(a,sum); ToFloat(sum,a);
            Sync<HardEvent::V_MTE3>(); DataCopy(residual[offset],a,cols);
            Mul(square,sum,sum,cols); PipeBarrier<PIPE_V>();
            ReduceSum(other,square,tmp,cols); PipeBarrier<PIPE_V>();
            Muls(other,other,1.0f/cols,1); PipeBarrier<PIPE_V>();
            Adds(other,other,eps,1); PipeBarrier<PIPE_V>();
            Sqrt(other,other,1); PipeBarrier<PIPE_V>();
            Duplicate(tmp,1.0f,1); PipeBarrier<PIPE_V>();
            Div(other,tmp,other,1); PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams scalarCopy{1,sizeof(float),0,0,0};
            DataCopyPad(inv[row],other,scalarCopy);
            Sync<HardEvent::V_S>();
            float scale=other.GetValue(0);
            Sync<HardEvent::S_V>();
            Muls(sum,sum,scale,cols); PipeBarrier<PIPE_V>();
            Mul(sum,sum,w,cols); PipeBarrier<PIPE_V>();
            if(hasBias) { Add(sum,sum,b,cols); PipeBarrier<PIPE_V>(); }
            Sync<HardEvent::MTE3_V>();
            FromFloat(a,sum); Sync<HardEvent::V_MTE3>();
            DataCopy(out[offset],a,cols);
            Sync<HardEvent::MTE3_MTE2>();
            Sync<HardEvent::V_MTE2>();
        }
    }
private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> fp,raw;
    GlobalTensor<T> in1,in2,weight,bias,out,residual;
    GlobalTensor<float> inv;
    uint32_t rows,cols; float eps; bool hasBias;
};
#endif
