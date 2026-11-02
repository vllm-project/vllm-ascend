#pragma once
#include "lib/matmul_intf.h"
#include "lib/matmul/constant_tiling.h"
namespace Const {
template<int N,int BN,int BK>
__aicore__ constexpr MatmulConfig MakeConfig() {
 auto c=GetMDLConfig(false,false,0,false,false,false,true);
 c.basicM=16;c.basicN=BN;c.basicK=BK;
 c.singleCoreM=16;c.singleCoreN=N;c.singleCoreK=2048;
 c.enableSetBias=false;c.enableQuantVector=false;c.enableSetDefineData=false;
 return c;
}
template<bool Fixed,class A,class B,class C,int N,int BN,int BK>struct Mode {
 static constexpr auto Value=GetMDLConfig(false,false,0,false,false,false,true);
};
template<class A,class B,class C,int N,int BN,int BK>struct Mode<true,A,B,C,N,BN,BK>{
 static constexpr auto Config=MakeConfig<N,BN,BK>();
 static constexpr auto Value=AscendC::GetMatmulApiTiling<A,B,C,C>(Config);
};
}
