#pragma once
namespace optiling {
struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

class TilingBaseClass {
public:
    bool DoTiling(
            int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, 
            int64_t expertNum, int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
            bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0, 
            int64_t aivCoreNum, int64_t ubSizePlatForm) 
    {
        bool ret = GetShapeAttrsInfo(m, cols, topK, expertCapacity, expertNum, activeNum, dropPadMode, expertTokensCountOrCumsumFlag,
            expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0);

        if (!ret){
            return ret;
        }
        ret = GetPlatformInfo(aivCoreNum, ubSizePlatForm);
        if (!ret){
            return ret;
        }
        ret = DoOpTiling();
        if (!ret){
            return ret;
        }
        ret = GetWorkspaceSize();
        if (!ret){
            return ret;
        }
        ret = PostTiling();
        if (!ret){
            return ret;
        }
        tilingKey_ = GetTilingKey();
        
        return true;
    }

//protected:
    virtual bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) = 0;
    virtual bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, 
        int64_t expertNum, int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
        bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) = 0;

    virtual bool DoOpTiling() = 0;
    virtual bool GetWorkspaceSize() = 0;
    virtual bool PostTiling() = 0;
    virtual uint64_t GetTilingKey() const = 0;
//protected:
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0, 0, 0, 0, 0, 0, 0};
};

}