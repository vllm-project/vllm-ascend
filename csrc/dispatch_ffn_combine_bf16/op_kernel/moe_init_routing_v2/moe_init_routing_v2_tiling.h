#pragma once

#include "tiling_base.h"


namespace optiling {
const static int64_t TILING_KEY_DROPLESS_SORT_ONE_CORE = 10001;
const static int64_t TILING_KEY_DROPLESS_SORT_MULTI_CORE = 10002;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE = 10011;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE = 10012;
const static int64_t TILING_KEY_HIGH_PERFORMANCE = 20000;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t INDEX_INPUT_X = 0;
const static int64_t INDEX_INPUT_EXPERT_IDX = 1;
const static int64_t ATTR_ACTIVE_ROWS = 0;
const static int64_t ATTR_EXPERT_CAPACITY = 1;
const static int64_t ATTR_EXPERT_NUM = 2;
const static int64_t ATTR_DROP_PAD_MODE = 3;
const static int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
const static int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
const static int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
const static int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_TOKENS_COUNT = 2;
const static int64_t ONE_CORE_SORT_BUFFER_310P = 24;


inline static int64_t CeilLog4(int64_t x) {
  return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y) {
  if (y == 0) {
    return 0;
  }
  return x <= y ? x : x % y;
}

template <class T>
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}


struct MoeV2VBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};

struct MoeV2VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};

struct MoeV2SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};

struct MoeV2GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t activateRows = 0;
    int64_t perCoreRows = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t perCoreLoops = 0;
    int64_t lastCoreLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
    int64_t colLoops = 0;
};

struct MoeInitRoutingV2TilingData {
    int64_t coreNum;
    int64_t n;
    int64_t cols;
    int64_t k;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    int64_t expertTokensBeforeCapacityFlag;
    MoeV2VBSComputeTilingData vbsComputeParamsOp;
    MoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV2SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    MoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};


class MoeInitRoutingV2TilingBase : public TilingBaseClass {

protected:
  bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) override;
  bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity,
  int64_t expertNum, int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
  bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) override;

  bool DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  bool GetWorkspaceSize() override;


protected:
  bool CheckTokenCount(int64_t num, const char* tag);

  void Tiling4GatherOutCompute();
  void Tiling4SrcToDstCompute();
  void Tiling4SrcToDstCapacityCompute();
  void Tiling4SortOutCompute();
  void Tiling4VMSMiddleCompute();
  void Tiling4VBSCompute();
  void ShowTilingData();
  void Tiling4VBSMultiCoreCompute(MoeV2VBSComputeTilingData* tilingData);
  void Tiling4VBSOneCoreCompute(MoeV2VBSComputeTilingData* tilingData);
  bool IsFullLoad();




  int64_t aivNum = 0;
  int64_t sortLoopMaxElement = 0;
  int64_t mrgSortListMaxElement = 2040;
  int64_t totalLength = 0;
  int64_t activateNum = 0;
  int64_t expertCapacity = 0;
  int64_t expertNum = 0;
  int64_t dropPadMode = 0;
  int64_t expertTokensCountOrCumsumFlag = 0;
  bool expertTokensBeforeCapacityFlag = false;
  int64_t inuptXDtypeSize_ = 0;
  bool isFullLoad = false;
  const char *opName = "DispatchFFNCombine Tiling Debug";

public:
  MoeInitRoutingV2TilingData moeInitRoutingTilingData;
};


bool MoeInitRoutingV2TilingBase::DoOpTiling() {
  sortLoopMaxElement =
      (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
  isFullLoad = IsFullLoad();
  Tiling4VBSCompute();
  Tiling4VMSMiddleCompute();
  Tiling4SortOutCompute();
  Tiling4SrcToDstCompute();
  Tiling4SrcToDstCapacityCompute();
  Tiling4GatherOutCompute();
  ShowTilingData();
  return true;
};

uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_HIGH_PERFORMANCE;
  }
  if (dropPadMode == 0) {
    if (totalLength <= sortLoopMaxElement) {  
      return TILING_KEY_DROPLESS_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROPLESS_SORT_MULTI_CORE;
    }
  } else {
    if (totalLength <= sortLoopMaxElement) {
      return TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
    }
  }
  return tilingKey_;
}


void MoeInitRoutingV2TilingBase::ShowTilingData()
{
    OP_LOGD(opName,
              "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld, k:%ld, expertCapacity:%ld, expertNum:%ld, "
              "dropPadMode:%ld, expertTokensCountOrCumsumFlag:%ld, expertTokensBeforeCapacityFlag:%ld",
              moeInitRoutingTilingData.coreNum, moeInitRoutingTilingData.n,
              moeInitRoutingTilingData.cols, moeInitRoutingTilingData.k,
              moeInitRoutingTilingData.expertCapacity, moeInitRoutingTilingData.expertNum,
              moeInitRoutingTilingData.dropPadMode, moeInitRoutingTilingData.expertTokensCountOrCumsumFlag,
              moeInitRoutingTilingData.expertTokensBeforeCapacityFlag);
    OP_LOGD(opName,
              "MoeV2VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
              "perCorePerLoopElements:%ld, "
              "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
              "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld",
              moeInitRoutingTilingData.vbsComputeParamsOp.needCoreNum,
              moeInitRoutingTilingData.vbsComputeParamsOp.perCoreElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.perCoreLoops,
              moeInitRoutingTilingData.vbsComputeParamsOp.perCorePerLoopElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.perCoreLastLoopElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.lastCoreElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.lastCoreLoops,
              moeInitRoutingTilingData.vbsComputeParamsOp.lastCorePerLoopElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.lastCoreLastLoopElements,
              moeInitRoutingTilingData.vbsComputeParamsOp.oneLoopMaxElements);
    OP_LOGD(opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
              moeInitRoutingTilingData.vmsMiddleComputeParamsOp.needCoreNum);
    OP_LOGD(opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
              moeInitRoutingTilingData.sortOutComputeParamsOp.oneLoopMaxElements);
    OP_LOGD(
        opName,
        "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
        moeInitRoutingTilingData.srcToDstComputeParamsOp.needCoreNum,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.activateRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.perCoreRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.perCorePerLoopRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.perCoreLastLoopRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.lastCoreRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.lastCorePerLoopRows,
        moeInitRoutingTilingData.srcToDstComputeParamsOp.lastCoreLastLoopRows);
    OP_LOGD(opName,
              "SrcToDstComputeCapacityTilingData is needCoreNum:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
              "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.needCoreNum,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.perCoreRows,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.perCorePerLoopRows,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.perCoreLastLoopRows,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.lastCoreRows,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.lastCorePerLoopRows,
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.lastCoreLastLoopRows);
    OP_LOGD(
        opName,
        "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
        moeInitRoutingTilingData.gatherOutComputeParamsOp.needCoreNum,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.activateRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.perCoreRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.perCorePerLoopRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.perCoreLastLoopRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.lastCoreRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.lastCorePerLoopRows,
        moeInitRoutingTilingData.gatherOutComputeParamsOp.lastCoreLastLoopRows);
}



bool MoeInitRoutingV2TilingBase::GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity,
  int64_t expertNum, int64_t activateNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
  bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) {

  this->activateNum = activateNum;
  this->expertCapacity = expertCapacity;
  this->expertNum = expertNum;
  this->dropPadMode = dropPadMode;
  this->expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
  this->expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
  if (dropPadMode == 1) {
    expertTokensCountOrCumsumFlag = 0;
  } else {
    expertTokensBeforeCapacityFlag = false;
  }
  moeInitRoutingTilingData.cols = cols;
  moeInitRoutingTilingData.n = m;
  moeInitRoutingTilingData.k = topK;
  moeInitRoutingTilingData.expertCapacity = expertCapacity;
  moeInitRoutingTilingData.expertNum = expertNum;
  moeInitRoutingTilingData.dropPadMode = dropPadMode;
  moeInitRoutingTilingData.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
  moeInitRoutingTilingData.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
  totalLength = moeInitRoutingTilingData.n * moeInitRoutingTilingData.k;
  inuptXDtypeSize_ = inuptXDtypeSize;
  return true;
}

bool MoeInitRoutingV2TilingBase::GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) {
  aivNum = aivCoreNum;
  aicoreParams_.blockDim = aivCoreNum;
  aicoreParams_.ubSize = ubSizePlatForm;
  moeInitRoutingTilingData.coreNum = aivCoreNum;
  return true;
}

bool MoeInitRoutingV2TilingBase::GetWorkspaceSize() {
  size_t sortWorkspaceSize = totalLength * sizeof(float) * NUM_TWO * NUM_THREE;
  size_t scatterWorkspaceSize = totalLength * sizeof(int32_t) * NUM_TWO;
  size_t expertTokenFlagSize = aivNum * 2 * sizeof(int32_t);
  workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
  return true;
}


void MoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(MoeV2VBSComputeTilingData* tilingData) {
    tilingData->needCoreNum = 1;
    tilingData->perCoreElements = totalLength;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = tilingData->perCoreElements;
    tilingData->perCoreLastLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreElements = tilingData->perCoreElements;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreLastLoopElements = tilingData->perCoreElements;
}

void MoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(MoeV2VBSComputeTilingData* tilingData) {
      //Tiling4VBSMultiCoreCompute
      int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);
      needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
      needCoreNum = std::min(needCoreNum, aivNum);
      if (needCoreNum > 0) {
          int64_t perCoreElements = totalLength / needCoreNum;
          int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
          int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
          int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
          if (lastCoreElement > alineCeilPerCoreElements) {
            perCoreElements = alineCeilPerCoreElements;
            needCoreNum = CeilDiv(totalLength, perCoreElements);
          } else {
            perCoreElements = alineFloorPerCoreElements;
          }
          tilingData->needCoreNum = needCoreNum;
          do {
              tilingData->perCoreElements = perCoreElements;
              tilingData->perCoreLoops = CeilDiv(tilingData->perCoreElements, sortLoopMaxElement);  // 每个核处理的loop数
              tilingData->perCorePerLoopElements = std::min(tilingData->perCoreElements, sortLoopMaxElement);
              tilingData->perCoreLastLoopElements = tilingData->perCoreElements - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopElements;
              tilingData->lastCoreElements = totalLength - (tilingData->needCoreNum - 1) * tilingData->perCoreElements;
              tilingData->lastCoreLoops = tilingData->perCoreLoops;
              int64_t tmp = CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops);
              int64_t lastCorePerLoopElements =
                  CeilDiv(CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops), SORT32_ALIGN_ELEMENT) *
                  SORT32_ALIGN_ELEMENT;
              tilingData->lastCorePerLoopElements = lastCorePerLoopElements;
              tilingData->lastCoreLastLoopElements = tilingData-> lastCoreElements - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopElements;
              perCoreElements -= SORT32_ALIGN_ELEMENT;
          } while (tilingData->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
      }
}

void MoeInitRoutingV2TilingBase::Tiling4VBSCompute() {
  auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  tilingData->oneLoopMaxElements = sortLoopMaxElement;
  if (totalLength <= sortLoopMaxElement) {
    Tiling4VBSOneCoreCompute(tilingData);
    return;
  }
  Tiling4VBSMultiCoreCompute(tilingData);
}

void MoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute() {
  auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
  if (vbsComputeTilingData->needCoreNum <= MRG_LIST_NUM) {
      tilingData->needCoreNum = 0;
  } else {
      int64_t needCoreNum = CeilDiv(vbsComputeTilingData->needCoreNum, MRG_LIST_NUM);
      tilingData->needCoreNum = needCoreNum;
  }
}

void MoeInitRoutingV2TilingBase::Tiling4SortOutCompute() {
  auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
  tilingData->oneLoopMaxElements = mrgSortListMaxElement;
}


void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

  int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                           (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->needCoreNum = 0;
    return;
  }

  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->needCoreNum = needCoreNum;
  int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
  tilingData->perCoreRows = perCoreRows;
  if (perLoopMaxRows >= tilingData->perCoreRows) {
      tilingData->perCorePerLoopRows = tilingData->perCoreRows;
      tilingData->perCoreLastLoopRows = tilingData->perCoreRows;
  } else {
      tilingData->perCorePerLoopRows = perLoopMaxRows;
      tilingData->perCoreLastLoopRows = tilingData->perCoreRows - (CeilDiv(tilingData->perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
  }
  tilingData->lastCoreRows = lastCoreNum;
  if (perLoopMaxRows >= tilingData->lastCoreRows) {
      tilingData->lastCorePerLoopRows = tilingData->lastCoreRows;
      tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows;
  } else {
      tilingData->lastCorePerLoopRows = perLoopMaxRows;
      tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows - (CeilDiv(tilingData->lastCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
  }
}

void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);

  if (perCoreRows <= 0) {
    tilingData->needCoreNum = 0;
    return;
  }

  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->needCoreNum = needCoreNum;
  int64_t cols = moeInitRoutingTilingData.cols;
  tilingData->perCoreRows = perCoreRows;
  int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
  tilingData->lastCoreRows = lastCoreRows;


  int64_t rowSize =
      (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
  int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

  if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
    tilingData->perCorePerLoopRows = perCoreRows;
    tilingData->perCoreLastLoopRows = perCoreRows;
    tilingData->lastCorePerLoopRows = lastCoreRows;
    tilingData->lastCoreLastLoopRows = lastCoreRows;
    tilingData->perCoreLoops = 1;
    tilingData->lastCoreLoops = 1;
    tilingData->perLoopCols = cols;
    tilingData->lastLoopCols = cols;
    tilingData->colLoops = 1;

  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
    int64_t baseMaxColsSize = (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    if (cols < MAX_COLS_ONE_LOOP) {
      basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                           static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols =
          (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }
    tilingData->perLoopCols = (std::min(baseMaxCols, cols));
    tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
    tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);
    tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
  }
}

void MoeInitRoutingV2TilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->activateRows = totalLength;
    if (dropPadMode == 0) {
        tilingData->activateRows = activateNum;
    }
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0 || moeInitRoutingTilingData.cols <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }
    tilingData->needCoreNum = CeilDiv(totalLength, perCoreRows);
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    int64_t rowSize = (perCoreRows * sizeof(int32_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
        tilingData->perCorePerLoopRows = perCoreRows;
        tilingData->perCoreLastLoopRows = perCoreRows;
        tilingData->lastCorePerLoopRows = lastCoreRows;
        tilingData->lastCoreLastLoopRows = lastCoreRows;
        tilingData->perCoreLoops = 1;
        tilingData->lastCoreLoops = 1;
        tilingData->perLoopCols = cols;
        tilingData->lastLoopCols = cols;
        tilingData->colLoops = 1;
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - baseMaxColsSize) /
                                     static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - colSize) /
                                 static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - rowSize) / inuptXDtypeSize_ /
                          ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        }
        tilingData->perLoopCols = (std::min(baseMaxCols, cols));
        tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
        tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}


bool MoeInitRoutingV2TilingBase::IsFullLoad()
{
    if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.cols > MAX_COLS_ONE_LOOP ||
        this->dropPadMode == 1) {
        return false;
    }
    int64_t sortBufferNum = ONE_CORE_SORT_BUFFER;

    int64_t sortSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * sortBufferNum;
    int64_t otherSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace = CeilDiv(this->expertNum * int64_t(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    int64_t perCoreXRows = moeInitRoutingTilingData.n / aivNum;
    int64_t remainder = moeInitRoutingTilingData.n % aivNum;
    // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
    perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
    int64_t gatherSpace =
        CeilDiv(moeInitRoutingTilingData.cols * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreXRows;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - sortSpace - otherSpace - expertSpace - gatherSpace;
    return remainUbAfterSort > 0;
}

}