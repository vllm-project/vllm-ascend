#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include "acl/acl.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);

    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    int64_t K = shape[0];
    int64_t N = shape[1];
    std::vector<int64_t> shapeNz = {N / 32, K / 16, 16, 32};

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shapeNz.data(), shapeNz.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, shapeNz.data(), shapeNz.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret =
            CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, aclFormat::ACL_FORMAT_ND, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                          void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

// 保存二进制数据到文件的函数
void SaveToBinFile(const std::string &filename, const void *data, size_t size, size_t elementSize)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    file.write(static_cast<const char *>(data), size * elementSize);
    file.close();
    std::cout << "数据已保存到: " << filename << std::endl;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t E = 4;
    int64_t M = 8192;
    int64_t N = 4096;
    int64_t K = 7168;
    std::vector<int64_t> xShape = {M, K};
    std::vector<std::vector<int64_t>> weightShape = {{K, N}, {K, N}, {K, N}, {K, N}};
    std::vector<std::vector<int64_t>> weightScaleShape = {{N}, {N}, {N}, {N}};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr[E];
    void *weightScaleDeviceAddr[E];
    void *xScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *outputScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *output = nullptr;
    aclTensor *outputScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<std::vector<int8_t>> weightHostData(E, std::vector<int8_t>(N * K, 0));
    std::vector<std::vector<float>> weightScaleHostData(E, std::vector<float>(N, 0));
    std::vector<float> xScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData = {2048, 4096, 6144, 8192};
    std::vector<int8_t> outputHostData(M * N / 2, 0);
    std::vector<float> outputScaleHostData(M, 0);

    // 读取数据
    const std::string dataDir = "data";

    try {
        // 检查数据目录是否存在
        if (!std::filesystem::exists(dataDir) || !std::filesystem::is_directory(dataDir)) {
            throw std::runtime_error("数据目录不存在: " + dataDir);
        }

        // 读取xHostData
        std::ifstream xFile(dataDir + "/xHostData.bin", std::ios::binary);
        if (!xFile)
            throw std::runtime_error("无法打开xHostData.bin");
        xFile.read(reinterpret_cast<char *>(xHostData.data()), xHostData.size() * sizeof(int8_t));

        // 读取weightHostData
        std::ifstream weightFile(dataDir + "/weightHostData.bin", std::ios::binary);
        if (!weightFile)
            throw std::runtime_error("无法打开weightHostData.bin");
        std::vector<int8_t> tmpWeightHostData(E * N * K, 0);
        weightFile.read(reinterpret_cast<char *>(tmpWeightHostData.data()), tmpWeightHostData.size() * sizeof(int8_t));
        for (int i = 0; i < E; ++i) {
            std::copy(tmpWeightHostData.begin() + i * N * K, tmpWeightHostData.begin() + (i + 1) * N * K,
                      weightHostData[i].begin());
        }

        // 读取weightScaleHostData
        std::ifstream weightScaleFile(dataDir + "/weightScaleHostData.bin", std::ios::binary);
        if (!weightScaleFile)
            throw std::runtime_error("无法打开weightScaleHostData.bin");
        std::vector<float> tmpWeightScaleHostData(E * N, 0);
        weightScaleFile.read(reinterpret_cast<char *>(tmpWeightScaleHostData.data()),
                             tmpWeightScaleHostData.size() * sizeof(float));
        for (int i = 0; i < E; ++i) {
            std::copy(tmpWeightScaleHostData.begin() + i * N, tmpWeightScaleHostData.begin() + (i + 1) * N,
                      weightScaleHostData[i].begin());
        }

        // 读取xScaleHostData
        std::ifstream xScaleFile(dataDir + "/xScaleHostData.bin", std::ios::binary);
        if (!xScaleFile)
            throw std::runtime_error("无法打开xScaleHostData.bin");
        xScaleFile.read(reinterpret_cast<char *>(xScaleHostData.data()), xScaleHostData.size() * sizeof(float));

        std::cout << "所有数据文件已成功读取并赋值给变量" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensorList
    ret = CreateAclTensorListNz<int8_t>(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_INT8, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightScale aclTensorList
    ret = CreateAclTensorList<float>(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr,
                                     aclDataType::ACL_FLOAT, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xScale aclTensor
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                          aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8,
                          aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outputScale aclTensor
    ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          aclFormat::ACL_FORMAT_ND, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulSwigluQuantWeightNZ第一段接口
    ret = aclnnGroupedMatmulSwigluQuantWeightNZTensorListGetWorkspaceSize(x, weight, nullptr, nullptr, weightScale,
                                                                          xScale, groupList, output, outputScale,
                                                                          nullptr, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZTensorListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulSwigluQuantWeightNZ第二段接口
    ret = aclnnGroupedMatmulSwigluQuantWeightNZTensorList(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZTensorList failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    SaveToBinFile("data/out1Data.bin", out1Data.data(), out1Data.size(), sizeof(int8_t));

    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    SaveToBinFile("data/out2Data.bin", out2Data.data(), out2Data.size(), sizeof(float));

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    for (int i = 0; i < E; i++) {
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(weightScaleDeviceAddr[i]);
    }
    aclrtFree(xScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
