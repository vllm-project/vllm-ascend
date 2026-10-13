// Boundary checks use ACL directly: invalid tiling must not enter Torch's async/GIL error path.
#include <acl/acl.h>
#include <aclnnop/aclnn_generic_block_sparse_attention.h>
#include <aclnnop/aclnn_generic_block_sparse_attention_metadata.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void Check(int status)
{
    if (status != 0) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(message == nullptr ? "ACL failure" : message);
    }
}

template<class T>
aclTensor *Tensor(std::vector<int64_t> shape, aclDataType dtype, std::vector<T> data,
                  std::vector<aclTensor *> &tensors, std::vector<void *> &allocations)
{
    void *address = nullptr;
    const uint64_t bytes = data.size() * sizeof(T);
    Check(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    allocations.push_back(address);
    Check(aclrtMemcpy(address, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    auto *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                   shape.data(), shape.size(), address);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    tensors.push_back(tensor);
    return tensor;
}

int main(int argc, char **argv)
{
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : 0;
        Check(aclInit(nullptr));
        Check(aclrtSetDevice(device));
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream));
        std::vector<aclTensor *> tensors;
        std::vector<void *> allocations;
        auto q = Tensor<uint16_t>({1, 16, 128}, ACL_FLOAT16, std::vector<uint16_t>(16 * 128), tensors, allocations);
        auto k = Tensor<uint16_t>({128, 1, 128}, ACL_FLOAT16, std::vector<uint16_t>(128 * 128), tensors, allocations);
        auto v = Tensor<uint16_t>({128, 1, 128}, ACL_FLOAT16, std::vector<uint16_t>(128 * 128), tensors, allocations);
        auto out = Tensor<uint16_t>({1, 16, 128}, ACL_FLOAT16, std::vector<uint16_t>(16 * 128), tensors, allocations);
        auto idx = Tensor<int32_t>({1, 1, 1}, ACL_INT32, {0}, tensors, allocations);
        auto count = Tensor<int32_t>({1, 1}, ACL_INT32, {1}, tensors, allocations);
        auto cuq = Tensor<int64_t>({2}, ACL_INT64, {0, 1}, tensors, allocations);
        auto cukv = Tensor<int64_t>({2}, ACL_INT64, {0, 128}, tensors, allocations);
        auto used = Tensor<int32_t>({1}, ACL_INT32, {128}, tensors, allocations);
        auto meta = Tensor<int32_t>({1024}, ACL_INT32, std::vector<int32_t>(1024), tensors, allocations);
        char layout[] = "TND";
        for (int64_t y : {1, 128, 0, -1, 129, 256}) {
            const bool valid = y == 1 || y == 128;
            const int64_t values[] = {1, y};
            auto *block = aclCreateIntArray(values, 2);
            if (block == nullptr) {
                throw std::runtime_error("aclCreateIntArray failed");
            }
            for (bool metadata : {true, false}) {
                uint64_t workspaceSize = 0;
                aclOpExecutor *executor = nullptr;
                int status;
                if (metadata) {
                    status = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
                        idx, count, cuq, cukv, nullptr, used, 1, 128, 16, 1, 128, block, layout, layout,
                        4, 1, 0, 0, -1, -1, 0, false, meta, &workspaceSize, &executor);
                } else {
                    status = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
                        q, k, v, idx, count, meta, nullptr, nullptr, nullptr, nullptr, nullptr,
                        cuq, cukv, nullptr, used, nullptr, block, layout, layout, 4, 0.08838834764831845,
                        1, 0, 0.0, 0, -1, -1, 0, 0, false, out, nullptr, &workspaceSize, &executor);
                }
                const char *name = metadata ? "metadata" : "main";
                // Main-op upper-bound validation lives in host tiling, which
                // may run at execution rather than GetWorkspaceSize time.
                if (status == 0) {
                    void *workspace = nullptr;
                    if (workspaceSize != 0) {
                        Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
                        allocations.push_back(workspace);
                    }
                    status = metadata ? aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream) :
                                        aclnnGenericBlockSparseAttention(workspace, workspaceSize, executor, stream);
                    if (status == 0) {
                        status = aclrtSynchronizeStream(stream);
                    }
                }
                if (valid) {
                    Check(status);
                    std::cout << "ACCEPT " << name << " Y=" << y << std::endl;
                } else {
                    const char *message = aclGetRecentErrMsg();
                    // Host tiling failures can be wrapped as a generic ACL error.
                    if (status == 0) {
                        throw std::runtime_error("Invalid blockY was unexpectedly accepted");
                    }
                    std::cout << "REJECT " << name << " Y=" << y << " status=" << status << std::endl;
                    if (message != nullptr) {
                        std::cout << message << std::endl;
                    }
                }
            }
            aclDestroyIntArray(block);
        }
        for (auto *tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto *address : allocations) {
            Check(aclrtFree(address));
        }
        Check(aclrtDestroyStream(stream));
        Check(aclrtResetDevice(device));
        Check(aclFinalize());
        std::cout << "ALL PASS: Y=1/128 accepted by both APIs; 8 invalid-Y rejections" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
