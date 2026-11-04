#pragma once

/*
This header file is used to define the handle class and handle pool for the daemon process.
It includes necessary libraries and declares the daemon namespace.
*/

#include <cstdint>
#include <limits>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace mdaemon {

class Handle{
public:
    Handle() noexcept = default;
    ~Handle();

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept;
    Handle& operator=(Handle&& other) noexcept;

    bool allocate(int32_t device_id, uint64_t size);
    void release();
    void reset(int32_t device_id, aclrtDrvMemHandle handle, uint64_t granularity, uint64_t shareable_handle) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] int32_t deviceId() const noexcept;
    [[nodiscard]] uint64_t granularity() const noexcept;
    [[nodiscard]] aclrtDrvMemHandle rawHandle() const noexcept;
    [[nodiscard]] uint64_t shareableHandle() const noexcept;

    uint64_t exportShareableHandle();

private:
    void cleanup() noexcept;

    aclrtDrvMemHandle handle_{};
    int32_t device_id_{-1};
    uint64_t granularity_{0};
    bool owns_handle_{false};
    uint64_t shareable_handle_{std::numeric_limits<uint64_t>::max()};
};


class HandlePool {
public:
    explicit HandlePool(uint64_t page_size = 2ULL * 1024 * 1024);
    ~HandlePool();
    
    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    void initializeDevice(int32_t device_id,
                         uint64_t total_bytes,
                         const std::vector<int32_t>& npuid_list = {});
    Handle acquire(int32_t device_id);
    void release(int32_t device_id, Handle handle);
    // void exportShareableHandles(int32_t device_id);
    size_t available(int32_t device_id) const;
    size_t total(int32_t device_id) const;
    size_t used(int32_t device_id) const;
    uint64_t pageSize() const noexcept { return page_size_; }
    void shutdown();

    bool extendHandles(int32_t device_id,
                      size_t additional_count,
                      const std::vector<int32_t>& npuid_list = {});
    bool removeHandles(int32_t device_id, size_t remove_count);

    void memSetPidToShare(const std::vector<int32_t>& npuid_list, int32_t device_id);

    // Returns all device ids that have been initialized in this pool.
    std::vector<int32_t> listDeviceIds() const;

    // for debug
    void printStatus() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : device_handle_map_) {
            int32_t device_id = pair.first;
            size_t handle_count = pair.second.size();
            printf("Device ID: %d, Available Handles: %zu\n", device_id, handle_count);
        }
    }

private:
    // Map from device_id to a list of handles
    std::unordered_map<int32_t, std::deque<Handle>> device_handle_map_;

    // Map from device_id to total number of handles allocated for this pool.
    // Invariants (when initialized): total = available + used.
    std::unordered_map<int32_t, size_t> device_total_handle_map_;

    // Map from device_id to shareable handles, this is used to quickly set pid to 
    // shareable handles when new model comes in
    std::unordered_map<int32_t, std::vector<uint64_t>> device_shareable_handle_map_;

    uint64_t page_size_;
    mutable std::mutex mutex_;
    bool running_;

    void allocateHandles(int32_t device_id,
                        size_t count,
                        const std::vector<int32_t>& npuid_list = {});
    void ensureInitialized(int32_t device_id);
};

} // namespace mdaemon

