#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils.h"
#include "model_management.hpp"
#include "handle_pool.hpp"

namespace mdaemon {

// Provided by user later to share PIDs with external components (e.g., python side).
void SetPidToShare(const std::vector<int32_t>& npuid_list);

class Daemon {
public:
    Daemon() noexcept = default;
    ~Daemon();

    // Lifecycle controls
    void start();
    void stop();
    bool isRunning() const noexcept { return running_.load(); }
    // Single-iteration runner for driving from python/bindings without a thread
    void runOnce();

    // Model management bindings
    ModelConfig* registerModelFromMacro();
    ModelConfig* getModelConfig(uint64_t model_id);
    bool updateModelState(uint64_t model_id, ModelState new_state);
    ModelState getModelState(uint64_t model_id) const;
    std::vector<uint64_t> listModelIds() const;
    std::vector<int32_t> getCurrentModelNpuids() const;

    // HandlePool bindings
    HandlePool* ensureHandlePool(uint64_t granularity);
    void initializeHandlePoolDevice(uint64_t granularity, int32_t device_id, uint64_t total_bytes);
    size_t handlePoolAvailable(uint64_t granularity, int32_t device_id) const;
    bool extendHandles(uint64_t granularity, int32_t device_id, size_t count);
    bool removeHandles(uint64_t granularity, int32_t device_id, size_t count);

    struct HandlePoolDeviceSnapshot {
        uint64_t granularity{0};
        int32_t device_id{-1};
        uint64_t total_handles{0};
        uint64_t used_handles{0};
        uint64_t available_handles{0};
        uint64_t total_bytes{0};
        uint64_t used_bytes{0};
        uint64_t available_bytes{0};
    };

    struct ModelSnapshot {
        uint64_t model_id{0};
        ModelState state{ModelState::INVALID};
        MessageState message_state{MessageState::INVALID};
        int32_t model_npuid{-1};
        int32_t model_osid{-1};
        uint64_t allocated_bytes{0};
        uint64_t allocated_handles{0};
    };

    std::vector<HandlePoolDeviceSnapshot> snapshotHandlePools() const;
    std::vector<ModelSnapshot> snapshotModels(bool sync_message_space_from_shm = false);
    ModelSnapshot getModelSnapshot(uint64_t model_id, bool sync_message_space_from_shm = false);
    std::vector<std::string> drainLogs();

    // interactive operations between models and handles
    void SetPidToShare(const std::vector<int32_t>& npuid_list);
    bool allocateHandlesForModel(ModelConfig& config);
    void returnHandlesFromModel(ModelConfig& config);

    // for debug
    void printAll() const noexcept {
        std::lock_guard<std::mutex> guard(daemon_mutex_);
        std::cout << "=== Daemon State ===\n";
        model_manager_.printAll();
        for (const auto& pair : handle_pools_) {
            uint64_t granularity = pair.first;
            const HandlePool& pool = *pair.second;
            printf("HandlePool Granularity: %lu\n", granularity);
            pool.printStatus();
        };

    }
private:
    struct AllocatedHandle {
        uint64_t granularity{0};
        int32_t device_id{-1};
        Handle handle{};
    };

    void runLoop();
    HandlePool* findHandlePool(uint64_t granularity);

    std::vector<request_granularity> collectHandleRequests(const ModelMessageSpace& space) const;
    bool allocateForRequest(const request_granularity& req,
                            std::vector<AllocatedHandle>& allocated_handles);
    bool tryAcquireHandleExact(uint64_t granularity, int32_t device_id, Handle& out_handle);
    bool tryAcquireHandle(uint64_t& granularity, int32_t device_id, Handle& out_handle);
    void releaseAllocatedHandles(std::vector<AllocatedHandle>& allocated_handles);
    void writeHandlesToMessageSpace(ModelMessageSpace& space,
                                    const std::vector<AllocatedHandle>& allocated_handles);
    void returnHandlesFromCfg(ModelConfig& config);

    ModelConfigManager model_manager_;
    // granularity to HandlePool map
    std::unordered_map<uint64_t, std::unique_ptr<HandlePool>> handle_pools_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::chrono::steady_clock::time_point next_liveness_check_{std::chrono::steady_clock::time_point::min()};
    mutable std::mutex daemon_mutex_;

};

} // namespace mdaemon
