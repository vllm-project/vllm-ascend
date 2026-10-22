#pragma once

/*
This header file is used to define the model configuration for each model
in the daemon process. 
[NOTE(shijin)]: Current version only supports single-npu service, so this is
a placeholder for future multi-npu support.
*/

/*
 * model_management hierarchy
 *
 * ModelMacroSpace (shared macro region for semaphore-driven registration)
 * ├─ fields: shm_name, current_model_npuid, current_model_osid
 * ├─ helpers: reset(), hasCandidate(), get/set shm_name
 *
 * ModelMessageSpace (per-model message sync area)
 * ├─ fields: shm_name, state, model identifiers, handle list
 * ├─ helpers: get/set shm_name
 *
 * ModelConfig
 * ├─ Fields: model_id, state, ModelMessageSpace, mutex
 * ├─ Methods: setModelConfig(), get/set state, messageSpace accessors
 * └─ Static helpers: generateModelId(), buildShmName()
 *
 * ModelMacroRegistry (encapsulates MODEL_REG_SHM shared memory)
 * ├─ Opens/Mmaps FILE /dev/shm/vllm-model-registry
 * ├─ Exposes: data(), hasPendingRegistration(), writeShmName()
 * └─ Cleans up mapping and descriptor in cleanup()
 *
 * ModelConfigManager
 * ├─ Manages unordered_map<model_id, ModelConfig>
 * ├─ Guarded by map_mutex_
 * ├─ Operations: add/remove/get/update/isDone
 * ├─ Macro features: macroSpace(), registerModelFromMacro()
 * └─ Tracks total_models_ and holds ModelMacroRegistry instance
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <semaphore.h>

#include "utils.h"
#include "handle_pool.hpp"

namespace mdaemon {

namespace {
constexpr size_t kModelShmNameLength = 256;
constexpr int32_t kInvalidModelHandle = std::numeric_limits<int32_t>::max();

inline std::string_view defaultShmName() noexcept {
    static const std::string value = std::string(getShmPrefix()) + "null";
    return std::string_view(value);
}

inline std::string_view readShmName(const std::array<char, kModelShmNameLength>& buffer) noexcept {
    return std::string_view(buffer.data(), std::strlen(buffer.data()));
}

inline void writeShmName(std::array<char, kModelShmNameLength>& buffer, std::string_view value) noexcept {
    std::fill(buffer.begin(), buffer.end(), '\0');
    if (value.empty()) {
        return;
    }
    const size_t copy_count = std::min(value.size(), buffer.size() - 1);
    std::memcpy(buffer.data(), value.data(), copy_count);
}

inline bool isDefaultShmName(std::string_view value) noexcept {
    const std::string_view null_name = defaultShmName();
    return value.empty() || value == null_name;
}
} // namespace


// Model state class
enum class ModelState : uint8_t {
    UNINITIALIZED = 0, // model is not initialized
    INITIALIZED = 1, // model is initialized
    REGISTERED = 2, // model is registered with daemon
    // LOADING = 3, // model is loading weights
    // READY = 4, // model is ready to run
    // RUNNING = 5, // model is running
    // OFFLOADED = 6, // model is offloaded to CPU
    DONE = 7, // model has finished execution or killed by force
    INVALID = 8 // invalid state
};

// Model message space between daemon and model process
enum class MessageState : uint8_t {
    NONE = 0,
    REQUEST_REGISTER = 1, // model process requests registration
    REGISTERED = 2, // daemon acknowledges registration
    REQUEST_GET_HANDLES = 3, // model process requests handles
    HANDLES_READY = 4, // daemon acknowledges handles are ready
    REQUEST_RETURN_HANDLES = 5, // model process requests to return handles
    HANDLES_RETURNED = 6, // daemon acknowledges handles are returned
    DONE = 7, // model has finished execution
    INVALID = 8 // invalid state
};

struct pair_hash {
    size_t operator()(const std::pair<int32_t, uint64_t>& value) const noexcept {
        const size_t h1 = std::hash<int32_t>{}(value.first);
        const size_t h2 = std::hash<uint64_t>{}(value.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct handle_granularity {
    uint64_t shareable_handle{std::numeric_limits<uint64_t>::max()};
    uint64_t granularity{0};
    int32_t device_id{ -1 };
};

struct request_granularity {
    uint64_t request_num{0};
    uint64_t granularity{0};
    int32_t device_id{ -1 };
};

struct ModelMacroSpace {
    ModelMacroSpace() noexcept {
        reset();
    }

    std::string_view getShmName() const noexcept {
        return readShmName(shm_name);
    }

    void setShmName(std::string_view value = defaultShmName()) noexcept {
        writeShmName(shm_name, value);
    }

    void reset() noexcept {
        setShmName();
        current_model_npuid = kInvalidModelHandle;
        current_model_osid = kInvalidModelHandle;
    }

    bool hasCandidate() const noexcept {
        return current_model_npuid != kInvalidModelHandle &&
               current_model_osid != kInvalidModelHandle;
    }

    std::array<char, kModelShmNameLength> shm_name{};
    int32_t current_model_npuid{kInvalidModelHandle};
    int32_t current_model_osid{kInvalidModelHandle};
};

struct ModelMessageSpace {
    ModelMessageSpace() noexcept {
        setShmName();
    }

    std::string_view getShmName() const noexcept {
        return readShmName(shm_name);
    }

    void setShmName(std::string_view value = defaultShmName()) noexcept {
        writeShmName(shm_name, value);
    }

    std::array<char, kModelShmNameLength> shm_name{};
    MessageState state{MessageState::NONE};
    int32_t model_npuid{kInvalidModelHandle};
    int32_t model_osid{kInvalidModelHandle};
    int64_t heartbeat_ns{0};
    // handle_info_list containing all the handles in exchange between model and daemon
    std::array<handle_granularity, MAX_HANDLES_PER_MODEL> handle_info_list{};
    // active area in handle_info_list is [offset_st, offset_ed)
    int32_t offset_st{0}; // offset start for handle access, usually 0
    int32_t offset_ed{0}; // current total handles in the list
    // request list for model to request handles with specific granularity and device_id
    std::array<request_granularity, MAX_HANDLES_PER_MODEL> handle_request_list{};
};

class ModelMacroRegistry;

class ModelConfig {
public:
    ModelConfig() noexcept = default;
    ~ModelConfig();

    bool setModelConfig() noexcept;
    uint64_t getModelId() const noexcept;
    ModelState getState() const noexcept;
    void setState(ModelState new_state) noexcept;

    ModelMessageSpace& messageSpace();
    const ModelMessageSpace& messageSpace() const;
    bool syncMessageSpaceFromShm() noexcept;
    bool syncMessageSpaceToShm() noexcept;
    bool tryLockMessageSpace() noexcept;
    void unlockMessageSpace() noexcept;

    void addAllocatedHandle(int32_t canonical_device_id, Handle&& handle);
    void setPidToShareForAllocated(const std::vector<int32_t>& npuid_list);
    bool takeAllocatedHandle(int32_t device_id, uint64_t shareable_handle, Handle& out_handle);
    std::vector<Handle> drainAllocatedHandles();

    size_t allocatedHandleCount() const noexcept;
    uint64_t allocatedBytes() const noexcept;
    void unlinkMessageShm() noexcept;

private:
    // model id : a generated hash value
    uint64_t model_id_{std::numeric_limits<uint64_t>::max()};
    // model state
    ModelState state_{ModelState::UNINITIALIZED};
    // add shm space between model and daemon to sync information (direct mmap region)
    ModelMessageSpace* message_space_{nullptr};
    int message_fd_{-1};
    std::string message_shm_name_;
    sem_t* message_sem_{nullptr};
    std::string message_sem_name_;
    mutable std::mutex state_mutex_;
    // allocated handles for this model, to be released when model is done
    // member in allocated_handles_: key -- (device_id, shareable_handle), value -- Handle
    std::unordered_map<std::pair<int32_t, uint64_t>, Handle, pair_hash> allocated_handles_;
    mutable std::mutex allocated_mutex_;

    bool openAndMapMessageShm(bool create) noexcept;
    void closeMessageShmMapping() noexcept;
    bool openMessageSemaphore(bool create) noexcept;
    void closeMessageSemaphore() noexcept;

    static uint64_t generateModelId() noexcept;
    static std::string buildShmName(uint64_t model_id);

    friend class ModelConfigManager;
};

class ModelConfigManager {
public:
    ModelConfigManager() noexcept;
    ~ModelConfigManager();

    ModelConfig* addModelConfig();
    void removeModelConfig(uint64_t model_id);
    ModelConfig* getModelConfig(uint64_t model_id);

    ModelMacroSpace* macroSpace() const noexcept;
    ModelConfig* registerModelFromMacro();
    std::vector<uint64_t> listModelIds() const;

    bool isModelDone(uint64_t model_id);
    bool updateModelState(uint64_t model_id, ModelState new_state);
    ModelState getModelState(uint64_t model_id) const noexcept;

    int32_t getTotalModels() const noexcept {
        std::lock_guard<std::mutex> guard(map_mutex_);
        return total_models_;
    }
    std::vector<int32_t> getCurrentModelNpuids() const noexcept {
        std::lock_guard<std::mutex> guard(map_mutex_);
        return current_model_npuids_;
    }

    // for debug
    void printModelConfigs() const noexcept {
        std::lock_guard<std::mutex> guard(map_mutex_);
        printModelConfigsLocked();
    }

    void printAll() const noexcept {
        std::lock_guard<std::mutex> guard(map_mutex_);
        printf("Total Models: %d\n", total_models_);
        printModelConfigsLocked();
        ModelMacroSpace* macro_space = macroSpace();
        if (macro_space) {
            std::string_view shm_name = macro_space->getShmName();
                 printf("Macro Space Shm Name: %.*s\n",
                     static_cast<int>(shm_name.size()),
                     shm_name.data());
                 printf("  Current Model NPU ID: %d, Current Model OS ID: %d\n",
                     macro_space->current_model_npuid,
                     macro_space->current_model_osid);
        } else {
            printf("Macro Space is null.\n");
        }
    }
private:
    void printModelConfigsLocked() const noexcept {
        for (const auto& kv : model_config_map_) {
            const ModelConfig& config = *kv.second;
            const ModelMessageSpace& msg_space = config.messageSpace();
            std::string_view shm_name = msg_space.getShmName();
            ModelState state = config.getState();
            printf("Model ID: %lu, State: %d, Shm Name: %.*s\n",
                   config.getModelId(),
                   static_cast<int>(state),
                   static_cast<int>(shm_name.size()),
                   shm_name.data());
            // print message space info
                 printf("  Message State: %d, Model NPU ID: %d, Model OS ID: %d\n",
                     static_cast<int>(msg_space.state),
                     msg_space.model_npuid,
                     msg_space.model_osid);
            printf("  first 3 Handles:\n");
            for (size_t i = 0; i < 3; ++i) {
                const handle_granularity& handle_info = msg_space.handle_info_list[i];
                printf("    Handle %zu: Shareable Handle: %lu, Granularity: %lu, Device ID: %d\n",
                       i,
                       handle_info.shareable_handle,
                       handle_info.granularity,
                       handle_info.device_id);
            }
        }
    }

    // hash to ModelConfig map
    std::unordered_map<uint64_t, std::unique_ptr<ModelConfig>> model_config_map_; 
    mutable std::mutex map_mutex_;
    int32_t total_models_{0};
    std::unique_ptr<ModelMacroRegistry> macro_registry_;
    std::vector<int32_t> current_model_npuids_{};

};

} // namespace mdaemon