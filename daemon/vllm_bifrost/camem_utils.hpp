#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <assert.h>
#include <stdio.h>

extern "C" {
#include "acl/acl.h"
}

// meta function: get from env; otherwise default string
inline std::string getEnvOrDefault(const char* env_name, const std::string& default_value) {
    const char* env_value = std::getenv(env_name);
    if (env_value && env_value[0] != '\0') {
        return std::string(env_value);
    }
    return default_value;
}

inline int getEnvOrDefault(const char* env_name, int default_value) {
    const char* env_value = std::getenv(env_name);
    if (env_value && env_value[0] != '\0') {
        try {
            return std::stoi(env_value);
        } catch (const std::exception& e) {
            std::cerr << "Warning: invalid integer in env " << env_name
                      << "='" << env_value << "', using default " << default_value << "\n";
        }
    }
    return default_value;
}

// ------------------------------------------------------------------------------
// Constants:

inline const char* getShmPrefix() {
    static const std::string value = getEnvOrDefault("MDAEMON_SHM_PREFIX", std::string("/vllm-model-"));
    return value.c_str();
}

inline const char* getModelSemaphoreName() {
    static const std::string value =
        getEnvOrDefault("MDAEMON_SEMAPHORE_NAME", std::string("/vllm-model-semaphore"));
    return value.c_str();
}

inline const char* getModelRegShm() {
    static const std::string value =
        getEnvOrDefault("MDAEMON_MODEL_REG_SHM", std::string("/vllm-model-registry"));
    return value.c_str();
}

#define SHM_PREFIX getShmPrefix()
#define MAX_HANDLES_PER_MODEL 30720
                                    // Maximum number of handles per model. assuming 
                                    // each handle is at least 2MB, this allows up to 
                                    // 60GB per model upon all devices.
                                    // [TODO(shijin)]: This is currently for single-npu service,
                                    // need to reconsider for multi-npu service.
#define MODEL_SEMAPHORE_NAME getModelSemaphoreName()
#define MODEL_REG_SHM getModelRegShm()

constexpr size_t kModelShmNameLength = 256;
constexpr int32_t kInvalidModelHandle = std::numeric_limits<int32_t>::max();

// ------------------------------------------------------------------------------


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
