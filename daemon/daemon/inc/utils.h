// source: kvcached(https://github.com/ovg-project/kvcached)

#ifndef __SSHAREUTILS_H__
#define __SSHAREUTILS_H__

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>
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

inline bool daemonDebugLogsEnabled() {
    static const bool enabled = []() {
        const char* flag = std::getenv("MDAEMON_DEBUG_LOG");
        return flag && std::string_view(flag) == "1";
    }();
    return enabled;
}

inline size_t daemonLogBufferCapacity() {
    const int cap = getEnvOrDefault("MDAEMON_LOG_BUFFER_SIZE", 2000);
    return static_cast<size_t>(cap > 0 ? cap : 2000);
}

inline std::mutex& daemonLogMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::deque<std::string>& daemonLogBuffer() {
    static std::deque<std::string> buffer;
    return buffer;
}

inline std::string daemonLogNow() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tmv, "%H:%M:%S");
    return oss.str();
}

inline void daemonLogPushLine(const std::string& line) {
    std::lock_guard<std::mutex> guard(daemonLogMutex());
    auto& buffer = daemonLogBuffer();
    buffer.push_back(line);
    const size_t cap = daemonLogBufferCapacity();
    while (buffer.size() > cap) {
        buffer.pop_front();
    }
}

inline void daemonLogError(const std::string& message) {
    const std::string line = daemonLogNow() + " [ERROR] " + message;
    daemonLogPushLine(line);
    std::cerr << "\033[31m" << line << "\033[0m" << std::endl;
}

inline void daemonLogInfo(const std::string& message) {
    const std::string line = daemonLogNow() + " [INFO ] " + message;
    daemonLogPushLine(line);
    if (daemonDebugLogsEnabled()) {
        std::cout << line << std::endl;
    }
}

inline void daemonLogDebug(const std::string& message) {
    if (!daemonDebugLogsEnabled()) {
        return;
    }
    const std::string line = daemonLogNow() + " [DEBUG] " + message;
    daemonLogPushLine(line);
    std::cout << line << std::endl;
}

inline std::vector<std::string> daemonDrainLogs() {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> guard(daemonLogMutex());
    auto& buffer = daemonLogBuffer();
    out.assign(buffer.begin(), buffer.end());
    buffer.clear();
    return out;
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

// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------
// Helper functions:
inline void showError(aclError error_code, const std::string& file, int line) {
    if (error_code != 0) {
        std::string message = "acl Error, code: " + std::to_string(error_code) +
                              " at " + file + ":" + std::to_string(line);
        daemonLogError(message);
        throw std::runtime_error(message);
    }
}

#define CHECK_ERROR(error_code) showError(error_code, __FILE__, __LINE__)

inline void ensure_context(int32_t device) {
    CHECK_ERROR(aclrtSetDevice(device));
    aclrtContext pctx;
    CHECK_ERROR(aclrtGetCurrentContext(&pctx));
    if (!pctx) {
        // Ensure device context.
        CHECK_ERROR(aclrtCreateContext(&pctx, device));
        CHECK_ERROR(aclrtSetCurrentContext(pctx));
    }
}
// ------------------------------------------------------------------------------

#endif