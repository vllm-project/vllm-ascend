#pragma once
#include "common.h"
#include <future>

class KVTransferClient {
public:
    static KVTransferClient& getInstance();
    ~KVTransferClient();

    bool requestKVTransfer(const KVRequest& request);

    void shutdown();

    // Interface preserved (no-op for now)
    bool startWorkers();
    void stopWorkers();
    bool workersEnabled() const;

private:
    KVTransferClient() = default;
    KVTransferClient(const KVTransferClient&) = delete;
    KVTransferClient& operator=(const KVTransferClient&) = delete;
};