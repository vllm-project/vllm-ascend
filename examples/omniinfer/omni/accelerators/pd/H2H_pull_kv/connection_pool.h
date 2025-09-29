#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <optional>
#include <deque>
#include <condition_variable>

struct NodeSpec {
    std::string host;
    int port{0};
    int node_id{0};
};

class ConnectionPool {
public:
    static ConnectionPool& getInstance();

    struct ConnectOptions {
        int connect_timeout_ms = 3000;       // single connect timeout
        int handshake_timeout_ms = 3000;     // reserved for future handshake
        int max_retries = 1000;                // max retries per node during init
        int retry_backoff_ms = 200;          // initial backoff
        int max_backoff_ms = 2000;           // max backoff
        int overall_wait_ms = 300000;         // overall wait if allow_partial=false
        bool allow_partial = false;           // return success when some nodes are ready
        bool background_reconnect = true;    // keep reconnecting in background
        int health_check_interval_ms = 2000; // background loop interval
        int conns_per_node = 16;             // number of persistent connections per P node

        static ConnectOptions fromEnv();
    };

    // Initialize with "host:port:node_id;..." string
    bool initialize(const std::string& node_specs, const ConnectOptions& opts);

    // Initialize with ["host:port:node_id", ...]
    bool initialize(const std::vector<std::string>& node_specs_vec,
                    const ConnectOptions& opts);

    // Queries
    size_t expectedCount() const;
    size_t readyCount() const;
    bool isReady() const;                 // all nodes have at least one connection
    bool isNodeReady(int node_id) const;
    bool isConnected(int node_id) const;  // alias for compatibility

    // Node ids in configured order
    std::vector<int> getAllNodeIds() const;

    // Legacy single-conn helpers (kept for compatibility)
    std::optional<int> getSocketFd(int node_id) const;
    int getConnection(int node_id); // may create a new single conn lazily

    // NEW: multi-connection pooling API
    // Acquire a dedicated connection for node_id (wait up to timeout_ms, <0 to wait forever).
    // Returns fd >= 0 on success, -1 on timeout or failure.
    int acquireConnection(int node_id, int timeout_ms = -1);

    // Release a connection back to pool. If healthy=false, the fd is closed and replaced.
    void releaseConnection(int node_id, int fd, bool healthy);

    // Try reconnecting a specific node now (only affects legacy single conn)
    bool reconnect(int node_id);

    // Mark a legacy single connection disconnected
    void markDisconnected(int node_id);

    // Close everything and stop background thread
    void shutdown();
    void closeAll(); // alias

private:
    ConnectionPool() = default;
    ~ConnectionPool();

    ConnectionPool(const ConnectionPool&) = delete;
    ConnectionPool& operator=(const ConnectionPool&) = delete;

    bool parseNodeSpecs(const std::string& node_specs);
    bool parseNodeSpecs(const std::vector<std::string>& tokens);
    int connectWithRetry(const std::string& host, int port, const ConnectOptions& opts);
    int connectWithTimeout(const std::string& host, int port, int timeout_ms);
    void applySocketOptions(int fd) const;
    void backgroundReconnectLoop(ConnectOptions opts);

    // Helpers for multi-conn pool
    bool ensureNodePool(int node_id, int desired_conns);
    int createOneConnUnlocked(const NodeSpec& ns, int timeout_ms);

    struct NodeConnPool {
        std::vector<int> all;     // all open fds
        std::deque<int> idle;     // idle fds available to acquire
        std::mutex mu;
        std::condition_variable cv;
        int desired{0};           // target number of connections
    };

    mutable std::mutex mu_;
    std::vector<NodeSpec> nodes_;
    std::unordered_map<int,int> sockets_; // legacy: node_id -> single fd

    // Multi-conn pools per node
    std::unordered_map<int, NodeConnPool> pools_;

    std::atomic<bool> stop_{false};
    std::thread bg_thread_;
    ConnectOptions last_opts_{};
};