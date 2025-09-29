#include "connection_pool.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>

// ===== Log Helper Function =====
static void log_info(const std::string& msg) {
    std::cout << "[ConnectionPool] " << msg << std::endl;
}
static void log_warn(const std::string& msg) {
    std::cerr << "[ConnectionPool][WARN] " << msg << std::endl;
}
static void log_error(const std::string& msg) {
    std::cerr << "[ConnectionPool][ERROR] " << msg << std::endl;
}

// ===== socket helper =====
static int setNonBlocking(int fd, bool nb) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (nb) flags |= O_NONBLOCK; else flags &= ~O_NONBLOCK;
    return fcntl(fd, F_SETFL, flags);
}

ConnectionPool& ConnectionPool::getInstance() {
    static ConnectionPool inst;
    return inst;
}

ConnectionPool::~ConnectionPool() {
    shutdown();
}

ConnectionPool::ConnectOptions ConnectionPool::ConnectOptions::fromEnv() {
    auto getenv_i = [](const char* k, int defv)->int{
        if (const char* v = ::getenv(k)) { try { return std::stoi(v); } catch (...) {} }
        return defv;
    };
    auto getenv_b = [&](const char* k, bool defv)->bool{
        if (const char* v = ::getenv(k)) {
            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            if (s == "1" || s=="true" || s=="on") return true;
            if (s == "0" || s=="false"|| s=="off") return false;
        }
        return defv;
    };

    ConnectOptions o;
    o.connect_timeout_ms       = getenv_i("D_AGENT_CONNECT_TIMEOUT_MS", o.connect_timeout_ms);
    o.handshake_timeout_ms     = getenv_i("D_AGENT_HANDSHAKE_TIMEOUT_MS", o.handshake_timeout_ms);
    o.max_retries              = getenv_i("D_AGENT_MAX_RETRIES", o.max_retries);
    o.retry_backoff_ms         = getenv_i("D_AGENT_RETRY_BACKOFF_MS", o.retry_backoff_ms);
    o.max_backoff_ms           = getenv_i("D_AGENT_MAX_BACKOFF_MS", o.max_backoff_ms);
    o.overall_wait_ms          = getenv_i("D_AGENT_OVERALL_WAIT_MS", o.overall_wait_ms);
    o.allow_partial            = getenv_b("D_AGENT_ALLOW_PARTIAL", o.allow_partial);
    o.background_reconnect     = getenv_b("D_AGENT_BG_RECONNECT", o.background_reconnect);
    o.health_check_interval_ms = getenv_i("D_AGENT_HEALTH_CHECK_INTERVAL_MS", o.health_check_interval_ms);
    o.conns_per_node           = getenv_i("KVC_CONNS_PER_NODE", o.conns_per_node);
    return o;
}


bool ConnectionPool::parseNodeSpecs(const std::string& node_specs) {
    nodes_.clear();
    size_t start = 0;
    while (start < node_specs.size()) {
        size_t semi = node_specs.find(';', start);
        std::string token = node_specs.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if (!token.empty()) {
            // host:port:node_id
            size_t c1 = token.find(':');
            size_t c2 = token.find(':', c1 == std::string::npos ? 0 : c1 + 1);
            if (c1 != std::string::npos && c2 != std::string::npos) {
                NodeSpec ns;
                ns.host = token.substr(0, c1);
                ns.port = std::stoi(token.substr(c1 + 1, c2 - c1 - 1));
                ns.node_id = std::stoi(token.substr(c2 + 1));
                nodes_.push_back(ns);
            } else {
                log_error("Bad node spec: " + token);
            }
        }
        if (semi == std::string::npos) break;
        start = semi + 1;
    }
    return !nodes_.empty();
}

bool ConnectionPool::parseNodeSpecs(const std::vector<std::string>& tokens) {
    nodes_.clear();
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        size_t c1 = token.find(':');
        size_t c2 = token.find(':', c1 == std::string::npos ? 0 : c1 + 1);
        if (c1 != std::string::npos && c2 != std::string::npos) {
            NodeSpec ns;
            ns.host = token.substr(0, c1);
            ns.port = std::stoi(token.substr(c1 + 1, c2 - c1 - 1));
            ns.node_id = std::stoi(token.substr(c2 + 1));
            nodes_.push_back(ns);
        } else {
            log_error("Bad node spec: " + token);
        }
    }
    return !nodes_.empty();
}

void ConnectionPool::applySocketOptions(int fd) const {
    int yes = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
}

int ConnectionPool::connectWithTimeout(const std::string& host, int port, int timeout_ms) {
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    int rv;
    std::string portstr = std::to_string(port);
    if ((rv = ::getaddrinfo(host.c_str(), portstr.c_str(), &hints, &res)) != 0) {
        log_error("getaddrinfo(" + host + ":" + portstr + ") failed: " + gai_strerror(rv));
        return -1;
    }
    int fd = -1;
    for (addrinfo* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        setNonBlocking(fd, true);
        int rc = ::connect(fd, p->ai_addr, p->ai_addrlen);
        if (rc == 0) {
            setNonBlocking(fd, false);
            applySocketOptions(fd);
            ::freeaddrinfo(res);
            log_info("Connected to " + host + ":" + portstr);
            return fd;
        }
        if (rc < 0 && errno == EINPROGRESS) {
            pollfd pf{}; pf.fd = fd; pf.events = POLLOUT;
            int pr = ::poll(&pf, 1, timeout_ms);
            if (pr == 1 && (pf.revents & POLLOUT)) {
                int err = 0; socklen_t len = sizeof(err);
                if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0) {
                    setNonBlocking(fd, false);
                    applySocketOptions(fd);
                    ::freeaddrinfo(res);
                    log_info("Connected to " + host + ":" + portstr + " (after poll)");
                    return fd;
                }
            }
        }
        ::close(fd); fd = -1;
    }
    ::freeaddrinfo(res);
    log_warn("Timeout connecting to " + host + ":" + portstr);
    return -1;
}

int ConnectionPool::connectWithRetry(const std::string& host, int port, const ConnectOptions& opts) {
    int backoff = opts.retry_backoff_ms;
    for (int attempt = 0; attempt < opts.max_retries; ++attempt) {
        int fd = connectWithTimeout(host, port, opts.connect_timeout_ms);
        if (fd >= 0) return fd;
        log_warn("connect to " + host + ":" + std::to_string(port) +
                 " failed (attempt " + std::to_string(attempt+1) + "/" + std::to_string(opts.max_retries) +
                 "), retry after " + std::to_string(backoff) + "ms");
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
        backoff = std::min(opts.max_backoff_ms, backoff * 2);
    }
    log_error("connect to " + host + ":" + std::to_string(port) + " failed after max retries");
    return -1;
}

bool ConnectionPool::ensureNodePool(int node_id, int desired_conns) {
    // Find NodeSpec
    auto itns = std::find_if(nodes_.begin(), nodes_.end(), [&](const NodeSpec& n){ return n.node_id == node_id; });
    if (itns == nodes_.end()) return false;

    NodeConnPool& pool = pools_[node_id]; // default-construct in-place
    std::unique_lock<std::mutex> lk(pool.mu);
    pool.desired = desired_conns;
    // Create missing connections
    while ((int)pool.all.size() < desired_conns) {
        lk.unlock();
        int fd = connectWithRetry(itns->host, itns->port, last_opts_);
        lk.lock();
        if (fd >= 0) {
            pool.all.push_back(fd);
            pool.idle.push_back(fd);
        } else {
            // Break to avoid tight loop; background recon will keep trying
            break;
        }
    }
    return !pool.all.empty();
}

int ConnectionPool::createOneConnUnlocked(const NodeSpec& ns, int timeout_ms) {
    int fd = connectWithTimeout(ns.host, ns.port, timeout_ms);
    if (fd >= 0) applySocketOptions(fd);
    return fd;
}

bool ConnectionPool::initialize(const std::string& node_specs, const ConnectOptions& opts) {
    std::lock_guard<std::mutex> lk(mu_);
    sockets_.clear();
    pools_.clear();
    stop_ = false;
    last_opts_ = opts;
    if (!parseNodeSpecs(node_specs)) {
        log_error("parse node_specs failed");
        return false;
    }

    // Create multi-conn pools per node (best effort)
    size_t ok_nodes = 0;
    for (const auto& ns : nodes_) {
        // directly create
        // Default-construct in-place inside the map, then fill by reference to eliminate move/copy of NodeConnPool.
        NodeConnPool& pool = pools_[ns.node_id];
        pool.desired = opts.conns_per_node;
        // Try to open desired connections
        for (int i = 0; i < opts.conns_per_node; ++i) {
            int fd = connectWithRetry(ns.host, ns.port, opts);
            if (fd >= 0) {
                pool.all.push_back(fd);
                pool.idle.push_back(fd);
                if (i == 0) sockets_[ns.node_id] = fd; // legacy single fd
            } else {
                break;
            }
        }
        if (!pool.all.empty()) ok_nodes++;
    }

    bool all_ready = (ok_nodes == nodes_.size());
    if (!all_ready) {
        log_warn("connected nodes with at least one fd: " + std::to_string(ok_nodes) + "/" + std::to_string(nodes_.size()) +
                 (opts.allow_partial ? " (allow_partial)" : " (need all)"));
    } else {
        log_info("All nodes connected (multi-conn).");
    }

    if (opts.background_reconnect) {
        try {
            bg_thread_ = std::thread([this, opts]{ backgroundReconnectLoop(opts); });
            log_info("Background reconnect thread started.");
        } catch (...) {
            log_error("failed to start background reconnect thread");
        }
    }

    return all_ready || opts.allow_partial;
}

bool ConnectionPool::initialize(const std::vector<std::string>& node_specs_vec,
                                const ConnectOptions& opts) {
    std::lock_guard<std::mutex> lk(mu_);
    sockets_.clear();
    pools_.clear();
    stop_ = false;
    last_opts_ = opts;
    if (!parseNodeSpecs(node_specs_vec)) {
        log_error("parse node_specs_vec failed");
        return false;
    }

    size_t ok_nodes = 0;
    for (const auto& ns : nodes_) {
        NodeConnPool& pool = pools_[ns.node_id]; // in-place default
        pool.desired = opts.conns_per_node;
        for (int i = 0; i < opts.conns_per_node; ++i) {
            int fd = connectWithRetry(ns.host, ns.port, opts);
            if (fd >= 0) {
                pool.all.push_back(fd);
                pool.idle.push_back(fd);
                if (i == 0) sockets_[ns.node_id] = fd;
            } else {
                break;
            }
        }
        if (!pool.all.empty()) ok_nodes++;
    }

    bool all_ready = (ok_nodes == nodes_.size());
    if (!all_ready) {
        log_warn("connected nodes with at least one fd: " + std::to_string(ok_nodes) + "/" + std::to_string(nodes_.size()) +
                 (opts.allow_partial ? " (allow_partial)" : " (need all)"));
    } else {
        log_info("All nodes connected (multi-conn).");
    }

    if (opts.background_reconnect) {
        try {
            bg_thread_ = std::thread([this, opts]{ backgroundReconnectLoop(opts); });
            log_info("Background reconnect thread started.");
        } catch (...) {
            log_error("failed to start background reconnect thread");
        }
    }
    return all_ready || opts.allow_partial;
}

void ConnectionPool::backgroundReconnectLoop(ConnectOptions opts) {
    while (!stop_.load(std::memory_order_relaxed)) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& ns : nodes_) {
                // Legacy single-fd reconnect
                if (sockets_.find(ns.node_id) == sockets_.end()) {
                    int fd = connectWithTimeout(ns.host, ns.port, opts.connect_timeout_ms);
                    if (fd >= 0) {
                        sockets_[ns.node_id] = fd;
                        log_info("Background (legacy) reconnected node " + std::to_string(ns.node_id));
                    }
                }
                // Multi-conn pool top-up
                auto it = pools_.find(ns.node_id);
                if (it != pools_.end()) {
                    NodeConnPool& pool = it->second;
                    std::unique_lock<std::mutex> lkpool(pool.mu);
                    while ((int)pool.all.size() < pool.desired) {
                        lkpool.unlock();
                        int fd = connectWithTimeout(ns.host, ns.port, opts.connect_timeout_ms);
                        lkpool.lock();
                        if (fd >= 0) {
                            pool.all.push_back(fd);
                            pool.idle.push_back(fd);
                            pool.cv.notify_one();
                            log_info("Background added conn for node " + std::to_string(ns.node_id) +
                                     " size=" + std::to_string(pool.all.size()));
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(opts.health_check_interval_ms));
    }
}

size_t ConnectionPool::expectedCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nodes_.size();
}

size_t ConnectionPool::readyCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    // nodes with at least one conn
    size_t ready = 0;
    for (auto& ns : nodes_) {
        auto it = pools_.find(ns.node_id);
        if (it != pools_.end() && !it->second.all.empty()) ready++;
    }
    return ready;
}

bool ConnectionPool::isReady() const {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& ns : nodes_) {
        auto it = pools_.find(ns.node_id);
        if (it == pools_.end() || it->second.all.empty()) return false;
    }
    return !nodes_.empty();
}

bool ConnectionPool::isNodeReady(int node_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pools_.find(node_id);
    if (it == pools_.end()) return false;
    return !it->second.all.empty();
}

bool ConnectionPool::isConnected(int node_id) const {
    return isNodeReady(node_id);
}

std::vector<int> ConnectionPool::getAllNodeIds() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int> ids;
    ids.reserve(nodes_.size());
    for (const auto& ns : nodes_) ids.push_back(ns.node_id);
    return ids;
}

std::optional<int> ConnectionPool::getSocketFd(int node_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = sockets_.find(node_id);
    if (it == sockets_.end()) return std::nullopt;
    return it->second;
}

int ConnectionPool::getConnection(int node_id) {
    // Legacy: just pick or create one
    auto fdopt = getSocketFd(node_id);
    if (fdopt.has_value()) return *fdopt;
    // try create one
    auto it = std::find_if(nodes_.begin(), nodes_.end(), [&](const NodeSpec& n){ return n.node_id == node_id; });
    if (it == nodes_.end()) return -1;
    int fd = connectWithTimeout(it->host, it->port, last_opts_.connect_timeout_ms);
    if (fd >= 0) {
        std::lock_guard<std::mutex> lk(mu_);
        sockets_[node_id] = fd;
    }
    return fd;
}

int ConnectionPool::acquireConnection(int node_id, int timeout_ms) {
    NodeConnPool* pool = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pools_.find(node_id);
        if (it == pools_.end()) return -1;
        pool = &it->second;
    }
    std::unique_lock<std::mutex> lk(pool->mu);
    if (timeout_ms < 0) {
        pool->cv.wait(lk, [&]{ return !pool->idle.empty(); });
    } else {
        if (!pool->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]{ return !pool->idle.empty(); })) {
            return -1;
        }
    }
    int fd = pool->idle.front();
    pool->idle.pop_front();
    return fd;
}

void ConnectionPool::releaseConnection(int node_id, int fd, bool healthy) {
    // Find NodeSpec
    NodeSpec ns{};
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto itns = std::find_if(nodes_.begin(), nodes_.end(), [&](const NodeSpec& n){ return n.node_id == node_id; });
        if (itns == nodes_.end()) { ::close(fd); return; }
        ns = *itns;
    }

    NodeConnPool* pool = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pools_.find(node_id);
        if (it == pools_.end()) { ::close(fd); return; }
        pool = &it->second;
    }

    std::unique_lock<std::mutex> lk(pool->mu);
    if (!healthy) {
        // remove fd from all[]
        auto itf = std::find(pool->all.begin(), pool->all.end(), fd);
        if (itf != pool->all.end()) pool->all.erase(itf);
        lk.unlock();
        ::close(fd);
        lk.lock();
        // try to recreate to keep size
        if ((int)pool->all.size() < pool->desired) {
            lk.unlock();
            int nfd = connectWithTimeout(ns.host, ns.port, last_opts_.connect_timeout_ms);
            lk.lock();
            if (nfd >= 0) {
                pool->all.push_back(nfd);
                pool->idle.push_back(nfd);
                pool->cv.notify_one();
            }
        }
    } else {
        // return to idle
        pool->idle.push_back(fd);
        pool->cv.notify_one();
    }
}

bool ConnectionPool::reconnect(int node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto itns = std::find_if(nodes_.begin(), nodes_.end(), [&](const NodeSpec& n){ return n.node_id == node_id; });
    if (itns == nodes_.end()) return false;
    // close legacy single
    auto it = sockets_.find(node_id);
    if (it != sockets_.end()) { ::close(it->second); sockets_.erase(it); }
    int fd = connectWithTimeout(itns->host, itns->port, last_opts_.connect_timeout_ms);
    if (fd < 0) return false;
    sockets_[node_id] = fd;
    return true;
}

void ConnectionPool::markDisconnected(int node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = sockets_.find(node_id);
    if (it != sockets_.end()) {
        ::close(it->second);
        sockets_.erase(it);
        log_warn("markDisconnected: node " + std::to_string(node_id));
    }
}

void ConnectionPool::shutdown() {
    stop_.store(true, std::memory_order_relaxed);
    if (bg_thread_.joinable()) {
        try { bg_thread_.join(); } catch (...) {}
    }
    // Close multi-conn pools
    for (auto& kv : pools_) {
        NodeConnPool& pool = kv.second;
        std::lock_guard<std::mutex> lk(pool.mu);
        for (int fd : pool.all) { ::close(fd); }
        pool.all.clear();
        pool.idle.clear();
    }
    pools_.clear();

    // Close legacy singles
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : sockets_) {
        ::close(kv.second);
        log_info("shutdown: closed node " + std::to_string(kv.first));
    }
    sockets_.clear();
    nodes_.shrink_to_fit();
    log_info("ConnectionPool shutdown complete");
}

void ConnectionPool::closeAll() {
    shutdown();
}