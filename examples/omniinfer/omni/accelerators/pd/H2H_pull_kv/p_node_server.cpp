#include "common.h"
#include "kv_mem_resolver.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop.store(true); }
static void set_reuseaddr(int fd) { int yes = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)); }

static void set_nodelay(int fd) {
#ifdef TCP_NODELAY
    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
#endif
}

static int getenv_int(const char* name, int defv) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defv;
    try { return std::stoi(v); } catch (...) { return defv; }
}

static void set_keepalive(int fd) {
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
#ifdef TCP_KEEPIDLE
    int idle = getenv_int("P_TCP_KEEPIDLE", 60);   // Idle seconds before first probe
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
    int intvl = getenv_int("P_TCP_KEEPINTVL", 10); // Probe interval seconds
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#ifdef TCP_KEEPCNT
    int cnt = getenv_int("P_TCP_KEEPCNT", 5);      // Consecutive failure count
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
}

static std::string get_peer(int sock) {
    sockaddr_in addr{}; socklen_t len = sizeof(addr);
    if (getpeername(sock, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
        char ip[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
        return std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
    }
    return "unknown";
}

static bool recv_header_and_body(int sock, MessageHeader& hdr, std::vector<uint8_t>& body) {
    if (!recv_all(sock, &hdr, sizeof(hdr))) return false;
    if (hdr.magic != PROTOCOL_MAGIC || hdr.type != MessageType::REQUEST) {
        std::cerr << "[P] Invalid request header (magic/type)\n";
        return false;
    }
    body.resize(hdr.body_length);
    if (hdr.body_length && !recv_all(sock, body.data(), hdr.body_length)) {
        std::cerr << "[P] Failed to receive request body, errno=" << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

static bool send_response_with_data(int sock,
                                    uint32_t resp_node_id_in_header,
                                    const std::string& request_id,
                                    const std::vector<uint64_t>& p_block_ids) {
    TensorMetadata meta{};
    std::vector<kvmem::BlockView> views;
    views.reserve(p_block_ids.size());
    size_t total_bytes = 0;
    const uint32_t local_mem_index = 0;

    for (uint64_t pb : p_block_ids) {
        kvmem::BlockView bv{};
        if (!kvmem::p_get_block(local_mem_index, pb, bv) || !bv.ptr || !bv.bytes) {
            std::cerr << "[P] Missing P block: pb=" << pb << std::endl;
            return false;
        }
        total_bytes += bv.bytes;
        views.push_back(bv);
    }
    meta.total_size = total_bytes;

    std::vector<uint8_t> resp_body =
        serialize_response(request_id, resp_node_id_in_header, ErrorCode::SUCCESS, p_block_ids, meta);

    MessageHeader rh{};
    rh.magic = PROTOCOL_MAGIC;
    rh.version = PROTOCOL_VERSION;
    rh.type = MessageType::RESPONSE;
    rh.body_length = static_cast<uint32_t>(resp_body.size());
    rh.node_id = resp_node_id_in_header;

    if (!send_all(sock, &rh, sizeof(rh))) return false;
    if (!resp_body.empty() && !send_all(sock, resp_body.data(), resp_body.size())) return false;

    if (total_bytes) {
        std::vector<iovec> iov(views.size());
        for (size_t i = 0; i < views.size(); ++i) { iov[i].iov_base = views[i].ptr; iov[i].iov_len = views[i].bytes; }
        if (!writev_all(sock, iov.data(), static_cast<int>(iov.size()), total_bytes)) return false;
    }
    return true;
}

// Per-client-connection session threads (no RCV timeout, blocking read, keepalive maintains long-lived connection)
static void handle_client_session(uint32_t server_node_id, int client_sock, uint64_t client_id, const std::string& peer) {
    set_nodelay(client_sock);
    set_keepalive(client_sock);

    while (!g_stop.load()) {
        MessageHeader req_hdr{};
        std::vector<uint8_t> req_body;
        if (!recv_header_and_body(client_sock, req_hdr, req_body)) break;

        std::string request_id;
        uint32_t req_node_from_body = 0;
        std::vector<uint64_t> p_block_ids;
        if (!parse_request(req_body, request_id, req_node_from_body, p_block_ids)) {
            std::cerr << "[P " << server_node_id << "][client #" << client_id << " peer=" << peer
                      << "] Failed to parse request body" << std::endl;
            break;
        }

        uint32_t req_node_from_hdr = req_hdr.node_id;

        std::cout << "[P " << server_node_id << "][client #" << client_id << " peer=" << peer
                  << "] <- request id='" << request_id << "' blocks=" << p_block_ids.size()
                  << " hdr.node_id=" << req_node_from_hdr
                  << " body.node_id=" << req_node_from_body
                  << std::endl;

        if (!send_response_with_data(client_sock, req_node_from_hdr, request_id, p_block_ids)) {
            std::cerr << "[P " << server_node_id << "][client #" << client_id << " peer=" << peer
                      << "] send_response_with_data failed" << std::endl;
            break;
        }
        std::cout << "[P " << server_node_id << "][client #" << client_id << " peer=" << peer
                  << "] -> response OK id='" << request_id << "' blocks=" << p_block_ids.size() << std::endl;
    }
    close(client_sock);
    std::cout << "[P " << server_node_id << "][client #" << client_id << " peer=" << peer << "] disconnected" << std::endl;
}

static kvmem::ScalarType parse_dtype(const std::string& s) {
    std::string t = s;
    for (auto& c: t) c = std::tolower(c);
    if (t == "bf16") return kvmem::ScalarType::BF16;
    if (t == "f16" ) return kvmem::ScalarType::F16;
    if (t == "f32" ) return kvmem::ScalarType::F32;
    if (t == "f64" ) return kvmem::ScalarType::F64;
    if (t == "i8"  ) return kvmem::ScalarType::I8;
    if (t == "i16" ) return kvmem::ScalarType::I16;
    if (t == "i32" ) return kvmem::ScalarType::I32;
    if (t == "i64" ) return kvmem::ScalarType::I64;
    return kvmem::ScalarType::BF16;
}

int main(int argc, char** argv) {
    uint32_t node_id = 0;
    uint16_t port = 15000;

    std::string mmap_path;
    std::string dtype_str = "bf16";
    size_t offset_dtype = 0;
    size_t start_offset = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--node_id" && i + 1 < argc) node_id = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoul(argv[++i]));
        else if (a == "--mmap" && i + 1 < argc) mmap_path = argv[++i];
        else if (a == "--dtype" && i + 1 < argc) dtype_str = argv[++i];
        else if (a == "--offset_dtype" && i + 1 < argc) offset_dtype = static_cast<size_t>(std::stoull(argv[++i]));
        else if (a == "--start_offset" && i + 1 < argc) start_offset = static_cast<size_t>(std::stoull(argv[++i]));
    }

    if (mmap_path.empty() || offset_dtype == 0) {
        std::cerr << "[P " << node_id << "] Missing required args: --mmap <path> --offset_dtype <N> [--dtype bf16] [--start_offset 0]" << std::endl;
        return 1;
    }

    auto dtype = parse_dtype(dtype_str);
    std::cout << "[P " << node_id << "] Start on 0.0.0.0:" << port
              << " mmap=" << mmap_path << " dtype=" << dtype_str
              << " offset_dtype=" << offset_dtype
              << " start_offset=" << start_offset << std::endl;

    if (!kvmem::init_p_side_from_mmap(mmap_path, dtype, offset_dtype, "127.0.0.1", port, start_offset)) {
        std::cerr << "[P " << node_id << "] init_p_side_from_mmap failed" << std::endl;
        return 1;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { std::cerr << "[P] socket failed: " << std::strerror(errno) << std::endl; return 1; }
    set_reuseaddr(listen_fd);

    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons(port);
    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[P] bind(" << port << ") failed: " << std::strerror(errno) << std::endl; ::close(listen_fd); return 1;
    }
    if (::listen(listen_fd, 512) != 0) {
        std::cerr << "[P] listen failed: " << std::strerror(errno) << std::endl; ::close(listen_fd); return 1;
    }
    std::cout << "[P " << node_id << "] Listening on 0.0.0.0:" << port << std::endl;

    static std::atomic<uint64_t> g_client_seq{0};

    while (!g_stop.load()) {
        sockaddr_in cli{}; socklen_t cl = sizeof(cli);
        int client = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&cli), &cl);
        if (client < 0) {
            if (errno == EINTR) continue;
            if (g_stop.load()) break;
            std::cerr << "[P] accept failed: " << std::strerror(errno) << std::endl;
            continue;
        }
        set_nodelay(client);
        set_keepalive(client);

        uint64_t cid = ++g_client_seq;
        std::string peer = get_peer(client);
        std::cout << "[P " << node_id << "] accepted client #" << cid << " peer=" << peer << std::endl;

        std::thread th(handle_client_session, node_id, client, cid, peer);
        th.detach();
    }

    ::close(listen_fd);
    kvmem::shutdown_all();
    std::cout << "[P " << node_id << "] Cleaning up..." << std::endl;
    return 0;
}