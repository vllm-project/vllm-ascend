#include <cstdint>
#include <optional>
#include <string>
namespace ock::mmc::kv_event {
std::optional<std::string> ExtractBlockHashFromObjectKey(const std::string&);
uint64_t BlockHashHexToU64(const std::string&);
}
extern "C" uint64_t check_hash(const char* key) {
    auto hash = ock::mmc::kv_event::ExtractBlockHashFromObjectKey(std::string(key));
    return hash ? ock::mmc::kv_event::BlockHashHexToU64(*hash) : UINT64_MAX;
}
