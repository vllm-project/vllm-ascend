// Memcache 1.3.0 event parser compatibility. Keys remain unchanged.
#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
namespace ock::mmc::kv_event {
std::optional<std::string> ExtractBlockHashFromObjectKey(const std::string& key) {
    if (key.find("@partial@") != std::string::npos) return std::nullopt;
    const auto last = key.rfind('@');
    if (last != std::string::npos && last > 0) {
        const auto previous = key.rfind('@', last - 1);
        if (previous != std::string::npos) {
            const auto hash = key.substr(previous + 1, last - previous - 1);
            const auto rank = key.substr(last + 1);
            if (hash.size() == 64 && !rank.empty() &&
                std::all_of(hash.begin(), hash.end(), [](unsigned char c) { return std::isxdigit(c); }) &&
                std::all_of(rank.begin(), rank.end(), [](unsigned char c) { return std::isdigit(c); })) return hash;
        }
    }
    // Do not use RTLD_NEXT: Python loads Memcache with RTLD_LOCAL, so the
    // original symbol is not necessarily visible in this library's scope.
    auto hash = last == std::string::npos ? key : key.substr(last + 1);
    if (hash.empty() || !std::all_of(hash.begin(), hash.end(),
            [](unsigned char c) { return std::isxdigit(c); })) return std::nullopt;
    return hash;
}
}
