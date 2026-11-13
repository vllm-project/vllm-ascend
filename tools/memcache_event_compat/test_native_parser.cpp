#include <cassert>
#include <iostream>
#include <optional>
#include <string>
#include <cstdint>
namespace ock::mmc::kv_event {
std::optional<std::string> ExtractBlockHashFromObjectKey(const std::string&);
uint64_t BlockHashHexToU64(const std::string&);
}
int main(){
 const std::string h="3bb61b5f4f104508bfce20107f10324ef835fff2a6bf10015b8fcea4db906503";
 using ock::mmc::kv_event::ExtractBlockHashFromObjectKey;
 assert(ExtractBlockHashFromObjectKey("model@0@"+h+"@0")==h);
 assert(ExtractBlockHashFromObjectKey("model@"+h+"@3")==h);
 assert(ExtractBlockHashFromObjectKey("model@group:0@layer_id:3@"+h)==h);
 assert(ExtractBlockHashFromObjectKey("dspark-prefix-v1@identity@pp0@tp0@"+h)==h);
 assert(!ExtractBlockHashFromObjectKey("model@partial@request@0@0@127@0"));
 std::cout<<"native parser assertions passed; converted_hash="<<ock::mmc::kv_event::BlockHashHexToU64(h)<<std::endl;
}
