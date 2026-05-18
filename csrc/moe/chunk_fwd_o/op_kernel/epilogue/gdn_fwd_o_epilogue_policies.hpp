/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_GDN_FWD_O_EPILOGUE_POLICIES_HPP
#define CATLASS_EPILOGUE_GDN_FWD_O_EPILOGUE_POLICIES_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue {

struct EpilogueAtlasGDNFwdOQkmask {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
};

struct EpilogueAtlasGDNFwdOOutput {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
};

}  // namespace Catlass::Epilogue

#endif  // CATLASS_EPILOGUE_GDN_FWD_O_EPILOGUE_POLICIES_HPP
