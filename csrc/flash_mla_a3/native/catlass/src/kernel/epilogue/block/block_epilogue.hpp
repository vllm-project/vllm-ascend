/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Catlass::Epilogue::Block

#include "block_epilogue_mla_softmax.hpp"
#include "block_epilogue_mla_rescal_o.hpp"
#include "block_epilogue_mla_fd_rescal_o.hpp"
#include "block_epilogue_mla_tp1_softmax.hpp"
#include "block_epilogue_mla_tp1_rescal_o.hpp"
#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
