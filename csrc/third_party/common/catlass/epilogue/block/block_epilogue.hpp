/**
 * Copyright (c) 2025 Tianjin University, Ltd.
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

#include "catlass/epilogue/block/block_epilogue_elemwise_no_source.hpp"
#include "catlass/epilogue/block/block_epilogue_elemwise_one_source.hpp"
#include "catlass/epilogue/block/block_epilogue_fa_softmax.hpp"
#include "catlass/epilogue/block/block_epilogue_fa_rescale_o.hpp"
#include "catlass/epilogue/block/block_epilogue_mla_softmax.hpp"
#include "catlass/epilogue/block/block_epilogue_mla_rescale_o.hpp"
#include "catlass/epilogue/block/block_epilogue_mla_fd_rescale_o.hpp"
#include "catlass/epilogue/block/block_epilogue_per_token_dequant.hpp"
#include "catlass/epilogue/block/block_epilogue_gemm.hpp"
#include "catlass/epilogue/block/block_epilogue_gemv.hpp"
#include "catlass/epilogue/block/block_epilogue_mla_tp1_softmax.hpp"
#include "catlass/epilogue/block/block_epilogue_mla_tp1_rescale_o.hpp"
#include "catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp"
#include "catlass/epilogue/block/block_epilogue_rescale_o_no_split_row.hpp"
#include "catlass/epilogue/block/block_epilogue_w4a4_per_token_per_channel_dequant.hpp"
#include "catlass/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
#include "catlass/epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
#include "catlass/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "catlass/epilogue/block/block_epilogue_gdn_fwdo_output.hpp"

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
