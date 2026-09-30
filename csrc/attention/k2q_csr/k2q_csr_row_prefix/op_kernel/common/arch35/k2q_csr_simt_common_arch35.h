/**
 * SIMT 公共常量（对齐 CUDA CTA / SIMT best-practices）。
 */
#ifndef K2Q_CSR_SIMT_COMMON_ARCH35_H
#define K2Q_CSR_SIMT_COMMON_ARCH35_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace k2q_csr_simt {

/** 搬运/不规则访存：1024；LAUNCH_BOUND 与 Dim3 必须同常量 */
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 256;
#else
constexpr uint32_t THREAD_NUM = 1024;
#endif

} // namespace k2q_csr_simt

#endif
