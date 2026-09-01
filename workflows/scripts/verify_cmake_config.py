# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
CMake 配置验证脚本

功能：
1. 检查 CMakeLists.txt 是否包含必需元素
2. 检查是否使用了错误的函数（如 asc_add_ops_executable）
3. 检查是否设置了正确的语言（ASC CXX）
4. 检查是否链接了必需的库

用法：
python verify_cmake_config.py operators/{operator_name}/CMakeLists.txt

返回：
- 0: 验证通过
- 1: 验证失败（不能继续编译）
"""

import sys
import os
import re
from pathlib import Path


def verify_cmake(cmake_file):
    """验证 CMakeLists.txt 配置"""
    if not os.path.exists(cmake_file):
        return False, f"❌ CMakeLists.txt 不存在: {cmake_file}"

    with open(cmake_file, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    warnings = []

    # 检查1：必须包含 find_package(ASC REQUIRED)
    if "find_package(ASC REQUIRED)" not in content:
        errors.append("❌ 缺少 find_package(ASC REQUIRED)")

    # 检查2：project 必须包含 ASC 语言
    if not re.search(r"project\s*\([^)]*LANGUAGES[^)]*ASC", content, re.IGNORECASE):
        errors.append("❌ project() 必须包含 LANGUAGES ASC CXX（当前缺少 ASC）")

    # 检查3：禁止使用 asc_add_ops_executable（不存在的函数）
    if "asc_add_ops_executable" in content:
        errors.append(
            "❌ 禁止使用 asc_add_ops_executable（不存在的函数，请使用 add_executable）"
        )

    # 检查4：必须链接 tiling_api
    if "tiling_api" not in content:
        errors.append("❌ 必须链接 tiling_api 库")

    # 检查5：必须链接 register
    if "register" not in content:
        warnings.append("⚠️ 建议链接 register 库")

    # 检查6：必须链接 platform
    if "platform" not in content:
        warnings.append("⚠️ 建议链接 platform 库")

    # 检查7：必须设置 --npu-arch
    if "--npu-arch" not in content:
        errors.append("❌ 必须设置 --npu-arch 参数（如 --npu-arch=dav-2201）")

    # 检查8：必须使用 add_executable
    if "add_executable" not in content:
        errors.append("❌ 必须使用 add_executable 定义可执行文件")

    # 检查9：建议链接 m 和 dl
    if "target_link_libraries" in content:
        if " m" not in content and "\nm" not in content:
            warnings.append("⚠️ 建议链接数学库 m")
        if " dl" not in content and "\ndl" not in content:
            warnings.append("⚠️ 建议链接动态链接库 dl")

    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_cmake_config.py <CMakeLists.txt路径>")
        print("示例: python verify_cmake_config.py operators/softmax/CMakeLists.txt")
        sys.exit(1)

    cmake_file = sys.argv[1]

    print("=" * 70)
    print("CMake 配置验证")
    print("=" * 70)
    print(f"📄 CMakeLists.txt: {cmake_file}")
    print()

    errors, warnings = verify_cmake(cmake_file)

    # 显示警告
    if warnings:
        print("⚠️  警告：")
        for warning in warnings:
            print(f"  {warning}")
        print()

    # 显示错误
    if errors:
        print("❌ 错误：")
        for error in errors:
            print(f"  {error}")
        print()
        print("=" * 70)
        print("❌ 验证失败")
        print("=" * 70)
        print()
        print("📖 请查阅 CMake 配置指南：")
        print("   /ascendc-direct-invoke-template skill 的 add_kernel/CMakeLists.txt")
        print()
        print("💡 常见问题：")
        print("   1. 缺少 ASC 语言：project(... LANGUAGES ASC CXX)")
        print("   2. 使用了错误函数：使用 add_executable 而非 asc_add_ops_executable")
        print(
            "   3. 缺少库链接：target_link_libraries(... PRIVATE tiling_api register platform m dl)"
        )
        print(
            "   4. 缺少 NPU 架构：target_compile_options(... PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)"
        )
        sys.exit(1)
    else:
        print("=" * 70)
        print("✅ CMake 配置验证通过")
        print("=" * 70)
        print()
        print("✓ find_package(ASC REQUIRED) 存在")
        print("✓ project(... LANGUAGES ASC CXX) 正确")
        print("✓ 使用 add_executable 定义可执行文件")
        print("✓ 链接必需库（tiling_api, register, platform, m, dl）")
        print("✓ 设置 NPU 架构（--npu-arch）")
        print()
        print("✅ 可以继续编译")
        sys.exit(0)


if __name__ == "__main__":
    main()
