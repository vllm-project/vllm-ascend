#!/usr/bin/env python3
"""
精度调试 - 边界值测试数据生成工具

生成各种边界值测试数据，用于验证算子的精度和鲁棒性。
"""

import numpy as np
import sys
import argparse


def generate_boundary_cases(shape, dtype, output_dir="."):
    """
    生成边界值测试数据

    Args:
        shape: 数据形状 (M, N, K) 或 (M, N)
        dtype: 数据类型 ('fp16', 'fp32', 'int8')
        output_dir: 输出目录
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    # 边界值定义
    if dtype == 'fp16':
        boundary_values = {
            "zero": 0.0,
            "tiny": 1e-4,  # FP16 最小正常数
            "small": 1e-3,
            "normal": 1.0,
            "large": 100.0,
            "saturation": 65504.0,  # FP16 最大值
            "negative": -1.0,
            "neg_saturation": -65504.0,
        }
    elif dtype == 'fp32':
        boundary_values = {
            "zero": 0.0,
            "tiny": 1e-10,
            "small": 1e-6,
            "normal": 1.0,
            "large": 1e6,
            "huge": 1e10,
            "negative": -1.0,
        }
    else:  # int8
        boundary_values = {
            "zero": 0,
            "min": -128,
            "max": 127,
            "normal": 42,
        }

    # 生成每个边界值的测试数据
    for name, value in boundary_values.items():
        data = np.full(shape, value, dtype=np_type)
        filename = f"{output_dir}/boundary_{name}_{dtype}.npy"
        np.save(filename, data)
        print(f"生成: {filename} (value={value})")


def generate_random_aligned(shape, dtype, output_dir=".", seed=42):
    """
    生成32字节对齐的随机测试数据

    Args:
        shape: 原始形状
        dtype: 数据类型
        output_dir: 输出目录
        seed: 随机种子
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    np.random.seed(seed)

    # 检查并调整对齐
    element_size = np.dtype(np_type).itemsize
    aligned_size = 32 // element_size

    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    # 生成随机数据
    data = np.random.rand(*adjusted_shape).astype(np_type)

    filename = f"{output_dir}/random_aligned_{'_'.join(map(str, shape))}_{dtype}.npy"
    np.save(filename, data)

    print(f"生成: {filename}")
    print(f"  原始形状: {shape}")
    print(f"  调整形状: {tuple(adjusted_shape)} (32字节对齐)")
    print(f"  数据范围: [{data.min():.6f}, {data.max():.6f}]")


def generate_unaligned(shape, dtype, output_dir=".", seed=42):
    """
    生成非对齐的随机测试数据

    Args:
        shape: 原始形状
        dtype: 数据类型
        output_dir: 输出目录
        seed: 随机种子
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    np.random.seed(seed)

    # 确保非对齐
    unaligned_shape = list(shape)
    unaligned_shape[-1] = shape[-1] + 1  # 加1破坏对齐

    data = np.random.rand(*unaligned_shape).astype(np_type)

    filename = f"{output_dir}/random_unaligned_{'_'.join(map(str, shape))}_{dtype}.npy"
    np.save(filename, data)

    print(f"生成: {filename}")
    print(f"  形状: {tuple(unaligned_shape)} (非对齐)")


def main():
    parser = argparse.ArgumentParser(description="生成精度调试测试数据")
    parser.add_argument("--shape", nargs="+", type=int, required=True,
                        help="数据形状，如: 8 16 16")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "int8"], default="fp32",
                        help="数据类型")
    parser.add_argument("--output", default=".",
                        help="输出目录")
    parser.add_argument("--type", choices=["boundary", "aligned", "unaligned", "all"],
                        default="all", help="生成的数据类型")

    args = parser.parse_args()

    shape = tuple(args.shape)

    if args.type in ["boundary", "all"]:
        print("\n【生成边界值数据】")
        generate_boundary_cases(shape, args.dtype, args.output)

    if args.type in ["aligned", "all"]:
        print("\n【生成对齐随机数据】")
        generate_random_aligned(shape, args.dtype, args.output)

    if args.type in ["unaligned", "all"]:
        print("\n【生成非对齐随机数据】")
        generate_unaligned(shape, args.dtype, args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("用法示例:")
        print("  python3 gen_boundary_test.py --shape 8 16 16 --dtype fp16")
        print("  python3 gen_boundary_test.py --shape 8 16 --dtype fp32 --type boundary")
        sys.exit(1)

    main()
