# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved. 
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

"""
tfas算法超参调试工具
tfas_profiler.py - 分析prefill服务器日志并进行性能分析

该脚本用于解析prefill服务器的日志文件，提取关键性能指标，
并通过分段线性拟合法分析系统性能特性，确定组batch_tokens的最优值token_budget, 和对应的动态调整参数adjust_param。
 
 
使用方法：
    python tfas_profiler.py --log-file /path/to/prefill_server.log --output-path /path/to/your_output
    
必需参数：
    --log-file LOG_FILE 指定prefill服务器日志文件的路径
    --output-path ANALYSIS_OUTPUT_PATH 指定生成的分析绘图图片位置
    
脚本功能：
    1. 从日志中提取运行序列数，执行时间，token数和等待请求数
    2. 计算每个序列的输入长度
    3. 筛选出输入长度为1025的记录进行分析
    4. 使用IQR方法过滤异常值
    5. 对性能数据进行分段线性拟合
    6. 输出拟合结果和系统甜点值分析
    
输出格式：
    脚本会输出分段线性拟合的结果，包括：
    - token_budget
    - 动态调整参数 adjust_param
    - 分析曲线图.png
"""
    
def get_best_split(x, y):
    # 初始化
    min_error = float("inf")
    best_split = None 
    # 穷举所有可能的分割点（排除最前和最后一个点）
    for i in range(4, len(x) -1):
        x1, y1 = x[3:i], y[3:i]
        x2, y2 = x[i:], y[i:]
        # 拟合每一段
        model1 = LinearRegression().fit(x1, y1)
        model2 = LinearRegression().fit(x2, y2)
        # 预测并计算 MSE （均方误差）
        y1_pred = model1.predict(x1)
        y2_pred = model2.predict(x2)
        error = np.mean((y1 - y1_pred) ** 2) + np.mean((y2 - y2_pred) ** 2)
        if error < min_error:
            min_error = error
            best_split = i 
    return best_split 

def get_update_params(output_path, segments, models, x, y):
    # 计算动态调整参数
    intercept_set = []
    slope_set = []
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=1, label="Profiling data", alpha=0.5)
    for (x_seg, y_seg), model in zip(segments, models):
        x_pred = x_seg
        y_pred = model.predict(x_pred)
        
        # 获取最终拟合出的线性模型
        slope = model.coef_[0]
        intercept = model.intercept_
        intercept_set.append(intercept)
        slope_set.append(slope)
        plt.plot(
            x_pred, y_pred, linewidth=2, label="y=%.4f + %.4f x" % (intercept, slope)
        )
    plt.title("Piecewise Linear Fit with Outlier Removal")
    plt.xlabel("prefill batch size")
    plt.ylabel('prefill time (s)')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_path, "profiller.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    
    numerator = intercept_set[-1] - intercept_set[-2]
    denominator = slope_set[-2] - slope_set[-1]
    B_start = int(numerator / denominator * 1024) if denominator != 0 else float('inf')
    tfas_intercept = intercept_set[-1]
    tfas_slope = slope_set[-1]
    print("token_budget = ", B_start)
    print("adjust_param =", tfas_intercept/tfas_slope)
    
def IQR_filter(df):
    groups = df.groupby("seqs")
    
    result = {}
    for name, group in groups:
        timeseries = group.etime
        Q1 = timeseries.quantile(0.25)
        Q3 = timeseries.quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义正常范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        tf_filtered = timeseries[
            (timeseries >= lower_bound) & (timeseries <= upper_bound)
        ]
        result[name] = tf_filtered.mean()
    return result 

def profiler_1K_fixed(log_file_path, output_path):
    pattern1=r"current num reqs = (\d+), num_input_tokens = (\d+)"
    pattern2=r"execute model cost:([\d.]+)=([\d.+]+)"
    records_running_seqs = []
    records_execute_time = []
    records_tokens = []
    
    try:
        with open(log_file_path, "r", encoding="utf-8") as file:
            for line in file:
                match1=re.findall(pattern1, line)
                match2=re.findall(pattern2, line)
                if match1:
                    records_running_seqs.append(int(match1[0][0]))
                    records_tokens.append(int(match1[0][1]))
                elif match2:
                    records_execute_time.append(float(match2[0][1].split('+')[1]))
                else:
                    pass
    except FileNotFoundError as e:
        raise ValueError({e})
    except Exception as e:
        raise ValueError({e})
    
    # 获取profiler数据
    N = min(len(records_running_seqs), len(records_execute_time))
    df = pd.DataFrame()
    df["seqs"] = records_running_seqs[:N]
    df["etime"] = records_execute_time[:N]
    df["tokens"] = records_tokens[:N]
    df = df[df.seqs>0]
    df["input_len"] = df["tokens"] / df["seqs"]
 
    # 获取prefiler统计结果
    df_sel = df[df.input_len == 1025]
    my_result = IQR_filter(df_sel)
    
    # 计算初始甜点值
    x = np.array(list(my_result.keys()))
    y = np.array(list(my_result.values()))
    x = x.reshape(-1, 1)
    best_split = get_best_split(x, y)
    
    # 分段线性拟合
    x1, y1 = x[3:best_split], y[3:best_split]
    x2, y2 = x[best_split:], y[best_split:]
    model1 = LinearRegression().fit(x1, y1)
    model2 = LinearRegression().fit(x2, y2)
    models = [model1, model2]
    segments = [(x1, y1), (x2, y2)]
    get_update_params(output_path, segments, models, x, y)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True, 
                       help='prefill server file path')
    parser.add_argument('--output-path', type=str, default='',
                       help='analysis output file path')
    args = parser.parse_args()
    profiler_1K_fixed(args.log_file, args.output_path)