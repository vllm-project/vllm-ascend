import random
import datetime
import numpy as np
import os
import sys
import torch

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# 计算相对误差
def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

# 日志打印函数
def print_log(data=None, level='INFO'):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S"), level, data))


# 计算绝对误差
def cal_abs_diff(real_data, expect_data, diff_thd):
    return list(map(lambda x, y: cal_relative_diff(x, y, diff_thd), real_data, expect_data))

def display_output_tf(real_data, compare_data, expect_data, start, end, diff_thd):
    print_log('---------------------------------------------------------------------------------------')
    print_log('Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(NPU-FP16)  RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], compare_data[j], real_data[j],
                abs(np.float64(compare_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(compare_data[j], real_data[j], diff_thd)))
# 打印误差结果
def display_output(real_data, expect_data, start, end, diff_thd):
    print_log('---------------------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
    else:
        for i in range(10):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            j = i + start
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))


# 打印精度失败结果
def display_error_output_mp(real_data, expect_data, relative_diff, start, diff_thd):
    print_log('Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    count = 0
    for i in range(len(relative_diff)):
        j = i + start
        if relative_diff[j] > diff_thd:
            count += 1
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                start + i + 1, expect_data[j], real_data[j], abs(np.float64(expect_data[j]) - np.float64(real_data[j])),
                cal_relative_diff(expect_data[j], real_data[j], diff_thd)))
        if count == 100:
            break
    print_log('---------------------------------------------------------------------------------------')

def display_error_output(real_data, expect_data, err_idx, relative_diff, start, end, diff_thd):
    print_log('Error Line-----------------------------------------------------------------------------')
    # print_log('Error ids-----------------------------------------------------------------------------')
    # for i in err_idx:
        # print(i, end = ' ')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log('---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if len_err <= 20 or count < 10 or count > len_err - 10:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10:
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s  \t %07s ' % (dot_3, dot_3, dot_3, dot_3, dot_3, dot_3))
    print_log('---------------------------------------------------------------------------------------')



# 对比精度结果
def data_compare(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1):
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        print_log(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' % (float(split_count), max_diff_hd))
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32),
                                 diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0
    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"
    print_log('---------------------------------------------------------------------------------------')
    print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print_log('---------------------------------------------------------------------------------------')
    print_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' % (diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Maximum error is: %s. Tolerance threshold is: %s.' % (max_error, max_diff_hd))
    # if result == "Failed":
    display_error_output(real_data, data_compe, err_idx, err_diff, start, end, diff_thd)
    return result, fulfill_percent, max_error, err_idx

def format_convert(file_name, to_dtype):
    view_data = np.fromfile(os.path.join(WORKSPACE, "data", file_name), dtype=np.int16)
    return torch.from_numpy(view_data).view(to_dtype).to(torch.float).numpy()

if __name__ == "__main__":
    data_type = str(sys.argv[1])
    if data_type == "half" or data_type == "fp16" or data_type == "float16":
        to_dtype = torch.float16
    elif data_type == "bf16" or data_type == "bfloat16":
        to_dtype = torch.bfloat16
    v_npu = format_convert("o_npu.bin", to_dtype)
    v_ref = format_convert("o_ref.bin", to_dtype)
    data_compare(v_npu, v_ref, 0.0001, 0.00001)