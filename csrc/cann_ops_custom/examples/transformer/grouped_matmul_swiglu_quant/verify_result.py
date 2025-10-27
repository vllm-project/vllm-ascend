import sys
import numpy as np

relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4

def verify_result(output, golden, dtype):
    output = np.fromfile(output, dtype).reshape(-1)
    golden = np.fromfile(golden, dtype).reshape(-1)
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=2**-13,
                                           atol=1,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    print("diff element count: ", len(different_element_indexes))
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

def main():
    print("============================================")
    print("Compare out1:")

    try:
        res1 = verify_result("data/out1Data.bin", "data/quant_output_np.bin", np.int8)
        if not res1:
            raise ValueError("[ERROR] out1 result error")
        else:
            print("out1 compare pass")
    except Exception as e:
        print(e)
        sys.exit(1)

    print("============================================")
    print("Compare out2:")

    try:
        res2 = verify_result("data/out2Data.bin", "data/quant_scale_output_np.bin", np.float32)
        if not res2:
            raise ValueError("[ERROR] out2 result error")
        else:
            print("out2 compare pass")
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
