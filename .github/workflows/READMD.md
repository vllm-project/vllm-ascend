# E2E Test Workflow Guide

This document provides a guide on how to manage and extend the E2E test suite for `vllm-ascend`. It covers how to add new test cases and understand the automatic partitioning mechanism.

## 1. Adding a New Test Case

All E2E test cases are defined and managed in the `.github/workflows/scripts/config.yaml` file.

### Steps

1. **Prepare the Test Script**: Ensure your test script (`.py` file) is placed in the appropriate location under the `tests/e2e/` directory (e.g., `tests/e2e/singlecard/` or `tests/e2e/multicard/`).

2. **Modify `config.yaml`**:
    Open `.github/workflows/scripts/config.yaml` and locate the corresponding test suite (e.g., `e2e-singlecard` or `e2e-multicard-2-cards`).

3. **Add Configuration Entry**:
    Add a new entry under the corresponding list. Each entry contains the following fields:
    * `name`: The relative path to the test file. If you only need to run a specific test function within the file, use `::` as a separator, e.g., `path/to/test.py::test_func`.
    * `estimated_time`: The estimated time (in seconds) required to run the test. **This field is crucial** as it is used for automatic load balancing (partitioning).
    * `is_skipped` (Optional): If set to `true`, the test will be skipped.

### Example

Suppose you want to add a new test named `tests/e2e/singlecard/test_new_feature.py` with an estimated runtime of 120 seconds:

```yaml
suites:
  e2e-singlecard:
    # ... other existing tests ...
    - name: tests/e2e/singlecard/test_new_feature.py
      estimated_time: 120
```

To add a specific test function:

```yaml
    - name: tests/e2e/singlecard/test_new_feature.py::test_specific_case
      estimated_time: 60
```

## 2. Automatic Partitioning Mechanism

To speed up CI execution, we support splitting large test suites into multiple parallel Jobs (partitions). The partitioning logic is primarily implemented in the `auto_partition` function in `.github/workflows/scripts/run_suite.py`.

### Principle

The partitioning algorithm uses a Greedy Approach to achieve load balancing, aiming to make the total estimated runtime of each partition as equal as possible.

1. **Read Configuration**: The script reads all non-skipped test cases and their `estimated_time` from `config.yaml`.
2. **Sort(Balanced Assignment)**: Test cases are sorted by `estimated_time` in descending order. This ensures that the heaviest tasks are distributed first to achieve optimal load balancing across partitions.
3. **Assign**: Iterating through the sorted test cases, each case is assigned to the partition (Bucket) with the current minimum total time.
4. **Re-sort (Fast Feedback)**: Within each partition, tests are re-sorted by `estimated_time` in ascending order. This allows the CI to cover as many test cases as possible in the early stages.
    > TIP: If you need to prioritize a new test case, you can temporarily set its estimated_time to 0 to ensure it runs first, then update it to the actual value later.

### How to Modify Partitioning Logic

If you need to adjust the partitioning strategy, please modify the `.github/workflows/scripts/run_suite.py` file.

* **Algorithm Location**: `auto_partition` function.
* **Input Parameters**:
    * `files`: List of test files (including `estimated_time`).
    * `rank`: Index of the current partition (0 to size-1).
    * `size`: Total number of partitions.
* **Invocation**:
    CI workflows (e.g., `.github/workflows/_e2e_test.yaml`) call the script via command-line arguments:
    ```bash
    python3 .github/workflows/scripts/run_suite.py --suite <suite_name> --auto-partition-id <index> --auto-partition-size <total_count>
    ```

### Notes

* **Accurate Estimated Time**: To achieve the best load balancing, please provide an accurate `estimated_time` in `config.yaml`. If a new test is very time-consuming but the estimated time is set too low, it may cause a specific partition to timeout.
* **Number of Partitions**: The number of partitions (`auto-partition-size`) is typically defined in the `strategy.matrix` of the GitHub Actions workflow definition file (e.g., `_e2e_test.yaml`).

## 3. Running Tests Locally

You can use the `run_suite.py` script to run test suites locally:

```bash
# Run the full e2e-singlecard suite
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard

# Simulate partitioned execution (e.g., partition 0 of 2)
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard --auto-partition-id 0 --auto-partition-size 2
```
