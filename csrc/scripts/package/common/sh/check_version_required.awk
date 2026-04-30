# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function strip(input) {
    sub("^ +", "", input)
    sub(" +$", "", input)
    return input
}

function check_compatible_le(version_arr, len_version_arr, require_arr, len_require_arr,  i) {
    for (i = 1; i <= len_require_arr; i++) {
        if (require_arr[i] == "") {
            continue
        }
        # len_version_arr lt len_require_arr
        if (i > len_version_arr) {
            return 1
        }
        if (version_arr[i] < require_arr[i]) {
            return 1
        }
        if (version_arr[i] > require_arr[i]) {
            return 0
        }
    }
    if (len_version_arr > len_require_arr) {
        return 1
    }
    # len_version_arr eq len_require_arr
    return 1
}

function check_compatible_lt(version_arr, len_version_arr, require_arr, len_require_arr,  i) {
    for (i = 1; i <= len_require_arr; i++) {
        if (require_arr[i] == "") {
            continue
        }
        # len_version_arr lt len_require_arr
        if (i > len_version_arr) {
            return 1
        }
        if (version_arr[i] < require_arr[i]) {
            return 1
        }
        if (version_arr[i] > require_arr[i]) {
            return 0
        }
    }
    if (len_version_arr > len_require_arr) {
        return 0
    }
    # len_version_arr eq len_require_arr
    return 0
}

function check_compatible_ge(version_arr, len_version_arr, require_arr, len_require_arr,  i) {
    for (i = 1; i <= len_require_arr; i++) {
        if (require_arr[i] == "") {
            continue
        }
        # len_version_arr lt len_require_arr
        if (i > len_version_arr) {
            return 0
        }
        if (version_arr[i] < require_arr[i]) {
            return 0
        }
        if (version_arr[i] > require_arr[i]) {
            return 1
        }
    }
    if (len_version_arr > len_require_arr) {
        return 1
    }
    # len_version_arr eq len_require_arr
    return 1
}

function check_compatible_gt(version_arr, len_version_arr, require_arr, len_require_arr,  i) {
    for (i = 1; i <= len_require_arr; i++) {
        if (require_arr[i] == "") {
            continue
        }
        # len_version_arr lt len_require_arr
        if (i > len_version_arr) {
            return 0
        }
        if (version_arr[i] < require_arr[i]) {
            return 0
        }
        if (version_arr[i] > require_arr[i]) {
            return 1
        }
    }
    if (len_version_arr > len_require_arr) {
        return 1
    }
    # len_version_arr eq len_require_arr
    return 0
}

function check_compatible_eq(version_arr, len_version_arr, require_arr, len_require_arr,  i) {
    for (i = 1; i <= len_require_arr; i++) {
        if (require_arr[i] == "") {
            continue
        }
        # len_version_arr lt len_require_arr
        if (i > len_version_arr) {
            return 0
        }
        if (version_arr[i] != require_arr[i]) {
            return 0
        }
    }
    if (len_version_arr > len_require_arr) {
        return 1
    }
    # len_version_arr eq len_require_arr
    return 1
}

function check_compatible(version_arr, len_version_arr, require,  require_arr, len_require_arr, pos) {
    len_require_arr = split(require, require_arr, ".")

    pos = match(require_arr[1], /^>=/)
    if (pos != 0) {
        require_arr[1] = substr(require_arr[1], pos + RLENGTH)
        return check_compatible_ge(version_arr, len_version_arr, require_arr, len_require_arr)
    }

    pos = match(require_arr[1], /^>/)
    if (pos != 0) {
        require_arr[1] = substr(require_arr[1], pos + RLENGTH)
        return check_compatible_gt(version_arr, len_version_arr, require_arr, len_require_arr)
    }

    pos = match(require_arr[1], /^<=/)
    if (pos != 0) {
        require_arr[1] = substr(require_arr[1], pos + RLENGTH)
        return check_compatible_le(version_arr, len_version_arr, require_arr, len_require_arr)
    }

    pos = match(require_arr[1], /^</)
    if (pos != 0) {
        require_arr[1] = substr(require_arr[1], pos + RLENGTH)
        return check_compatible_lt(version_arr, len_version_arr, require_arr, len_require_arr)
    }

    return check_compatible_eq(version_arr, len_version_arr, require_arr, len_require_arr)
}

BEGIN {
    len_all_required_arr = split(all_required, all_required_arr, ",")

    compated = 0

    in_gt = 0
    matched_gt = 0

    len_version_arr = split(version, version_arr, ".")

    for (i = 1; i <= len_all_required_arr; i++) {
        all_required_arr[i] = strip(all_required_arr[i])
        one_compated = check_compatible(version_arr, len_version_arr, all_required_arr[i])

        pos = match(all_required_arr[i], /^>/)
        if (pos != 0) {
            gt_require = 1
            lt_require = 0
            eq_require = 0
        } else {
            pos = match(all_required_arr[i], /^</)
            if (pos != 0) {
                gt_require = 0
                lt_require = 1
                eq_require = 0
            } else {
                gt_require = 0
                lt_require = 0
                eq_require = 1
            }
        }

        if (matched_gt) {
            if (one_compated) {
                # gt after gt, all compated.
                if (gt_require) {
                    matched_gt = 1
                    in_gt = 1
                    continue
                }
                # lt after gt, all compated.
                if (lt_require) {
                    compated = 1
                    matched_gt = 0
                    in_gt = 0
                    break
                }
                # eq after gt, all compated.
                if (eq_require) {
                    # eq compated, go.
                    compated = 1
                    break
                }
            } else {
                # miscompated.
                # gt after gt, compated first gt. miscompated second gt.
                if (gt_require) {
                    matched_gt = 0
                    in_gt = 1
                    continue
                }
                # lt after gt, compated first gt. miscompated second lt.
                if (lt_require) {
                    matched_gt = 0
                    in_gt = 0
                    continue
                }
                # eq after gt, compated first gt. miscompated second eq.
                if (eq_require) {
                    continue
                }
            }
        } else {
            if (one_compated) {
                if (gt_require) {
                    matched_gt = 1
                    in_gt = 1
                    continue
                }
                if (lt_require) {
                    if (in_gt) {
                        matched_gt = 0
                        in_gt = 0
                        continue
                    } else {
                        compated = 1
                        break
                    }
                }
                if (eq_require) {
                    # eq compated, go.
                    compated = 1
                    break
                }
            } else {
                # miscompated.
                if (gt_require) {
                    matched_gt = 0
                    in_gt = 1
                    continue
                }
                if (lt_require) {
                    matched_gt = 0
                    in_gt = 0
                    continue
                }
                if (eq_require) {
                    continue
                }
            }
        }
    }

    if (matched_gt) {
        compated = 1
    }

    if (compated == 0) {
        printf("F")
    } else {
        printf("T")
    }
}
