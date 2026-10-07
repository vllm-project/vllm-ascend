#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys

import pandas as pd


def list_pt_from_excel(excel_path, sheet_name, pt_dir):
    if not os.path.exists(excel_path):
        print(f"ERROR: Excel file not found: {excel_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(pt_dir):
        print(f"ERROR: pt directory not found: {pt_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "Testcase_Name" not in df.columns:
        print(f"ERROR: Column 'Testcase_Name' not found in sheet '{sheet_name}'", file=sys.stderr)
        sys.exit(1)

    pt_files = []
    missing = []
    for case_name in df["Testcase_Name"]:
        pt_path = os.path.join(pt_dir, f"{case_name}.pt")
        if os.path.isfile(pt_path):
            pt_files.append(pt_path)
        else:
            missing.append(case_name)

    if missing:
        print(f"WARNING: {len(missing)} cases have no matching .pt file: {missing}", file=sys.stderr)

    if not pt_files:
        print("ERROR: No matching .pt files found for any case in Excel", file=sys.stderr)
        sys.exit(1)

    print(",".join(pt_files))


def main():
    parser = argparse.ArgumentParser(description="Extract Testcase_Name from Excel and map to .pt files in pt_dir")
    parser.add_argument("excel_path", type=str, help="Path to Excel file")
    parser.add_argument("pt_dir", type=str, help="Directory containing .pt files")
    parser.add_argument("--sheet", "-S", type=str, default="Sheet1", help="Sheet name (default: Sheet1)")
    args = parser.parse_args()

    list_pt_from_excel(args.excel_path, args.sheet, args.pt_dir)


if __name__ == "__main__":
    main()
