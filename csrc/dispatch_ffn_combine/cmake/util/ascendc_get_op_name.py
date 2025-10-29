#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import configparser


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini-file", help="op info ini.")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    op_config = configparser.ConfigParser()
    op_config.read(args.ini_file)
    for section in op_config.sections():
        print(section, end="-")
        print(op_config.get(section, "opFile.value"), end="\n")
