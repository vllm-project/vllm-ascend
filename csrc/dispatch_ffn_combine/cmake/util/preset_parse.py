#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import os
import sys


def read_json(file):
    with open(file, "r") as fd:
        config = json.load(fd)
    return config


def get_config_opts(file):
    config = read_json(file)

    src_dir = os.path.abspath(os.path.dirname(file))
    opts = ""

    for conf in config:
        if conf == "configurePresets":
            for node in config[conf]:
                macros = node.get("cacheVariables")
                if macros is not None:
                    for key in macros:
                        opts += "-D{}={} ".format(key, macros[key]["value"])

    opts = opts.replace("${sourceDir}", src_dir)
    print(opts)


if __name__ == "__main__":
    get_config_opts(sys.argv[1])
