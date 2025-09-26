# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import logging
import os
import platform
import shutil
import sys
import stat
import tempfile
from multiprocessing import cpu_count
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

logging.basicConfig(level=logging.INFO)

build_root_dir = 'build/lib.' + platform.system().lower() + '-' + platform.machine() + '-' + str(
    sys.version_info.major) + '.' + str(sys.version_info.minor)

logging.info('%s', build_root_dir)

extensions = []
ignore_folders = ['build', 'test', 'tests']
conf_folders = ['conf']


def get_root_path(root):
    if os.path.dirname(root) in ['', '.']:
        return os.path.basename(root)
    else:
        return get_root_path(os.path.dirname(root))


def copy_file(src, dest):
    if os.path.exists(dest):
        return
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    if os.path.isdir(src):
        shutil.copytree(src, dest)
    else:
        shutil.copyfile(src, dest)


def touch_init_file():
    init_file_name = os.path.join(tempfile.mkdtemp(), '__init__.py')
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(init_file_name, flags, modes), 'w'):
        pass
    return init_file_name


init_file = touch_init_file()
logging.info('%s', init_file)


def compose_extensions(root='.'):
    for file_ in os.listdir(root):
        abs_file = os.path.join(root, file_)
        if os.path.isfile(abs_file):
            if abs_file.endswith('.py') and "setup.py" not in abs_file:
                extensions.append(Extension(get_root_path(abs_file) + '.*', [abs_file]))
            elif abs_file.endswith('.c') or abs_file.endswith('.pyc'):
                continue
            else:
                copy_file(abs_file, os.path.join(build_root_dir, abs_file))
            if abs_file.endswith('__init__.py'):
                copy_file(init_file, os.path.join(build_root_dir, abs_file))
        else:
            if os.path.basename(abs_file) in ignore_folders:
                continue
            if os.path.basename(abs_file) in conf_folders:
                copy_file(abs_file, os.path.join(build_root_dir, abs_file))
            compose_extensions(abs_file)


compose_extensions()
os.remove(init_file)

setup(
    name='optiquant',
    version='0.0.1',
    ext_modules=cythonize(
        extensions,
        nthreads=cpu_count(),
        compiler_directives=dict(always_allow_keywords=True),
        include_path=[]),
    cmdclass=dict(build_ext=build_ext)
)