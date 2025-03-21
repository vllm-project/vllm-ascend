#
# Adapted from https://github.com/vllm-project/vllm/blob/main/collect_env.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple

from vllm.envs import environment_variables

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    'SystemEnv',
    [
        'torch_version',
        'is_debug_build',
        'gcc_version',
        'clang_version',
        'cmake_version',
        'os',
        'libc_version',
        'python_version',
        'python_platform',
        'pip_version',  # 'pip' or 'pip3'
        'pip_packages',
        'conda_packages',
        'cpu_info',
        'vllm_version',  # vllm specific field
        'vllm_ascend_version',  # vllm ascend specific field
        'env_vars',
        'npu_info',  # ascend specific field
        'cann_info',  # ascend specific field
    ])

DEFAULT_CONDA_PATTERNS = {
    "torch",
    "numpy",
    "soumith",
    "mkl",
    "magma",
    "triton",
    "optree",
    "transformers",
    "zmq",
    "pynvml",
}

DEFAULT_PIP_PATTERNS = {
    "torch",
    "numpy",
    "mypy",
    "flake8",
    "triton",
    "optree",
    "onnx",
    "transformers",
    "zmq",
    "pynvml",
}


def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = True if type(command) is str else False
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=shell)
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == 'win32':
        enc = 'oem'
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split('\n')[0]


def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_CONDA_PATTERNS
    conda = os.environ.get('CONDA_EXE', 'conda')
    out = run_and_read_all(run_lambda, "{} list".format(conda))
    if out is None:
        return out

    return "\n".join(line for line in out.splitlines()
                     if not line.startswith("#") and any(name in line
                                                         for name in patterns))


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'gcc --version', r'gcc (.*)')


def get_clang_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'clang --version',
                                     r'clang version (.*)')


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version',
                                     r'cmake (.*)')


def _parse_version(version, version_tuple):
    version_str = version_tuple[-1]
    if version_str.startswith('g'):
        if '.' in version_str:
            git_sha = version_str.split('.')[0][1:]
            date = version_str.split('.')[-1][1:]
            return f"{version} (git sha: {git_sha}, date: {date})"
        else:
            git_sha = version_str[1:]  # type: ignore
            return f"{version} (git sha: {git_sha})"
    return version


def get_vllm_version():
    from vllm import __version__, __version_tuple__
    return _parse_version(__version__, __version_tuple__)


def get_vllm_ascend_version():
    from vllm_ascend._version import __version__, __version_tuple__
    return _parse_version(__version__, __version_tuple__)


def get_cpu_info(run_lambda):
    rc, out, err = 0, '', ''
    if get_platform() == 'linux':
        rc, out, err = run_lambda('lscpu')
    elif get_platform() == 'win32':
        rc, out, err = run_lambda(
            'wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID, \
        CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE'
        )
    elif get_platform() == 'darwin':
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = 'None'
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info


def get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('win32'):
        return 'win32'
    elif sys.platform.startswith('cygwin'):
        return 'cygwin'
    elif sys.platform.startswith('darwin'):
        return 'darwin'
    else:
        return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'sw_vers -productVersion',
                                     r'(.*)')


def get_windows_version(run_lambda):
    system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
    wmic_cmd = os.path.join(system_root, 'System32', 'Wbem', 'wmic')
    findstr_cmd = os.path.join(system_root, 'System32', 'findstr')
    return run_and_read_all(
        run_lambda,
        '{} os get Caption | {} /v Caption'.format(wmic_cmd, findstr_cmd))


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'lsb_release -a',
                                     r'Description:\t(.*)')


def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cat /etc/*-release',
                                     r'PRETTY_NAME="(.*)"')


def get_os(run_lambda):
    from platform import machine
    platform = get_platform()

    if platform == 'win32' or platform == 'cygwin':
        return get_windows_version(run_lambda)

    if platform == 'darwin':
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return 'macOS {} ({})'.format(version, machine())

    if platform == 'linux':
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())

        return '{} ({})'.format(platform, machine())

    # Unknown platform
    return platform


def get_python_platform():
    import platform
    return platform.platform()


def get_libc_version():
    import platform
    if get_platform() != 'linux':
        return 'N/A'
    return '-'.join(platform.libc_ver())


def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    # People generally have `pip` as `pip` or `pip3`
    # But here it is invoked as `python -mpip`
    def run_with_pip(pip):
        out = run_and_read_all(run_lambda, pip + ["list", "--format=freeze"])
        return "\n".join(line for line in out.splitlines()
                         if any(name in line for name in patterns))

    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'
    out = run_with_pip([sys.executable, '-mpip'])

    return pip_version, out


def get_npu_info(run_lambda):
    return run_and_read_all(run_lambda, 'npu-smi info')


def get_cann_info(run_lambda):
    out = run_and_read_all(run_lambda, 'lscpu | grep Architecture:')
    cpu_arch = str(out).split()[-1]
    return run_and_read_all(
        run_lambda,
        'cat /usr/local/Ascend/ascend-toolkit/latest/{}-linux/ascend_toolkit_install.info'
        .format(cpu_arch))


def get_env_vars():
    env_vars = ''
    secret_terms = ('secret', 'token', 'api', 'access', 'password')
    report_prefix = ("TORCH", "PYTORCH", "ASCEND_", "ATB_")
    for k, v in os.environ.items():
        if any(term in k.lower() for term in secret_terms):
            continue
        if k in environment_variables:
            env_vars = env_vars + "{}={}".format(k, v) + "\n"
        if k.startswith(report_prefix):
            env_vars = env_vars + "{}={}".format(k, v) + "\n"

    return env_vars


def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
    else:
        version_str = debug_mode_str = 'N/A'

    sys_version = sys.version.replace("\n", " ")

    conda_packages = get_conda_packages(run_lambda)

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version='{} ({}-bit runtime)'.format(
            sys_version,
            sys.maxsize.bit_length() + 1),
        python_platform=get_python_platform(),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=conda_packages,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        cpu_info=get_cpu_info(run_lambda),
        vllm_version=get_vllm_version(),
        vllm_ascend_version=get_vllm_ascend_version(),
        env_vars=get_env_vars(),
        npu_info=get_npu_info(run_lambda),
        cann_info=get_cann_info(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}

CPU:
{cpu_info}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()

# both the above code and the following code use `strip()` to
# remove leading/trailing whitespaces, so we need to add a newline
# in between to separate the two sections
env_info_fmt += "\n"

env_info_fmt += """
vLLM Version: {vllm_version}
vLLM Ascend Version: {vllm_ascend_version}

ENV Variables:
{env_vars}

NPU:
{npu_info}

CANN:
{cann_info}
""".strip()


def pretty_str(envinfo):

    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string

    mutable_dict = envinfo._asdict()

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict['pip_packages'] = replace_if_empty(
        mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(
        mutable_dict['conda_packages'])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict['pip_packages']:
        mutable_dict['pip_packages'] = prepend(
            mutable_dict['pip_packages'], '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        mutable_dict['conda_packages'] = prepend(
            mutable_dict['conda_packages'], '[conda] ')
    mutable_dict['cpu_info'] = envinfo.cpu_info
    mutable_dict['npu_info'] = envinfo.npu_info
    mutable_dict['cann_info'] = envinfo.cann_info
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if TORCH_AVAILABLE and hasattr(torch, 'utils') and hasattr(
            torch.utils, '_crash_handler'):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump)
                for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                '%Y-%m-%d %H:%M:%S')
            msg = "\n*** Detected a minidump at {} created on {}, ".format(latest, creation_time) + \
                  "if this is related to your bug please include it when you file a report ***"
            print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
