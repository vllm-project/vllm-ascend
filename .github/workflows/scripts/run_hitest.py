from __future__ import annotations

import os
import subprocess
import sys


def run_hitest_script() -> tuple[int, str, str]:
    """
    执行同级目录下 hitest.sh 脚本
    :return: (return_code, stdout, stderr)
    """
    # 获取当前脚本所在目录，拼接 hitest.sh 绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shell_script_path = os.path.join(script_dir, "hitest.sh")

    # 校验脚本文件存在
    if not os.path.exists(shell_script_path):
        raise FileNotFoundError(f"hitest.sh not found at: {shell_script_path}")
    # 增加可执行权限（CI容器内可能缺少x权限）
    os.chmod(shell_script_path, 0o755)

    # 执行shell脚本，捕获输出
    proc = subprocess.run(
        ["/bin/bash", "-el", shell_script_path],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    """入口函数，供CI流水线调用"""
    try:
        ret_code, out, err = run_hitest_script()
    except FileNotFoundError as e:
        print(f"::error::{str(e)}")
        sys.exit(99)

    # 打印完整日志
    print("===== hitest.sh stdout =====")
    print(out)
    print("===== hitest.sh stderr =====")
    print(err)

    # 脚本非0退出码则CI失败
    if ret_code != 0:
        print(f"::error::hitest.sh exited with non-zero code: {ret_code}")
        sys.exit(ret_code)

    print("hitest.sh execute success")
    sys.exit(0)


if __name__ == "__main__":
    main()
