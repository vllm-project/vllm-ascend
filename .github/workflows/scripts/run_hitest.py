from __future__ import annotations

import os
import subprocess
import sys


def run_hitest_script() -> tuple[int, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shell_script_path = os.path.join(script_dir, "hitest.sh")

    if not os.path.exists(shell_script_path):
        raise FileNotFoundError(f"hitest.sh not found at: {shell_script_path}")
    os.chmod(shell_script_path, 0o755)
    
    env = os.environ.copy()
    
    # 在 run_hitest_script 函数 env=os.environ.copy() 后加
    print("DEBUG X_APIG_APPCODE:", env.get("X_APIG_APPCODE"))
    print("DEBUG APP_KEY:", env.get("APP_KEY"))
    print("DEBUG APP_SECRET:", env.get("APP_SECRET"))
    env["X_APIG_APPCODE"] = os.environ.get("X_APIG_APPCODE", "")
    env["APP_KEY"] = os.environ.get("APP_KEY", "")
    env["APP_SECRET"] = os.environ.get("APP_SECRET", "")

    proc = subprocess.run(
        ["/bin/bash", "-el", shell_script_path],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    try:
        ret_code, out, err = run_hitest_script()
    except FileNotFoundError as e:
        print(f"::error::{str(e)}")
        sys.exit(99)

    print("===== hitest.sh stdout =====")
    print(out)
    print("===== hitest.sh stderr =====")
    print(err)

    if ret_code != 0:
        print(f"::error::hitest.sh exited with non-zero code: {ret_code}")
        sys.exit(ret_code)

    print("hitest.sh execute success")
    sys.exit(0)


if __name__ == "__main__":
    main()
