import os
import sys
import subprocess


def run_setup():
    # 强制设置 Python 环境为 UTF-8 模式
    os.environ["PYTHONUTF8"] = "1"

    py_exe = sys.executable
    root_dir = os.path.dirname(os.path.abspath(__file__))
    req_txt = os.path.join(root_dir, "requirements(RAPID-RTX).txt")
    marker = os.path.join(root_dir, ".deps_installed")

    if os.path.exists(marker):
        print("[RAPID-RTX] Python Dependencies already installed.")
        return

    if os.path.exists(req_txt):
        print(f"[RAPID-RTX] Reading requirements from {req_txt}...")
        # 1. 手动读取 requirements.txt，指定 utf-8 编码，并过滤掉空行和注释
        with open(req_txt, "r", encoding="utf-8") as f:
            install_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not install_list:
            print("[RAPID-RTX] No packages found in requirements.txt.")
            return

        print(f"[RAPID-RTX] Packages to install: {install_list}")

        # 2. 直接将列表作为参数传给 pip，不再使用 -r 参数，彻底避开编码问题
        cmd = [py_exe, "-s", "-m", "pip", "install"] + install_list
        subprocess.check_call(cmd)

        # 3. 写入哨兵文件
        with open(marker, "w") as f:
            f.write("OK")
        print("[RAPID-RTX] Successfully installed all libraries.")

    else:
        print(f"[RAPID-RTX] ERROR: {req_txt} not found!")
        sys.exit(1)


if __name__ == "__main__":
    run_setup()