"""ComfyUI 看门狗：进程挂了就自动拉起来，免得界面上只看到"连不上 8188"。

为什么需要它：ComfyUI 在 ROCm 上遇到显存分配失败时**不抛 Python 异常，直接 abort 进程**
（日志里是 `LLVM ERROR: Can't get available size` + `Fatal Python error: Aborted`）。
之后所有请求都是连接被拒。人肉重启太慢，这个脚本每分钟探一次，挂了就重新拉起。

    python comfy_watchdog.py                 # 常驻
    python comfy_watchdog.py --once          # 探一次，挂了就拉
    python comfy_watchdog.py --port 8188 --comfy-dir /path/to/ComfyUI --interval 60

几个必须注意的点（都是踩过的）：

* **探测用 `/system_stats`，2 秒超时**：慢不等于挂（可能正在加载 6 GB 的底模）。
* **杀进程用 `wmic process ... call terminate`**，不要用 `taskkill /F /PID`：
  Windows 会复用 PID，taskkill 有概率杀错进程。
* **重启前等 12 秒**：ComfyUI 刚崩的头几秒 GPU 上下文还没释放，立刻重启会再崩一次
  （表现为连续两次 `LLVM ERROR`）。
* **冷却 60 秒**：避免"启动失败 → 立刻再试"的死循环。
* **训练时不要开它**：它会把 ComfyUI 拉起来跟训练抢显存。训练脚本应该先把它停掉。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def log(msg: str, logfile: Path | None = None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def alive(base: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/system_stats", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    found = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            tail = line.split()
            if tail and tail[-1].isdigit():
                found.add(int(tail[-1]))
    return sorted(found)


def kill(pids: list[int], logfile: Path | None):
    for pid in pids:
        subprocess.run(["wmic", "process", "where", f"processid={pid}", "call", "terminate"],
                       capture_output=True, text=True, timeout=20)
        log(f"  已杀掉残留进程 {pid}", logfile)


def start(comfy_dir: Path, wait_s: int = 90, logfile: Path | None = None) -> bool:
    """启动 ComfyUI 并等它就绪。用 `python main.py`，不依赖特定启动脚本。"""
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--cache-none"]
    try:
        subprocess.Popen(cmd, cwd=str(comfy_dir),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"  启动失败：{e}", logfile)
        return False
    for _ in range(wait_s // 3):
        time.sleep(3)
        if alive("http://127.0.0.1:8188"):
            log("  ComfyUI 已就绪", logfile)
            return True
    log(f"  启动后 {wait_s} 秒仍未就绪", logfile)
    return False


def check_once(comfy_dir: Path, port: int, logfile: Path | None) -> str:
    base = f"http://127.0.0.1:{port}"
    if alive(base):
        return "alive"
    log("ComfyUI 无响应，准备重启", logfile)
    busy = pids_on_port(port)
    if busy:
        kill(busy, logfile)
        time.sleep(12)          # 等 GPU 上下文释放，否则会连撞两次 LLVM ERROR
    else:
        time.sleep(6)
    return "restarted" if start(comfy_dir, logfile=logfile) else "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="restart ComfyUI when it dies")
    ap.add_argument("--comfy-dir", default=".", help="ComfyUI 目录（含 main.py）")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--log", default="", help="日志文件；空 = 只打印")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    comfy_dir = Path(a.comfy_dir).resolve()
    logfile = Path(a.log) if a.log else None

    if a.once:
        print(check_once(comfy_dir, a.port, logfile))
        return 0

    log(f"看门狗启动：端口 {a.port}，每 {a.interval}s 探一次", logfile)
    last_restart = 0.0
    while True:
        try:
            if not alive(f"http://127.0.0.1:{a.port}") and time.time() - last_restart >= 60:
                log("检测到掉线", logfile)
                check_once(comfy_dir, a.port, logfile)
                last_restart = time.time()
        except Exception as e:
            log(f"探测异常：{e}", logfile)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
