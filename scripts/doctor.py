"""Pre-flight checks for this project (stdlib only).

Usage: python scripts/doctor.py      (also: make doctor)
Exit code 1 if a REQUIRED check fails; WARN lines are advisory.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

PORTS: list[int] = [8080, 6379, 9090]
MIN_DISK_GB = 5
RECOMMENDED_WSL_GB = 5
ok_all = True


def report(level: str, msg: str) -> None:
    global ok_all
    if level == "FAIL":
        ok_all = False
    print(f"[{level:>4}] {msg}")


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.returncode, (out.stdout or out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def total_ram_gb() -> float | None:
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024**3
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        kb = int(meminfo.read_text().split("MemTotal:")[1].split()[0])
        return kb / 1024**2
    return None


def main() -> int:
    print(f"Doctor for {Path.cwd().name} — {platform.system()} {platform.release()}\n")

    version = platform.python_version()
    newer = tuple(map(int, version.split(".")[:2])) >= (3, 12)
    suffix = "" if newer else " — targets 3.12+ (uv installs it)"
    report("OK" if newer else "WARN", f"Python {version}{suffix}")

    for tool, required, hint in [
        ("uv", True, "winget install astral-sh.uv"),
        ("docker", True, "install Docker Desktop (WSL2 backend)"),
        ("make", False, "winget install ezwinports.make"),
        ("git", True, "winget install Git.Git"),
    ]:
        path = shutil.which(tool)
        if path:
            report("OK", f"{tool} found")
        else:
            report("FAIL" if required else "WARN", f"{tool} not found → {hint}")

    if shutil.which("docker"):
        code, _ = run(["docker", "info", "--format", "{{.ServerVersion}}"])
        msg = "Docker daemon reachable" if code == 0 else "Docker daemon not running — start Docker Desktop"
        report("OK" if code == 0 else "FAIL", msg)
        code, out = run(["docker", "compose", "version", "--short"])
        msg = f"docker compose {out}" if code == 0 else "docker compose plugin missing"
        report("OK" if code == 0 else "FAIL", msg)

    ram = total_ram_gb()
    if ram is not None:
        report("OK" if ram >= 7.5 else "WARN", f"Physical RAM: {ram:.1f} GB")

    wslconfig = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".wslconfig"
    if platform.system() == "Windows":
        if wslconfig.exists():
            text = wslconfig.read_text(errors="ignore").lower()
            line = next((ln for ln in text.splitlines() if ln.strip().startswith("memory")), None)
            hint = f"has no memory= limit (recommended: memory={RECOMMENDED_WSL_GB}GB)"
            report("OK" if line else "WARN", f".wslconfig {line or hint}")
        else:
            rec = f"[wsl2] memory={RECOMMENDED_WSL_GB}GB processors=6 swap=2GB"
            report("WARN", f"No %UserProfile%\\.wslconfig — recommended: {rec}")

    free_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
    disk_msg = f"Free disk: {free_gb:.1f} GB (need ≥ {MIN_DISK_GB} GB for images)"
    report("OK" if free_gb >= MIN_DISK_GB else "FAIL", disk_msg)

    busy = []
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
    ports_msg = f"Ports in use (another stack running?): {busy}" if busy else "Ports free"
    report("OK" if not busy else "WARN", ports_msg)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        ollama = s.connect_ex(("127.0.0.1", 11434)) == 0
    msg = "Ollama reachable on :11434" if ollama else "Ollama not running (only needed for LLM profiles)"
    report("OK" if ollama else "WARN", msg)

    print("\nResult:", "READY" if ok_all else "NOT READY — fix the FAIL lines above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
