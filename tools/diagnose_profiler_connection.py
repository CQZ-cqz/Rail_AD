"""
工控机 3D 扫描器连接诊断脚本。

用途:
1) 检查 Python/环境信息和 MechEyeAPI 安装情况。
2) 检查网络配置（ipconfig、route、可选 ping）。
3) 检查 Mech-Eye SDK Python 接口是否可导入。
4) 尝试自动发现连接 find_and_connect。
5) 可选按 IP 直连 Profiler.connect(ip)。

示例:
  python tools/diagnose_profiler_connection.py
  python tools/diagnose_profiler_connection.py --ip 192.168.1.10
  python tools/diagnose_profiler_connection.py --ip 192.168.1.10 --save log/profiler_diag.txt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import locale
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def _decode_bytes(raw: bytes) -> str:
    """Decode subprocess bytes with locale-aware fallbacks (Windows-friendly)."""
    encodings = []
    pref = locale.getpreferredencoding(False)
    if pref:
        encodings.append(pref)
    encodings.extend(["utf-8", "cp936", "gbk", "cp1252"])

    tried = set()
    for enc in encodings:
        if not enc or enc.lower() in tried:
            continue
        tried.add(enc.lower())
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, _decode_bytes(proc.stdout), _decode_bytes(proc.stderr)
    except Exception as exc:
        return 999, "", f"{type(exc).__name__}: {exc}"


class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def pass_(self, msg: str) -> None:
        self.pass_count += 1
        self.lines.append(f"[PASS] {msg}")

    def warn(self, msg: str) -> None:
        self.warn_count += 1
        self.lines.append(f"[WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.fail_count += 1
        self.lines.append(f"[FAIL] {msg}")

    def dump(self) -> str:
        return "\n".join(self.lines)


def _check_python_env(r: Report) -> None:
    r.add("=== 1) Python / Environment ===")
    r.add(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    r.add(f"Python executable: {sys.executable}")
    r.add(f"Python version: {sys.version.split()[0]}")
    r.add(f"Platform: {platform.platform()}")

    try:
        ver = importlib.metadata.version("MechEyeAPI")
        r.pass_(f"MechEyeAPI installed: {ver}")
    except importlib.metadata.PackageNotFoundError:
        r.fail("MechEyeAPI package not found in current Python environment.")
    except Exception as exc:
        r.warn(f"Unable to read MechEyeAPI version: {exc}")

    r.add()


def _check_network(r: Report, ip: Optional[str]) -> None:
    r.add("=== 2) Network Basics ===")

    host = socket.gethostname()
    r.add(f"Hostname: {host}")
    try:
        addrs = socket.gethostbyname_ex(host)[2]
        r.add(f"Host IPv4(s): {', '.join(addrs) if addrs else '(none)'}")
    except Exception as exc:
        r.warn(f"Unable to resolve host IPs: {exc}")

    code, out, err = _run_cmd(["ipconfig"], timeout=10)
    if code == 0:
        r.pass_("ipconfig executed.")
        r.add("--- ipconfig (first 120 lines) ---")
        r.add("\n".join(out.splitlines()[:120]))
    else:
        r.warn(f"ipconfig failed: {err.strip() or f'exit code {code}'}")

    code, out, err = _run_cmd(["route", "print"], timeout=10)
    if code == 0:
        r.pass_("route print executed.")
        r.add("--- route print (first 120 lines) ---")
        r.add("\n".join(out.splitlines()[:120]))
    else:
        r.warn(f"route print failed: {err.strip() or f'exit code {code}'}")

    if ip:
        code, out, err = _run_cmd(["ping", "-n", "2", ip], timeout=10)
        if code == 0 and "TTL=" in out.upper():
            r.pass_(f"Ping to {ip} succeeded.")
        else:
            r.warn(f"Ping to {ip} failed or unstable.")
        r.add("--- ping output ---")
        r.add((out + "\n" + err).strip())
    else:
        r.warn("No --ip provided; ping check skipped.")

    r.add()


def _check_mecheye_sdk(r: Report, ip: Optional[str]) -> None:
    r.add("=== 3) Mech-Eye SDK Checks ===")

    try:
        shared = importlib.import_module("mecheye.shared")
        profiler_mod = importlib.import_module("mecheye.profiler")
        profiler_utils = importlib.import_module("mecheye.profiler_utils")
        r.pass_("Imported mecheye.shared / mecheye.profiler / mecheye.profiler_utils")
    except Exception:
        r.fail("Failed to import mecheye modules. SDK runtime or Python binding may be broken.")
        r.add(traceback.format_exc())
        r.add()
        return

    try:
        Profiler = getattr(profiler_mod, "Profiler")
        profiler = Profiler()
        r.pass_("Profiler object created.")
    except Exception:
        r.fail("Failed to construct Profiler object.")
        r.add(traceback.format_exc())
        r.add()
        return

    try:
        fac = getattr(profiler_utils, "find_and_connect")
        ok = bool(fac(profiler))
        if ok:
            r.pass_("find_and_connect succeeded (auto discovery works).")
            try:
                profiler.disconnect()
                r.pass_("Disconnected profiler after discovery test.")
            except Exception as exc:
                r.warn(f"Disconnect after discovery test failed: {exc}")
        else:
            r.fail("find_and_connect failed (no profiler discovered).")
    except Exception:
        r.fail("find_and_connect call raised exception.")
        r.add(traceback.format_exc())

    if ip:
        try:
            Profiler = getattr(profiler_mod, "Profiler")
            profiler2 = Profiler()
            status = profiler2.connect(ip)
            is_ok = bool(status.is_ok()) if hasattr(status, "is_ok") else False
            if is_ok:
                r.pass_(f"Direct connect({ip}) succeeded.")
                try:
                    profiler2.disconnect()
                    r.pass_("Disconnected profiler after direct IP test.")
                except Exception as exc:
                    r.warn(f"Disconnect after direct IP test failed: {exc}")
            else:
                r.fail(f"Direct connect({ip}) failed.")
                try:
                    show_error = getattr(shared, "show_error")
                    r.add("--- SDK error detail for direct connect ---")
                    # show_error writes to stdout; keep this call for vendor-specific diagnostics.
                    show_error(status)
                except Exception as exc:
                    r.warn(f"Unable to print SDK error detail: {exc}")
        except Exception:
            r.fail("Direct IP connect test raised exception.")
            r.add(traceback.format_exc())
    else:
        r.warn("No --ip provided; direct IP connect test skipped.")

    r.add()


def _final_summary(r: Report) -> None:
    r.add("=== 4) Summary ===")
    r.add(f"PASS: {r.pass_count}")
    r.add(f"WARN: {r.warn_count}")
    r.add(f"FAIL: {r.fail_count}")

    if r.fail_count == 0:
        r.pass_("No hard failures in diagnostic checks.")
    else:
        r.add("Suggested focus order:")
        r.add("1. 确认当前 Python 环境与厂家软件使用的 SDK 主版本一致。")
        r.add("2. 若发现失败但直连成功，优先排查工控机防火墙/多网卡/广播限制。")
        r.add("3. 若直连也失败，优先排查 IP/子网/路由/网线与设备上电状态。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mech-Eye profiler connection diagnostic tool")
    parser.add_argument("--ip", default="", help="Profiler IP for ping and direct connect test")
    parser.add_argument("--save", default="", help="Optional report output path, e.g. log/profiler_diag.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ip = args.ip.strip() or None

    report = Report()
    _check_python_env(report)
    _check_network(report, ip)
    _check_mecheye_sdk(report, ip)
    _final_summary(report)

    text = report.dump()
    print(text)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\nReport saved to: {out_path}")

    return 0 if report.fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
