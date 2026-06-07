#!/usr/bin/env python3
"""Shared lock helpers for long-running sweep orchestrators."""
from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator

LOCK_FILENAME = ".sweep.lock.json"


def _lock_metadata() -> Dict[str, object]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }


def _pid_alive_on_same_host(pid: object, hostname: object) -> bool:
    """Return True iff `pid` is still running AND we're on the recorded host.

    Used by `_format_existing_lock` to distinguish a live concurrent sweep
    from a stale lock left behind by SIGKILL / crash / power loss. Only
    returns True on the same hostname — across-host claims always look
    stale from this side, which is the conservative choice (we can't kill
    a remote process from here either).
    """
    try:
        pid_int = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        return False
    if pid_int is None or pid_int <= 0:
        return False
    if hostname is not None and hostname != socket.gethostname():
        return False
    try:
        os.kill(pid_int, 0)  # signal 0 = "are you alive?"
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but isn't ours — treat as alive (conservative).
        return True
    except OSError:
        return False
    return True


def _format_existing_lock(lock_path: Path) -> str:
    try:
        meta = json.loads(lock_path.read_text())
    except Exception:
        return f"another sweep already holds {lock_path} (lockfile unreadable)"
    pid = meta.get("pid")
    host = meta.get("hostname")
    alive = _pid_alive_on_same_host(pid, host)
    base = (
        f"another sweep already holds {lock_path}: "
        f"pid={pid} host={host} started_at={meta.get('started_at_iso')} "
        f"cwd={meta.get('cwd')}"
    )
    if not alive:
        # Lock owner is dead (or on a different host with no live PID we
        # can verify) — flag for manual cleanup. Lock is NOT auto-broken;
        # the user runs the unlock hint explicitly, preserving the strict
        # fail-loud contract.
        base += (
            f"\n  → recorded pid is not alive on this host; if you're "
            f"confident the previous sweep died, remove the stale lock "
            f"with:  rm {lock_path}"
        )
    return base


def acquire_out_dir_lock(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / LOCK_FILENAME
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(_format_existing_lock(lock_path)) from exc

    with os.fdopen(fd, "w") as f:
        json.dump(_lock_metadata(), f, indent=2, sort_keys=True)
        f.write("\n")
    return lock_path


def release_out_dir_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def hold_out_dir_lock(out_dir: Path) -> Iterator[Path]:
    lock_path = acquire_out_dir_lock(out_dir)
    try:
        yield lock_path
    finally:
        release_out_dir_lock(lock_path)


def claim_next_run_dir(base_dir: Path, *, prefix: str = "run-", width: int = 2) -> Path:
    """Atomically create the next numbered run directory under `base_dir`."""
    base_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base_dir / f"{prefix}{idx:0{width}d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            idx += 1
