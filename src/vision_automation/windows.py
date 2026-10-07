"""Thin Win32 helpers (ctypes): list windows, foreground control, polling waits.

Generic; nothing here knows about Notepad. All handles are declared as pointer-sized types so
they are not truncated on 64-bit Python.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, TypeVar

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

WM_CLOSE = 0x0010
SW_RESTORE = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

DESKTOP_CLASSES = frozenset({"Progman", "WorkerW"})  # the desktop shell windows


@dataclass(frozen=True)
class WinInfo:
    hwnd: int
    title: str
    cls: str
    exe: str  # lower-case image file name, e.g. "notepad.exe" ("" if it cannot be read)
    pid: int
    visible: bool


def _exe_name(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.replace("/", "\\").rsplit("\\", 1)[-1].lower()
        return ""
    finally:
        kernel32.CloseHandle(handle)


def get_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value


def get_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, len(buf))
    return buf.value


def _info(hwnd: int) -> WinInfo:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return WinInfo(int(hwnd), get_title(hwnd), get_class(hwnd), _exe_name(pid.value), pid.value,
                   bool(user32.IsWindowVisible(hwnd)))


def list_windows(visible_only: bool = True) -> list[WinInfo]:
    found: list[WinInfo] = []

    @_WNDENUMPROC
    def callback(hwnd, _lparam):
        if not visible_only or user32.IsWindowVisible(hwnd):
            found.append(_info(hwnd))
        return True

    user32.EnumWindows(callback, 0)
    return found


def window_exists(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd)) and bool(user32.IsWindowVisible(hwnd))


def foreground_hwnd() -> int:
    return int(user32.GetForegroundWindow() or 0)


def is_desktop_foreground() -> bool:
    hwnd = foreground_hwnd()
    return bool(hwnd) and get_class(hwnd) in DESKTOP_CLASSES


def bring_to_front(hwnd: int) -> bool:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    return foreground_hwnd() == int(hwnd)


def request_close(hwnd: int) -> None:
    """Ask a window to close (WM_CLOSE), exactly like clicking its X: the app may prompt to save."""
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


# ---- polling ---------------------------------------------------------------------------

T = TypeVar("T")


class WaitTimeout(TimeoutError):
    pass


def wait_for(predicate: Callable[[], T], timeout: float, what: str, interval: float = 0.1,
             sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic) -> T:
    """Poll until predicate() is truthy and return its value; raise WaitTimeout after `timeout` seconds."""
    deadline = clock() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if clock() >= deadline:
            raise WaitTimeout(f"timed out after {timeout:g}s waiting for {what}")
        sleep(interval)
