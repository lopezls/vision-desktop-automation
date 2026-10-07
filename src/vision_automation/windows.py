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

user32.GetDlgCtrlID.argtypes = [wintypes.HWND]
user32.GetDlgCtrlID.restype = ctypes.c_int
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                       wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
_WNDENUMCHILDPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumChildWindows.argtypes = [wintypes.HWND, _WNDENUMCHILDPROC, wintypes.LPARAM]
user32.EnumChildWindows.restype = wintypes.BOOL

WM_CLOSE = 0x0010
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
SMTO_ABORTIFHUNG = 0x0002
GA_ROOT = 2
FILENAME_EDIT_ID = 1001  # the "File name" edit control of the standard Open/Save dialogs
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


def is_foreground_within(hwnd: int) -> bool:
    """True when `hwnd` (or a window it owns/contains) has the keyboard focus."""
    fg = foreground_hwnd()
    return bool(fg) and (fg == int(hwnd) or int(user32.GetAncestor(fg, GA_ROOT) or 0) == int(hwnd))


def _control_text(hwnd, timeout_ms: int = 1000) -> str | None:
    """WM_GETTEXT with a timeout, so a hung dialog cannot freeze us. None if the control did not answer."""
    length = ctypes.c_size_t(0)
    if not user32.SendMessageTimeoutW(hwnd, WM_GETTEXTLENGTH, 0, 0, SMTO_ABORTIFHUNG, timeout_ms,
                                      ctypes.byref(length)):
        return None
    buf = ctypes.create_unicode_buffer(length.value + 2)
    got = ctypes.c_size_t(0)
    if not user32.SendMessageTimeoutW(hwnd, WM_GETTEXT, len(buf), ctypes.cast(buf, ctypes.c_void_p).value,
                                      SMTO_ABORTIFHUNG, timeout_ms, ctypes.byref(got)):
        return None
    return buf.value


def _descendants(parent) -> list[tuple[int, str, int]]:
    """(hwnd, class, control id) of every descendant window, depth-first as EnumChildWindows reports them."""
    rows: list[tuple[int, str, int]] = []

    @_WNDENUMCHILDPROC
    def callback(hwnd, _lparam):
        rows.append((int(hwnd), get_class(hwnd), int(user32.GetDlgCtrlID(hwnd))))
        return True

    user32.EnumChildWindows(parent, callback, 0)
    return rows


control_text = _control_text
descendants = _descendants


def filename_edit(dialog_hwnd: int) -> int | None:
    """The File name edit box of a standard Open/Save dialog.

    On current Windows it is an `Edit` (control id 1001) nested several levels deep inside a ComboBox,
    so GetDlgItem on the dialog does not see it. Fall back to the first Edit if the id differs.
    """
    edits = [(h, cid) for h, cls, cid in _descendants(dialog_hwnd) if cls.lower() == "edit"]
    for h, cid in edits:
        if cid == FILENAME_EDIT_ID:
            return h
    return edits[0][0] if edits else None


def get_dialog_filename(dialog_hwnd: int) -> str | None:
    """Text currently in the File name box of a standard file dialog, or None if it cannot be read."""
    edit = filename_edit(dialog_hwnd)
    return _control_text(edit) if edit else None


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
