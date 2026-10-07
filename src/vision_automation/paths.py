"""Windows path helpers. The Desktop may be redirected (e.g. into OneDrive)."""

import ctypes
import uuid
from ctypes import wintypes
from pathlib import Path

from .config import PROJECT_FOLDER_NAME

# FOLDERID_Desktop
_FOLDERID_DESKTOP = uuid.UUID("B4BFCC3A-DB2C-424C-B029-7FE99A87C641")


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def desktop_dir() -> Path:
    guid = _GUID()
    guid.Data1, guid.Data2, guid.Data3 = (
        _FOLDERID_DESKTOP.time_low,
        _FOLDERID_DESKTOP.time_mid,
        _FOLDERID_DESKTOP.time_hi_version,
    )
    for i, b in enumerate(_FOLDERID_DESKTOP.bytes[8:]):
        guid.Data4[i] = b

    out = ctypes.c_wchar_p()
    hr = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out))
    if hr != 0:
        raise OSError(f"SHGetKnownFolderPath failed (HRESULT {hr:#x})")
    try:
        return Path(out.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(out)


def project_dir(create: bool = True) -> Path:
    path = desktop_dir() / PROJECT_FOLDER_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
