"""The only Notepad-specific module: launch detection, blank-document handling, typing, Save As, closing.

Safety rules baked in:
  * Never discards the user's content: no "Don't save" is ever chosen. A save prompt that shows up
    is cancelled with Esc and reported.
  * Every keystroke group first checks that Notepad has the keyboard focus, so text can never be
    typed into another window.
  * Waits poll for a condition (a window appearing, a dialog closing) and only give up after a timeout.

The dialog titles ("Save As", "Confirm Save As") and the "Untitled - Notepad" window title are
assumptions about Windows 11 Notepad that the first real run confirms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from . import inputs
from .config import CONFIG, Config
from .windows import (
    WaitTimeout,
    WinInfo,
    bring_to_front,
    foreground_hwnd,
    get_title,
    list_windows,
    request_close,
    wait_for,
    window_exists,
)

log = logging.getLogger("vision_automation")

# Free-text target description handed to the generic search. The taskbar clause makes the search
# exclude the taskbar band (see search.excludes_taskbar); the taskbar also holds a Notepad icon.
TARGET_DESCRIPTION = "the Notepad desktop icon (not the taskbar)"

NOTEPAD_EXE = "notepad.exe"
DIALOG_CLASS = "#32770"


class NotepadError(RuntimeError):
    pass


def is_blank_title(title: str) -> bool:
    """'Untitled - Notepad' = a new, unmodified, unsaved document. '*Untitled' (modified) or a file name is not."""
    t = title.strip().lower()
    return t.startswith("untitled") and not t.startswith("*")


def notepad_windows(lister: Callable[[], list[WinInfo]] = list_windows) -> list[WinInfo]:
    return [w for w in lister() if w.exe == NOTEPAD_EXE and w.cls != DIALOG_CLASS and w.title]


class WindowsNotepad:
    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg

    # -- focus --

    def _require_focus(self, hwnd: int) -> None:
        if foreground_hwnd() == hwnd:
            return
        if not bring_to_front(hwnd):
            wait_for(lambda: bring_to_front(hwnd) or foreground_hwnd() == hwnd, 3.0,
                     "Notepad to take the keyboard focus")
        if foreground_hwnd() != hwnd:
            raise NotepadError("Notepad does not have the keyboard focus; refusing to send keystrokes")

    def _cancel_prompt(self, hwnd: int) -> None:
        """Esc cancels a 'save changes?' prompt (never 'Don't save'), leaving the user's document alone."""
        try:
            bring_to_front(hwnd)
            inputs.press("esc")
        except Exception as e:  # best effort; the caller raises the real error
            log.warning("[notepad] could not send Esc: %s", e)

    # -- before launch --

    def close_existing(self) -> None:
        """Close any Notepad window that is already open (gracefully), or fail without touching its content."""
        existing = notepad_windows()
        if not existing:
            return
        log.info("[notepad] closing %d already-open Notepad window(s)", len(existing))
        for w in existing:
            request_close(w.hwnd)
        try:
            wait_for(lambda: not notepad_windows(), self.cfg.close_timeout_s,
                     "the already-open Notepad window(s) to close")
        except WaitTimeout:
            for w in notepad_windows():
                self._cancel_prompt(w.hwnd)
            raise NotepadError(
                "An already-open Notepad window did not close, probably because it is asking whether to save "
                "changes. I cancelled the prompt and did not touch your document. Save or close that Notepad "
                "window yourself, then run again.") from None

    def snapshot(self) -> frozenset[int]:
        """Handles of the Notepad windows open right now (take it before the double-click)."""
        return frozenset(w.hwnd for w in notepad_windows())

    # -- after launch --

    def wait_for_launch(self, before: frozenset[int]) -> int:
        """Wait for a NEW Notepad window (not a fixed sleep) and give it the keyboard focus."""
        def new_window() -> WinInfo | None:
            return next((w for w in notepad_windows() if w.hwnd not in before), None)

        try:
            win = wait_for(new_window, self.cfg.launch_timeout_s, "a new Notepad window to appear")
        except WaitTimeout as e:
            raise NotepadError(f"{e}. The double-click may not have landed on the Notepad icon.") from None
        self._require_focus(win.hwnd)
        log.info("[notepad] window up: %r", get_title(win.hwnd))
        return win.hwnd

    def prepare_blank(self, hwnd: int) -> None:
        """Make sure the active tab is a blank untitled document, leaving any restored tabs alone."""
        title = get_title(hwnd)
        if is_blank_title(title):
            return
        log.info("[notepad] start state %r is not a blank document (restored session?); opening a new tab", title)
        for _ in range(2):
            self._require_focus(hwnd)
            inputs.hotkey("ctrl", "n")
            try:
                wait_for(lambda: is_blank_title(get_title(hwnd)), 4.0, "a blank 'Untitled' tab")
                return
            except WaitTimeout:
                continue
        raise NotepadError(f"could not get a blank document; the window title is {get_title(hwnd)!r}")

    # -- typing and saving --

    def type_text(self, hwnd: int, text: str) -> None:
        self._require_focus(hwnd)
        inputs.type_text(text)
        self._require_focus(hwnd)  # focus lost mid-typing would also fail the content check, but say so early

    def _dialog(self, kind: str) -> WinInfo | None:
        for w in list_windows():
            if w.exe != NOTEPAD_EXE or w.cls != DIALOG_CLASS:
                continue
            title = w.title.strip().lower()
            if (kind == "save" and title == "save as") or (kind == "confirm" and "confirm" in title):
                return w
        return None

    def save_as(self, hwnd: int, path: Path) -> None:
        """Ctrl+Shift+S, type the full path, Enter; accept the overwrite confirmation if it appears."""
        self._require_focus(hwnd)
        inputs.hotkey("ctrl", "shift", "s")
        dlg = wait_for(lambda: self._dialog("save"), self.cfg.dialog_timeout_s, "the Save As dialog")
        if foreground_hwnd() != dlg.hwnd:
            bring_to_front(dlg.hwnd)
        inputs.type_text(str(path))
        inputs.press("enter")

        def settled() -> str | None:
            if self._dialog("confirm"):
                return "confirm"
            return "closed" if not self._dialog("save") else None

        state = wait_for(settled, self.cfg.dialog_timeout_s, "the Save As dialog to finish")
        if state == "confirm":
            confirm = self._dialog("confirm")
            log.warning("[notepad] %s already exists; confirming overwrite", path.name)
            if foreground_hwnd() != confirm.hwnd:
                bring_to_front(confirm.hwnd)
            inputs.press("y")  # the dialog's "&Yes"
            wait_for(lambda: not self._dialog("confirm") and not self._dialog("save"),
                     self.cfg.dialog_timeout_s, "the overwrite confirmation to close")

    # -- closing --

    def close_document(self, hwnd: int) -> None:
        """Close our (saved) tab with Ctrl+W, then the window if other tabs keep it open."""
        if not window_exists(hwnd):
            return
        self._require_focus(hwnd)
        inputs.hotkey("ctrl", "w")
        try:
            wait_for(lambda: not window_exists(hwnd), 3.0, "Notepad to close with its last tab")
            return
        except WaitTimeout:
            pass
        # Other (restored) tabs keep the window open: close the window itself. Win11 Notepad keeps
        # those tabs' content for next time without prompting, so nothing is lost.
        request_close(hwnd)
        try:
            wait_for(lambda: not window_exists(hwnd), self.cfg.close_timeout_s, "Notepad to close")
        except WaitTimeout:
            self._cancel_prompt(hwnd)
            raise NotepadError(
                "Notepad did not close, probably because it is asking whether to save changes. I cancelled the "
                "prompt (never 'Don't save') and left it open. Check the Notepad window.") from None
