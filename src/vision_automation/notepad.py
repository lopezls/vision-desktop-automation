"""The only Notepad-specific module: launch detection, blank-document handling, typing, Save As, closing.

Safety rules baked in:
  * Never discards the user's content: no "Don't save" is ever chosen. A save prompt that shows up
    is cancelled with Esc and reported.
  * Every keystroke group first checks that the right window has the keyboard focus, so text can
    never be typed into another window.
  * Waits poll for a condition (a window appearing, a dialog closing) and only give up after a timeout.
  * Saving never depends on a single keystroke: the File name box is read back and must hold exactly
    the expected path before anything confirms it, and the confirmation is retried (bounded).

The dialog titles ("Save As", "Confirm Save As") and the "Untitled - Notepad" window title are
assumptions about Windows 11 Notepad that real runs confirm.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Protocol

from . import inputs
from .config import CONFIG, Config
from .windows import (
    WaitTimeout,
    WinInfo,
    bring_to_front,
    control_text,
    descendants,
    get_dialog_filename,
    get_title,
    is_foreground_within,
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
EDITOR_CLASS = "RichEditD2DPT"  # Notepad's text area: a real RichEdit window that answers WM_GETTEXT

_CHUNK = re.compile(r"\S+\s*|\s+")  # a word together with the whitespace typed after it


class NotepadError(RuntimeError):
    pass


class _TypingFailed(Exception):
    """A word could not be typed as intended (handled inside type_text)."""


def is_blank_title(title: str) -> bool:
    """'Untitled - Notepad' = a new, unmodified, unsaved document. '*Untitled' (modified) or a file name is not."""
    t = title.strip().lower()
    return t.startswith("untitled") and not t.startswith("*")


def notepad_windows(lister: Callable[[], list[WinInfo]] = list_windows) -> list[WinInfo]:
    return [w for w in lister() if w.exe == NOTEPAD_EXE and w.cls != DIALOG_CLASS and w.title]


def split_chunks(text: str) -> list[str]:
    """Words, each with its trailing whitespace: ['Title: ', 'eum ', 'occaecati\\n\\n', 'ullam ', ...]."""
    return _CHUNK.findall(text)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def first_difference(expected: str, actual: str) -> int:
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            return i
    return min(len(expected), len(actual))


def same_path(a: str | None, b: str) -> bool:
    """Exact match, or the same Windows path (case, slashes, redundant separators)."""
    if a is None:
        return False
    return a == b or os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class Ui(Protocol):
    """Everything WindowsNotepad needs from the OS, so the retry logic can be tested with a fake."""

    def hotkey(self, *keys: str) -> None: ...
    def press(self, key: str) -> None: ...
    def type_text(self, text: str, pause: bool = True) -> None: ...
    def press_repeated(self, key: str, n: int) -> None: ...
    def editor_text(self, hwnd: int) -> str | None: ...  # the document text, read from the editor control
    def has_focus(self, hwnd: int) -> bool: ...
    def focus(self, hwnd: int) -> bool: ...
    def dialog(self, kind: str) -> WinInfo | None: ...  # kind: "save" or "confirm"
    def filename_text(self, hwnd: int) -> str | None: ...


class Win32Ui:
    def hotkey(self, *keys: str) -> None:
        inputs.hotkey(*keys)

    def press(self, key: str) -> None:
        inputs.press(key)

    def type_text(self, text: str, pause: bool = True) -> None:
        inputs.type_text(text, pause=pause)

    def press_repeated(self, key: str, n: int) -> None:
        inputs.press_repeated(key, n)

    def editor_text(self, hwnd: int) -> str | None:
        """The document text straight from Notepad's editor control (WM_GETTEXT), not via the clipboard."""
        editor = next((h for h, cls, _ in descendants(hwnd) if cls == EDITOR_CLASS), None)
        return None if editor is None else control_text(editor)

    def has_focus(self, hwnd: int) -> bool:
        return is_foreground_within(hwnd)

    def focus(self, hwnd: int) -> bool:
        return bring_to_front(hwnd)

    def dialog(self, kind: str) -> WinInfo | None:
        for w in list_windows():
            if w.exe != NOTEPAD_EXE or w.cls != DIALOG_CLASS:
                continue
            title = w.title.strip().lower()
            if (kind == "save" and title == "save as") or (kind == "confirm" and "confirm" in title):
                return w
        return None

    def filename_text(self, hwnd: int) -> str | None:
        return get_dialog_filename(hwnd)


class WindowsNotepad:
    def __init__(self, cfg: Config = CONFIG, ui: Ui | None = None):
        self.cfg = cfg
        self.ui: Ui = ui or Win32Ui()

    # -- focus --

    def _require_focus(self, hwnd: int, what: str = "Notepad") -> None:
        """Refuse to send keystrokes unless `hwnd` has the keyboard focus (try to give it focus first)."""
        if self.ui.has_focus(hwnd):
            return
        self.ui.focus(hwnd)
        try:
            wait_for(lambda: self.ui.has_focus(hwnd) or (self.ui.focus(hwnd) and self.ui.has_focus(hwnd)),
                     self.cfg.focus_timeout_s, f"{what} to take the keyboard focus")
        except WaitTimeout:
            raise NotepadError(f"{what} does not have the keyboard focus; refusing to send keystrokes") from None

    def _cancel_prompt(self, hwnd: int) -> None:
        """Esc cancels a 'save changes?' prompt (never 'Don't save'), leaving the user's document alone."""
        try:
            self.ui.focus(hwnd)
            self.ui.press("esc")
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
            self.ui.hotkey("ctrl", "n")
            try:
                wait_for(lambda: is_blank_title(get_title(hwnd)), 4.0, "a blank 'Untitled' tab")
                return
            except WaitTimeout:
                continue
        raise NotepadError(f"could not get a blank document; the window title is {get_title(hwnd)!r}")

    # -- typing --

    def _read_doc(self, hwnd: int) -> str:
        """The document text, read back from the editor (newlines normalized to \\n)."""
        text = self.ui.editor_text(hwnd)
        if text is None:
            raise NotepadError("Could not read the document text from Notepad's editor control, so the typed "
                               "text cannot be verified.")
        return normalize_newlines(text)

    def _read_settled(self, hwnd: int) -> str:
        """Read the editor after a short pause, until two reads agree (autocorrect lands as the delimiter is typed)."""
        time.sleep(self.cfg.chunk_settle_s)
        text = self._read_doc(hwnd)
        for _ in range(5):
            time.sleep(0.02)
            again = self._read_doc(hwnd)
            if again == text:
                return text
            text = again
        return text

    def _clear_document(self, hwnd: int) -> None:
        self._require_focus(hwnd)
        self.ui.hotkey("ctrl", "a")
        self.ui.press("delete")
        if self._read_settled(hwnd) != "":
            raise NotepadError("Could not clear the Notepad document to re-type it.")

    def _repair_chunk(self, hwnd: int, prev: str, chunk: str) -> None:
        """Make the document equal prev + chunk after Notepad (or a dropped key) changed the typed word.

        Notepad's autocorrect fires when a delimiter (space, Enter) is typed after a word, so typing the same
        chunk again would be rewritten the same way. Instead: remove what this chunk produced, type the
        delimiter FIRST, step Left over it, type the word, then step Right. No delimiter is ever typed after
        the word, so nothing is corrected.
        """
        word = chunk.rstrip()
        delims = chunk[len(word):]
        for attempt in range(1, self.cfg.chunk_retries + 1):
            self._require_focus(hwnd)
            now = self._read_settled(hwnd)
            if not now.startswith(prev):
                raise _TypingFailed(f"text before the word {chunk!r} changed")
            self.ui.press_repeated("backspace", len(now) - len(prev))
            if self._read_settled(hwnd) != prev:
                raise _TypingFailed(f"could not remove the mistyped word {chunk!r}")
            if delims:
                self.ui.type_text(delims, pause=False)
                self.ui.press_repeated("left", len(delims))
                self.ui.type_text(word, pause=False)
                self.ui.press_repeated("right", len(delims))
            else:
                self.ui.type_text(word, pause=False)
            now = self._read_settled(hwnd)
            if now == prev + chunk:
                log.info("[notepad] repaired %r (Notepad's editor had changed it; attempt %d)", chunk, attempt)
                return
        raise _TypingFailed(f"the word {chunk!r} still does not come out as typed (editor shows {now[len(prev):]!r})")

    def _type_verified(self, hwnd: int, text: str) -> None:
        prev = ""
        if self._read_settled(hwnd) != "":
            raise NotepadError("The Notepad document is not empty; refusing to type into someone else's text.")
        for chunk in split_chunks(text):
            self._require_focus(hwnd)
            self.ui.type_text(chunk, pause=False)
            now = self._read_settled(hwnd)
            if now != prev + chunk:
                self._repair_chunk(hwnd, prev, chunk)
            prev += chunk

    def type_text(self, hwnd: int, text: str) -> None:
        """Type `text`, verifying the editor's own text word by word, then once more in full before returning.

        A word that Notepad rewrote or that lost a key is repaired in place. If the document still differs
        at the end, it is cleared (Ctrl+A, Delete) and typed again, up to `type_retries` times; after that
        this raises NotepadError. Never uses the clipboard.
        """
        last = ""
        for attempt in range(1 + self.cfg.type_retries):
            if attempt:
                log.warning("[notepad] document differs from the expected text; clearing and typing it again "
                            "(retry %d/%d)", attempt, self.cfg.type_retries)
                self._clear_document(hwnd)
            try:
                self._type_verified(hwnd, text)
                last = self._read_settled(hwnd)
            except _TypingFailed as e:
                log.warning("[notepad] typing check failed: %s", e)
                last = self._read_settled(hwnd)
            else:
                if last == text:
                    return
        i = first_difference(text, last)
        found = last[i] if i < len(last) else "<end of text>"
        want = text[i] if i < len(text) else "<end of text>"
        raise NotepadError(
            f"The text in Notepad still differs from the expected text after {1 + self.cfg.type_retries} "
            f"attempts: expected {len(text)} chars, found {len(last)}; first difference at index {i} "
            f"(expected {want!r}, found {found!r}). I stopped; the document was left as is.")

    # -- saving --

    def _path_in_box(self, dlg: int, expected: str) -> str | None:
        """The File name box text once it equals `expected`, or None if it does not within the read timeout."""
        try:
            return wait_for(lambda: (t := self.ui.filename_text(dlg)) is not None and same_path(t, expected) and t,
                            self.cfg.field_read_timeout_s, "the File name box to show the path")
        except WaitTimeout:
            return None

    def _enter_path(self, dlg: int, expected: str) -> None:
        """Type the path into the File name box and read it back; it must match exactly before we confirm."""
        for attempt in range(1 + self.cfg.field_retries):
            self._require_focus(dlg, "the Save As dialog")
            self.ui.hotkey("ctrl", "a")  # select whatever is in the box so typing replaces it
            self.ui.type_text(expected)
            if self._path_in_box(dlg, expected) is not None:
                return
            log.warning("[notepad] File name box holds %r, expected %r (try %d/%d)", self.ui.filename_text(dlg),
                        expected, attempt + 1, 1 + self.cfg.field_retries)
        text = self.ui.filename_text(dlg)
        shown = "unreadable" if text is None else repr(text)
        raise NotepadError(f"The File name box holds {shown} instead of {expected!r}; I did not confirm the save. "
                           "The Save As dialog was left open.")

    def _ensure_path(self, dlg: int, expected: str) -> None:
        """Before re-confirming: keep the box if it still holds the path, otherwise type it again."""
        if self._path_in_box(dlg, expected) is None:
            log.warning("[notepad] File name box changed to %r; typing the path again", self.ui.filename_text(dlg))
            self._enter_path(dlg, expected)

    def _dialog_outcome(self) -> str | None:
        """'confirm' (overwrite prompt up), 'closed' (Save As gone), or None if the dialog is still open."""
        def settled() -> str | None:
            if self.ui.dialog("confirm"):
                return "confirm"
            return "closed" if not self.ui.dialog("save") else None

        try:
            return wait_for(settled, self.cfg.save_confirm_wait_s, "the Save As dialog to respond")
        except WaitTimeout:
            return None

    def _accept_overwrite(self, path: Path) -> None:
        confirm = self.ui.dialog("confirm")
        if confirm is None:
            return
        log.warning("[notepad] %s already exists; confirming the overwrite", path.name)
        self._require_focus(confirm.hwnd, "the overwrite confirmation")
        self.ui.press("y")  # the dialog's "&Yes" (its default button is "No", so Enter would refuse)
        try:
            wait_for(lambda: not self.ui.dialog("confirm") and not self.ui.dialog("save"),
                     self.cfg.dialog_timeout_s, "the overwrite confirmation to close")
        except WaitTimeout:
            raise NotepadError("The overwrite confirmation did not close after answering Yes; "
                               "I left it open for you to look at.") from None

    def save_as(self, hwnd: int, path: Path) -> None:
        """Ctrl+Shift+S, put the full path in the File name box, verify it, then confirm.

        Confirmation does not rely on one key: Enter first; if the dialog is still open `save_confirm_wait_s`
        later (Enter swallowed, e.g. by filename autocomplete) the box is re-verified and Alt+S (the Save
        button's accelerator) is tried, up to `save_attempts` confirmations in total. An overwrite
        confirmation can appear after any of them and is answered Yes (never "Don't save").
        """
        self._require_focus(hwnd)
        self.ui.hotkey("ctrl", "shift", "s")
        dlg = wait_for(lambda: self.ui.dialog("save"), self.cfg.dialog_timeout_s, "the Save As dialog").hwnd
        self.complete_save_dialog(dlg, path)

    def complete_save_dialog(self, dlg: int, path: Path) -> None:
        """Fill in and confirm an already-open Save As dialog (the part of save_as after it opens)."""
        expected = str(path)
        self._enter_path(dlg, expected)

        tried: list[str] = []
        for attempt in range(1, self.cfg.save_attempts + 1):
            if attempt > 1:
                if not self.ui.dialog("save"):  # it closed while we were deciding
                    return
                self._ensure_path(dlg, expected)  # re-verify (or re-type) before pressing anything again
            self._require_focus(dlg, "the Save As dialog")
            if attempt == 1:
                self.ui.press("enter")
                tried.append("Enter")
            else:
                self.ui.hotkey("alt", "s")
                tried.append("Alt+S")
            outcome = self._dialog_outcome()
            if outcome == "confirm":
                self._accept_overwrite(path)
                return
            if outcome == "closed":
                return
            log.warning("[notepad] Save As still open %.1fs after %s (attempt %d/%d)",
                        self.cfg.save_confirm_wait_s, tried[-1], attempt, self.cfg.save_attempts)
        raise NotepadError(
            f"The Save As dialog is still open after {len(tried)} attempts ({', '.join(tried)}); "
            f"the File name box holds {self.ui.filename_text(dlg)!r}. I stopped and left the dialog open.")

    # -- closing --

    def close_document(self, hwnd: int) -> None:
        """Close our (saved) tab with Ctrl+W, then the window if other tabs keep it open."""
        if not window_exists(hwnd):
            return
        self._require_focus(hwnd)
        self.ui.hotkey("ctrl", "w")
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
