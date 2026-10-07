"""Save As behaviour against a simulated dialog: swallowed Enter, Alt+S retry, overwrite prompt, retry limit."""

from dataclasses import replace
from pathlib import Path

import pytest

from vision_automation.config import CONFIG
from vision_automation.notepad import DIALOG_CLASS, NotepadError, WindowsNotepad, same_path
from vision_automation.windows import WinInfo

PATH = Path(r"C:\Users\lorel\OneDrive\Desktop\tjm-project\post_3.txt")
NOTEPAD_HWND = 100
DIALOG_HWND = 1

# Short waits so a "no response" costs milliseconds, not seconds.
FAST = replace(CONFIG, save_confirm_wait_s=0.25, field_read_timeout_s=0.25, dialog_timeout_s=1.0,
               focus_timeout_s=0.3, save_attempts=3, field_retries=2)


class FakeUi:
    """A Save As dialog that behaves as configured.

    swallow:        {"enter": n, "alt+s": n}: that many presses of the key do nothing (the dialog stays open)
    overwrite_on:   keys after which the "Confirm Save As" prompt appears (once) instead of saving
    mangle_typing:  the first N times text is typed, the box ends up with extra characters (autocomplete)
    corrupt_on_swallow: a swallowed confirm key also alters the box text
    unreadable:     filename_text returns None
    no_dialog_focus: the dialog never gets the keyboard focus
    """

    def __init__(self, swallow=None, overwrite_on=(), mangle_typing=0, corrupt_on_swallow=False,
                 unreadable=False, no_dialog_focus=False):
        self.swallow = dict(swallow or {})
        self.overwrite_on, self.mangle_typing = set(overwrite_on), mangle_typing
        self.corrupt_on_swallow, self.unreadable, self.no_dialog_focus = corrupt_on_swallow, unreadable, no_dialog_focus
        self.events, self.field = [], ""
        self.dialog_open = self.confirm_open = self.saved = False
        self._selected = False
        self._overwrite_used = False

    # -- keyboard --
    def hotkey(self, *keys):
        key = "+".join(keys)
        self.events.append(key)
        if key == "ctrl+shift+s":
            self.dialog_open, self.field, self._selected = True, "*.txt", True
        elif key == "ctrl+a":
            self._selected = True
        elif key == "alt+s":
            self._confirm("alt+s")

    def type_text(self, text):
        self.events.append(("type", text))
        self.field = text if self._selected else self.field + text
        self._selected = False
        if self.mangle_typing > 0:
            self.mangle_typing -= 1
            self.field += "~autocomplete"

    def press(self, key):
        self.events.append(key)
        if key == "enter":
            self._confirm("enter")
        elif key == "y" and self.confirm_open:
            self.confirm_open = self.dialog_open = False
            self.saved = True

    def _confirm(self, key):
        if not self.dialog_open or self.confirm_open:
            return
        if self.swallow.get(key, 0) > 0:
            self.swallow[key] -= 1
            if self.corrupt_on_swallow:
                self.field += "~"
            return
        if key in self.overwrite_on and not self._overwrite_used:
            self._overwrite_used = self.confirm_open = True
            return
        self.dialog_open, self.saved = False, True

    # -- OS queries --
    def has_focus(self, hwnd):
        return not (self.no_dialog_focus and hwnd == DIALOG_HWND)

    def focus(self, hwnd):
        return self.has_focus(hwnd)

    def dialog(self, kind):
        if kind == "save" and self.dialog_open:
            return WinInfo(DIALOG_HWND, "Save As", DIALOG_CLASS, "notepad.exe", 1, True)
        if kind == "confirm" and self.confirm_open:
            return WinInfo(2, "Confirm Save As", DIALOG_CLASS, "notepad.exe", 1, True)
        return None

    def filename_text(self, hwnd):
        return None if self.unreadable or not self.dialog_open else self.field

    # -- helpers for assertions --
    @property
    def confirm_keys(self):
        return [e for e in self.events if e in ("enter", "alt+s")]

    @property
    def typed(self):
        return [e[1] for e in self.events if isinstance(e, tuple)]


def save(ui, cfg=FAST):
    WindowsNotepad(cfg, ui).save_as(NOTEPAD_HWND, PATH)
    return ui


# ---- the normal path ---------------------------------------------------------------------

def test_enter_saves_when_it_registers():
    ui = save(FakeUi())
    assert ui.saved and ui.confirm_keys == ["enter"]
    assert ui.typed == [str(PATH)]
    assert ui.events[0] == "ctrl+shift+s"


def test_path_is_verified_in_the_box_before_enter_is_pressed():
    ui = FakeUi()
    save(ui)
    assert ui.events.index("enter") > ui.events.index(("type", str(PATH)))
    assert ui.field == str(PATH)


# ---- Enter swallowed -> Alt+S -------------------------------------------------------------

def test_enter_swallowed_once_then_alt_s_saves():
    ui = save(FakeUi(swallow={"enter": 1}))
    assert ui.saved and ui.confirm_keys == ["enter", "alt+s"]
    assert ui.typed == [str(PATH)]  # the box still held the path, so it was NOT re-typed


def test_the_box_is_reverified_and_retyped_before_the_retry_if_it_changed():
    ui = save(FakeUi(swallow={"enter": 1}, corrupt_on_swallow=True))
    assert ui.saved and ui.confirm_keys == ["enter", "alt+s"]
    assert ui.typed == [str(PATH), str(PATH)]  # corrupted by the swallowed Enter -> typed again
    assert ui.events.index("alt+s") > len(ui.events) - 3  # Alt+S came after the re-typing


# ---- overwrite prompt ---------------------------------------------------------------------

def test_overwrite_prompt_appearing_after_a_retry_is_answered_yes():
    ui = save(FakeUi(swallow={"enter": 1}, overwrite_on={"alt+s"}))
    assert ui.saved and ui.confirm_keys == ["enter", "alt+s"]
    assert ui.events[-1] == "y"
    assert "n" not in ui.events and "esc" not in ui.events  # never refuses or cancels


def test_overwrite_prompt_after_the_first_enter():
    ui = save(FakeUi(overwrite_on={"enter"}))
    assert ui.saved and ui.confirm_keys == ["enter"] and ui.events[-1] == "y"


def test_overwrite_prompt_that_will_not_close_is_a_clear_error():
    ui = FakeUi(overwrite_on={"enter"})
    ui.press = lambda key, orig=ui.press: (ui.events.append(key) if key == "y" else orig(key))  # 'y' does nothing
    with pytest.raises(NotepadError, match="overwrite confirmation did not close"):
        save(ui)


# ---- bounded retries ----------------------------------------------------------------------

def test_retry_limit_reached_stops_with_a_clear_error_and_leaves_the_dialog_open():
    ui = FakeUi(swallow={"enter": 99, "alt+s": 99})
    with pytest.raises(NotepadError) as ei:
        save(ui)
    msg = str(ei.value)
    assert "still open after 3 attempts" in msg and "Enter, Alt+S, Alt+S" in msg and "post_3.txt" in msg
    assert ui.confirm_keys == ["enter", "alt+s", "alt+s"]  # exactly the bound, no more
    assert ui.dialog_open and not ui.saved
    assert "esc" not in ui.events and "n" not in ui.events  # nothing destructive or cancelling


@pytest.mark.parametrize("attempts", [1, 2, 4])
def test_retry_bound_is_configurable(attempts):
    ui = FakeUi(swallow={"enter": 99, "alt+s": 99})
    with pytest.raises(NotepadError):
        save(ui, replace(FAST, save_attempts=attempts))
    assert len(ui.confirm_keys) == attempts


# ---- the File name box must hold exactly the path ----------------------------------------

def test_autocomplete_mangling_the_box_is_fixed_by_retyping_before_confirming():
    ui = save(FakeUi(mangle_typing=1))
    assert ui.saved and ui.typed == [str(PATH), str(PATH)]
    assert ui.events.index("enter") > max(i for i, e in enumerate(ui.events) if isinstance(e, tuple))


def test_a_box_that_never_matches_stops_before_any_confirmation_key():
    ui = FakeUi(mangle_typing=99)
    with pytest.raises(NotepadError, match="did not confirm the save"):
        save(ui)
    assert ui.confirm_keys == [] and ui.dialog_open
    assert len(ui.typed) == 1 + FAST.field_retries


def test_an_unreadable_box_refuses_to_confirm():
    ui = FakeUi(unreadable=True)
    with pytest.raises(NotepadError, match="unreadable"):
        save(ui)
    assert ui.confirm_keys == []


# ---- keystroke focus checks stay ----------------------------------------------------------

def test_no_keys_are_sent_to_a_dialog_that_does_not_have_the_focus():
    ui = FakeUi(no_dialog_focus=True)
    with pytest.raises(NotepadError, match="keyboard focus"):
        save(ui)
    assert ui.confirm_keys == [] and ui.typed == []
    assert ui.events == ["ctrl+shift+s"]  # only the hotkey that opened it, sent while Notepad had focus


def test_notepad_must_have_focus_before_the_save_hotkey():
    class NoFocus(FakeUi):
        def has_focus(self, hwnd):
            return hwnd != NOTEPAD_HWND

    ui = NoFocus()
    with pytest.raises(NotepadError, match="keyboard focus"):
        save(ui)
    assert ui.events == []


# ---- path comparison ----------------------------------------------------------------------

def test_same_path():
    p = str(PATH)
    assert same_path(p, p)
    assert same_path(p.lower(), p) and same_path(p.replace("\\", "/"), p)
    assert not same_path(None, p)
    assert not same_path(p + "x", p) and not same_path(p.replace("post_3", "post_4"), p)
    assert not same_path(p + "~autocomplete", p) and not same_path("", p)
