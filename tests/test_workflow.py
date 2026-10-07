"""Workflow tests with fake screen/Notepad/searcher objects that record every action in order."""

from dataclasses import replace
from pathlib import Path

import pyautogui
import pytest
from PIL import Image

from vision_automation.config import CONFIG
from vision_automation.notepad import NotepadError
from vision_automation.planner import PopupAssessment
from vision_automation.posts import Post
from vision_automation.search import GroundingError, Located, PopupBlocked
from vision_automation.types import BBox, Point
from vision_automation.verifier import Verdict
from vision_automation.workflow import Workflow, WorkflowError

FAST = replace(CONFIG, dialog_timeout_s=0.3, popup_dismissals=2)


def posts(n=3):
    return [Post(i, f"title {i}", f"body line one\nbody line two {i}") for i in range(1, n + 1)]


class Events(list):
    def count_of(self, name):
        return sum(1 for e in self if e[0] == name)


class FakeScreen:
    def __init__(self, events):
        self.events, self.shots = events, []

    def show_desktop(self):
        self.events.append(("show_desktop",))

    def fresh_screenshot(self):
        img = Image.new("RGB", (64, 36))  # a distinct object every call
        self.shots.append(img)
        self.events.append(("capture", id(img)))
        return img

    def click(self, p):
        self.events.append(("click", p))

    def double_click(self, p):
        self.events.append(("double_click", p))

    def debug_screenshot(self):
        return Image.new("RGB", (64, 36))


class FakeSearcher:
    """Returns a different click point on every call, so reuse of old coordinates would be visible."""

    def __init__(self, events, script=None):
        self.events, self.calls, self.script = events, 0, list(script or [])

    def find(self, shot, description):
        self.calls += 1
        self.events.append(("find", id(shot)))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
        return Located(Point(100 * self.calls, 50 * self.calls), BBox(0, 0, 10, 10), Verdict("is_target"), calls=7)


class FakeNotepad:
    def __init__(self, events, fail_at=None, fail_with=None, written=None, skip_write=False):
        self.events, self.fail_at, self.fail_with = events, fail_at, fail_with or NotepadError("boom")
        self.written, self.skip_write, self.typed = written, skip_write, []
        self._hwnd = 100

    def _step(self, name):
        self.events.append((name,))
        if self.fail_at == name:
            raise self.fail_with

    def close_existing(self):
        self._step("close_existing")

    def snapshot(self):
        self.events.append(("snapshot",))
        return frozenset()

    def wait_for_launch(self, before):
        self._step("wait_for_launch")
        self._hwnd += 1
        return self._hwnd

    def prepare_blank(self, hwnd):
        self._step("prepare_blank")

    def type_text(self, hwnd, text):
        self._step("type_text")
        self.typed.append(text)

    def save_as(self, hwnd, path):
        self._step("save_as")
        if not self.skip_write:
            text = self.typed[-1] if self.written is None else self.written
            path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))  # Notepad writes CRLF

    def close_document(self, hwnd):
        self._step("close_document")


def make(tmp_path, events=None, notepad_kwargs=None, searcher_script=None, cfg=FAST):
    events = Events() if events is None else events
    screen = FakeScreen(events)
    notepad = FakeNotepad(events, **(notepad_kwargs or {}))
    searcher = FakeSearcher(events, searcher_script)
    wf = Workflow(searcher, screen, notepad, tmp_path, "the target", cfg, debug_dir=tmp_path / "dbg")
    return wf, events, screen, notepad, searcher


def names(events):
    return [e[0] for e in events]


# ---- order and freshness ----------------------------------------------------------------

def test_one_post_runs_every_step_in_order(tmp_path):
    wf, events, *_ = make(tmp_path)
    [res] = wf.run(posts(1))
    assert names(events) == ["close_existing", "show_desktop", "capture", "find", "snapshot", "double_click",
                             "wait_for_launch", "prepare_blank", "type_text", "save_as", "close_document"]
    assert res.path == tmp_path / "post_1.txt" and res.path.exists()
    assert res.click_point == Point(100, 50) and res.search_calls == 7


def test_content_typed_is_title_blank_line_body(tmp_path):
    wf, _, _, notepad, _ = make(tmp_path)
    wf.run(posts(1))
    assert notepad.typed == ["Title: title 1\n\nbody line one\nbody line two 1"]


def test_every_post_shows_desktop_takes_a_fresh_screenshot_and_never_reuses_coordinates(tmp_path):
    wf, events, screen, _, _ = make(tmp_path)
    wf.run(posts(3))
    assert events.count_of("show_desktop") == 3 and events.count_of("capture") == 3 and events.count_of("find") == 3
    # per post: show_desktop -> capture -> find, in that order, before the double-click
    idx = [i for i, e in enumerate(events) if e[0] == "show_desktop"]
    for i in idx:
        assert names(events[i:i + 3]) == ["show_desktop", "capture", "find"]
    # each search got the screenshot captured just before it, and each click uses that search's point
    assert len({e[1] for e in events if e[0] == "capture"}) == 3
    for cap, find in zip([e for e in events if e[0] == "capture"], [e for e in events if e[0] == "find"]):
        assert cap[1] == find[1]
    assert [e[1] for e in events if e[0] == "double_click"] == [Point(100, 50), Point(200, 100), Point(300, 150)]


def test_existing_notepad_is_dealt_with_before_anything_is_clicked(tmp_path):
    wf, events, *_ = make(tmp_path)
    wf.run(posts(1))
    assert names(events).index("close_existing") < names(events).index("show_desktop")


# ---- stop on first failure --------------------------------------------------------------

def test_stops_at_the_first_failure_and_never_touches_later_posts(tmp_path):
    wf, events, *_ = make(tmp_path, notepad_kwargs={"fail_at": "type_text", "fail_with": NotepadError("lost focus")})
    # post 1 succeeds first; make post 2 fail by failing on the second type_text
    notepad = wf.notepad
    original = notepad.type_text
    calls = {"n": 0}

    def flaky(hwnd, text):
        calls["n"] += 1
        if calls["n"] == 2:
            raise NotepadError("lost focus")
        original(hwnd, text)

    notepad.fail_at = None
    notepad.type_text = flaky
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(3))
    e = ei.value
    assert e.post_id == 2 and e.stage == "type" and "lost focus" in str(e)
    assert events.count_of("close_existing") == 2  # post 3 never started
    assert not (tmp_path / "post_2.txt").exists() and not (tmp_path / "post_3.txt").exists()
    assert (tmp_path / "post_1.txt").exists()
    assert e.debug_image is not None and Path(e.debug_image).exists()


def test_a_failed_search_stops_before_any_click(tmp_path):
    err = GroundingError("no verified match")
    err.trace_path = tmp_path / "trace.json"
    wf, events, *_ = make(tmp_path, searcher_script=[err])
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(2))
    assert ei.value.stage == "locate" and "no verified match" in str(ei.value) and "trace.json" in str(ei.value)
    assert events.count_of("double_click") == 0 and events.count_of("click") == 0
    assert events.count_of("wait_for_launch") == 0


@pytest.mark.parametrize("stage", ["close_existing", "wait_for_launch", "prepare_blank", "save_as", "close_document"])
def test_each_notepad_step_failure_is_reported_with_its_stage(tmp_path, stage):
    expected = {"close_existing": "preflight", "wait_for_launch": "launch", "prepare_blank": "blank",
                "save_as": "save", "close_document": "close"}[stage]
    wf, events, *_ = make(tmp_path, notepad_kwargs={"fail_at": stage})
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(2))
    assert ei.value.stage == expected and ei.value.post_id == 1
    assert events.count_of("find") <= 1  # never moved on to post 2


def test_failsafe_is_never_swallowed(tmp_path):
    wf, events, screen, *_ = make(tmp_path)
    screen.double_click = lambda p: (_ for _ in ()).throw(pyautogui.FailSafeException("corner"))
    with pytest.raises(pyautogui.FailSafeException):
        wf.run(posts(2))
    assert events.count_of("wait_for_launch") == 0


# ---- pop-ups ----------------------------------------------------------------------------

def _popup_block(x=900, y=300):
    a = PopupAssessment(True, "close", "update banner", "Close", "top-right X", "blocks")
    return PopupBlocked(a, Point(x, y), BBox(x - 5, y - 5, x + 5, y + 5))


def test_popup_is_dismissed_then_the_desktop_is_searched_again_from_a_fresh_screenshot(tmp_path):
    wf, events, *_ = make(tmp_path, searcher_script=[_popup_block()])
    wf.run(posts(1))
    seq = [e for e in names(events) if e in ("show_desktop", "capture", "find", "click", "double_click")]
    assert seq == ["show_desktop", "capture", "find", "click",
                   "show_desktop", "capture", "find", "double_click"]
    assert [e[1] for e in events if e[0] == "click"] == [Point(900, 300)]  # only the verified control


def test_endless_popups_stop_the_run_after_the_dismissal_limit(tmp_path):
    wf, events, *_ = make(tmp_path, searcher_script=[_popup_block() for _ in range(10)])
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(1))
    assert ei.value.stage == "locate" and "pop-ups" in str(ei.value)
    assert events.count_of("click") == FAST.popup_dismissals and events.count_of("double_click") == 0


# ---- verifying the saved file -----------------------------------------------------------

def test_crlf_in_the_saved_file_is_fine(tmp_path):
    wf, *_ = make(tmp_path)
    wf.run(posts(1))  # FakeNotepad writes CRLF


def test_dropped_characters_in_the_saved_file_stop_the_run(tmp_path):
    wf, *_ = make(tmp_path, notepad_kwargs={"written": "Title: title 1\n\nbody line one\nbody line two"})
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(2))
    assert ei.value.stage == "verify" and "does not contain the expected text" in str(ei.value)
    assert ei.value.post_id == 1


def test_a_missing_file_stops_the_run(tmp_path):
    wf, *_ = make(tmp_path, notepad_kwargs={"skip_write": True})
    with pytest.raises(WorkflowError) as ei:
        wf.run(posts(1))
    assert ei.value.stage == "verify" and "was not created" in str(ei.value)


def test_notepad_is_not_closed_when_verification_fails(tmp_path):
    wf, events, *_ = make(tmp_path, notepad_kwargs={"written": "wrong"})
    with pytest.raises(WorkflowError):
        wf.run(posts(1))
    assert events.count_of("close_document") == 0  # left open for the user to inspect
