import pytest
from PIL import Image

import vision_automation.inputs as inputs
import vision_automation.workflow as workflow
from vision_automation.config import CONFIG
from vision_automation.screen import neutral_cursor_spot
from vision_automation.types import BBox, Point

SCREEN = (1920, 1080)


# ---- where to park ----------------------------------------------------------------------

def test_default_windows_11_taskbar_parks_on_blank_taskbar_space():
    spot = neutral_cursor_spot(SCREEN, BBox(0, 0, 1920, 1032))  # 48px taskbar at the bottom
    assert spot == Point(480, 1056)  # in the band, a quarter along: left of the centered app icons


def test_other_dock_edges():
    assert neutral_cursor_spot(SCREEN, BBox(0, 48, 1920, 1080)) == Point(480, 24)       # top
    assert neutral_cursor_spot(SCREEN, BBox(60, 0, 1920, 1080)) == Point(30, 270)       # left
    assert neutral_cursor_spot(SCREEN, BBox(0, 0, 1860, 1080)) == Point(1890, 270)      # right


@pytest.mark.parametrize("work", [None, BBox(0, 0, 1920, 1080)])
def test_no_taskbar_band_parks_near_the_bottom_center_away_from_edges(work):
    spot = neutral_cursor_spot(SCREEN, work)  # e.g. auto-hide: touching the edge would reveal the taskbar
    assert spot == Point(960, 972)
    assert 100 < spot.y < SCREEN[1] - 100


def test_the_spot_is_never_a_corner_or_off_screen():
    for work in (None, BBox(0, 0, 1920, 1032), BBox(0, 48, 1920, 1080), BBox(60, 0, 1920, 1080), BBox(0, 0, 1860, 1080)):
        p = neutral_cursor_spot(SCREEN, work)
        assert 1 <= p.x < SCREEN[0] - 1 and 1 <= p.y < SCREEN[1] - 1
        assert (round(p.x), round(p.y)) not in {(0, 0), (1919, 0), (0, 1079), (1919, 1079)}


# ---- parking is mouse movement only -----------------------------------------------------

class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append((name, a, k))
        return record


@pytest.fixture
def fake_pyautogui(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(inputs, "pyautogui", rec)
    monkeypatch.setattr(inputs, "screen_size", lambda: SCREEN)
    return rec


def test_park_cursor_only_moves_the_pointer(fake_pyautogui):
    inputs.park_cursor(Point(480.4, 1056.2))
    assert [c[0] for c in fake_pyautogui.calls] == ["moveTo"]  # no click, doubleClick, press, write, hotkey...
    assert fake_pyautogui.calls[0][1] == (480, 1056)


@pytest.mark.parametrize("p", [Point(0, 0), Point(1919, 0), Point(0, 1079), Point(1919, 1079), Point(-5, 10), Point(2500, 10)])
def test_park_cursor_refuses_corners_and_off_screen_points(fake_pyautogui, p):
    with pytest.raises(inputs.OffScreenError):
        inputs.park_cursor(p)
    assert fake_pyautogui.calls == []


# ---- the workflow parks before EVERY screenshot, then waits, then captures --------------

def test_fresh_screenshot_parks_waits_for_tooltips_then_captures(monkeypatch):
    events = []
    monkeypatch.setattr(workflow, "screen_size", lambda: SCREEN)
    monkeypatch.setattr(workflow, "work_area", lambda: BBox(0, 0, 1920, 1032))
    monkeypatch.setattr(workflow.inputs, "park_cursor", lambda p: events.append(("park", p)))
    monkeypatch.setattr(workflow.time, "sleep", lambda s: events.append(("sleep", s)))
    monkeypatch.setattr(workflow, "wait_until_stable",
                        lambda timeout: events.append(("capture", timeout)) or Image.new("RGB", (4, 4)))

    img = workflow.LiveScreen(CONFIG).fresh_screenshot()
    assert img.size == (4, 4)
    assert events == [("park", Point(480, 1056)), ("sleep", CONFIG.park_settle_s), ("capture", CONFIG.stable_timeout_s)]


def test_every_call_parks_again(monkeypatch):
    parks = []
    monkeypatch.setattr(workflow, "screen_size", lambda: SCREEN)
    monkeypatch.setattr(workflow, "work_area", lambda: None)
    monkeypatch.setattr(workflow.inputs, "park_cursor", parks.append)
    monkeypatch.setattr(workflow.time, "sleep", lambda s: None)
    monkeypatch.setattr(workflow, "wait_until_stable", lambda timeout: Image.new("RGB", (4, 4)))
    screen = workflow.LiveScreen(CONFIG)
    for _ in range(3):
        screen.fresh_screenshot()
    assert len(parks) == 3


def test_settle_time_is_short_but_nonzero():
    assert 0.2 <= CONFIG.park_settle_s <= 1.0
