import logging

import pytest
import requests
from PIL import Image

from vision_automation.notepad import DIALOG_CLASS, is_blank_title, notepad_windows
from vision_automation.posts import Post, PostsError, fetch_posts, untypeable_chars
from vision_automation.screen import wait_until_stable
from vision_automation.windows import WaitTimeout, WinInfo, list_windows, wait_for, window_exists


# ---- posts ------------------------------------------------------------------------------

class Resp:
    def __init__(self, data=None, status=200, bad_json=False):
        self._data, self.status_code, self._bad = data, status, bad_json

    def json(self):
        if self._bad:
            raise requests.exceptions.JSONDecodeError("bad", "doc", 0)
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def api(n=12):
    return [{"userId": 1, "id": i, "title": f"t{i}", "body": f"b{i}\nmore"} for i in range(1, n + 1)]


def test_post_format_and_filename():
    p = Post(7, "a title", "line1\nline2")
    assert p.content == "Title: a title\n\nline1\nline2" and p.filename == "post_7.txt"


def test_fetch_takes_the_first_n_in_api_order():
    posts = fetch_posts(10, get=lambda url, timeout: Resp(api(100)))
    assert [p.id for p in posts] == list(range(1, 11))


def test_fetch_retries_then_succeeds():
    seq = [requests.ConnectionError("down"), Resp(status=503), Resp(api())]
    waits = []

    def get(url, timeout):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    posts = fetch_posts(3, get=get, sleep=waits.append)
    assert len(posts) == 3 and waits == [1.0, 2.0]


def test_fetch_gives_up_after_three_attempts_with_a_clear_error():
    def get(url, timeout):
        raise requests.Timeout("slow")

    with pytest.raises(PostsError, match="after 3 attempts"):
        fetch_posts(3, get=get, sleep=lambda s: None)


@pytest.mark.parametrize("payload", [{"not": "a list"}, [{"id": "x", "title": "t", "body": "b"}], api(2)])
def test_fetch_rejects_bad_payloads(payload):
    with pytest.raises(PostsError):
        fetch_posts(5, get=lambda url, timeout: Resp(payload))


def test_fetch_does_not_retry_unreadable_json():
    calls = []

    def get(url, timeout):
        calls.append(1)
        return Resp(bad_json=True)

    with pytest.raises(PostsError, match="could not read"):
        fetch_posts(1, get=get, sleep=lambda s: None)
    assert len(calls) == 1


def test_untypeable_characters_are_caught_before_anything_is_clicked():
    assert untypeable_chars("plain\ntext, with punctuation!") == []
    assert untypeable_chars("café — tab\t") == ["\t", "é", "—"]
    bad = [{"id": 1, "title": "café", "body": "x"}]
    with pytest.raises(PostsError, match="cannot be typed"):
        fetch_posts(1, get=lambda url, timeout: Resp(bad))


# ---- notepad pure helpers ---------------------------------------------------------------

@pytest.mark.parametrize("title,blank", [
    ("Untitled - Notepad", True), ("untitled - notepad", True), ("*Untitled - Notepad", False),
    ("post_3.txt - Notepad", False), ("*post_3.txt - Notepad", False), ("", False), ("Notepad", False)])
def test_is_blank_title(title, blank):
    assert is_blank_title(title) is blank


def test_notepad_windows_filters_by_process_not_title():
    wins = [
        WinInfo(1, "Untitled - Notepad", "WinUIDesktopWin32WindowClass", "notepad.exe", 10, True),
        WinInfo(2, "Save As", DIALOG_CLASS, "notepad.exe", 10, True),           # a Notepad dialog, not a window
        WinInfo(3, "notes.txt - Notepad++", "Notepad++", "notepad++.exe", 11, True),  # another app
        WinInfo(4, "", "Hidden", "notepad.exe", 10, True),                       # untitled helper window
        WinInfo(5, "Document - Notepad", "Notepad", "notepad.exe", 12, True),
    ]
    assert [w.hwnd for w in notepad_windows(lambda: wins)] == [1, 5]


# ---- windows helpers --------------------------------------------------------------------

def test_wait_for_returns_the_value_and_times_out():
    assert wait_for(lambda: "ok", 1, "x") == "ok"
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t["now"] += s

    with pytest.raises(WaitTimeout, match="waiting for a thing"):
        wait_for(lambda: None, 0.35, "a thing", interval=0.1, sleep=sleep, clock=lambda: t["now"])
    assert len(sleeps) >= 3


def test_wait_for_polls_until_the_condition_holds():
    state = {"n": 0}

    def pred():
        state["n"] += 1
        return state["n"] >= 3

    assert wait_for(pred, 5, "x", interval=0, sleep=lambda s: None) is True and state["n"] == 3


def test_list_windows_returns_typed_records_and_bad_handles_do_not_exist():
    wins = list_windows()
    assert all(isinstance(w, WinInfo) for w in wins)
    assert not window_exists(0)


# ---- stable screenshots -----------------------------------------------------------------

def _img(color, dot=None):
    im = Image.new("RGB", (200, 100), color)
    if dot:
        im.paste((255, 255, 255), dot)
    return im


def test_wait_until_stable_returns_once_two_captures_match():
    frames = [_img("black"), _img("blue"), _img("red"), _img("red")]
    got = wait_until_stable(capture_fn=lambda: frames.pop(0), sleep=lambda s: None)
    assert got.getpixel((0, 0)) == (255, 0, 0) and frames == []


def test_a_ticking_clock_does_not_count_as_motion():
    frames = [_img("black"), _img("black", (0, 0, 10, 4))]  # 40px of 20000 = 0.2%
    got = wait_until_stable(capture_fn=lambda: frames.pop(0), sleep=lambda s: None)
    assert frames == [] and got.size == (200, 100)


def test_wait_until_stable_gives_up_with_a_warning(caplog):
    t = {"now": 0.0}

    def sleep(s):
        t["now"] += s

    colors = iter(["black", "blue", "red", "green", "white", "gray", "navy", "teal", "olive", "lime"])
    with caplog.at_level(logging.WARNING, logger="vision_automation"):
        got = wait_until_stable(timeout=0.6, interval=0.25, capture_fn=lambda: _img(next(colors)),
                                sleep=sleep, clock=lambda: t["now"])
    assert got is not None and "still changing" in caplog.text
