"""Mouse/keyboard wrappers. pyautogui's failsafe is ON: move the mouse into a screen corner to abort."""

import time

import pyautogui

from .screen import screen_size, set_dpi_aware
from .types import Point

set_dpi_aware()
pyautogui.FAILSAFE = True  # never disable
pyautogui.PAUSE = 0.1

TYPE_INTERVAL = 0.03  # 0.01 dropped characters in Notepad on this machine


class OffScreenError(ValueError):
    pass


def _check_on_screen(p: Point) -> tuple[int, int]:
    w, h = screen_size()
    x, y = round(p.x), round(p.y)
    # Stay off the corners themselves: a corner click would trigger the failsafe.
    if not (1 <= x < w - 1 and 1 <= y < h - 1):
        raise OffScreenError(f"({x}, {y}) is outside the {w}x{h} screen")
    return x, y


def click(p: Point) -> None:
    x, y = _check_on_screen(p)
    pyautogui.click(x, y)


def double_click(p: Point) -> None:
    x, y = _check_on_screen(p)
    pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.doubleClick()


def park_cursor(p: Point) -> None:
    """Move the pointer out of the way (movement only: no click, no key). Used before screenshots so a
    hover highlight or tooltip never covers the icon being searched for."""
    x, y = _check_on_screen(p)
    pyautogui.moveTo(x, y, duration=0.15)


def move(p: Point) -> None:
    x, y = _check_on_screen(p)
    pyautogui.moveTo(x, y, duration=0.2)


def type_text(text: str, interval: float = TYPE_INTERVAL, pause: bool = True) -> None:
    """Type text key by key. `pause=False` skips pyautogui's 0.1s after-call pause (used when typing many
    short chunks, each followed by a read-back that provides the settling time). The failsafe stays on."""
    pyautogui.write(text, interval=interval, _pause=pause)


def press_repeated(key: str, n: int, interval: float = 0.02) -> None:
    """Press one key n times (cursor movement / Backspace), failsafe on."""
    if n > 0:
        pyautogui.press(key, presses=n, interval=interval, _pause=False)


def press(*keys: str) -> None:
    for k in keys:
        pyautogui.press(k)


def hotkey(*keys: str) -> None:
    pyautogui.hotkey(*keys)


def pause(seconds: float) -> None:
    time.sleep(seconds)
