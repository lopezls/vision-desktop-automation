"""The post-saving workflow: for each post, find the Notepad icon from a FRESH screenshot, launch,
type, save, close. Stops at the first failure with a clear error; never guess-clicks.

Per post (nothing is carried over between posts, in particular no coordinates):
  preflight  close any already-open Notepad window (gracefully; never discards content)
  locate     show the desktop (Win+D) -> wait for a stable screenshot -> search + verify
             (a blocking pop-up is dismissed only via a verified close/dismiss control)
  launch     double-click the verified point -> wait for the NEW Notepad window
  blank      make sure the active tab is a blank document
  type       type "Title: {title}\\n\\n{body}"
  save       Save As -> Desktop/tjm-project/post_{id}.txt (overwrite confirmation handled)
  verify     the saved file must contain exactly the expected text
  close      close our tab / the window

pyautogui's failsafe stays on: moving the mouse into a screen corner raises FailSafeException,
which is never swallowed here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyautogui
from PIL import Image

from . import inputs
from .annotate import save_image
from .config import CONFIG, DEBUG_DIR, Config
from .llm import timed
from .posts import Post
from .screen import capture, neutral_cursor_spot, screen_size, wait_until_stable, work_area
from .search import GroundingError, Located, PopupBlocked, Searcher
from .types import Point
from .windows import WaitTimeout, is_desktop_foreground, wait_for

log = logging.getLogger("vision_automation")


class ScreenOps(Protocol):
    def show_desktop(self) -> None: ...
    def fresh_screenshot(self) -> Image.Image: ...
    def click(self, p: Point) -> None: ...
    def double_click(self, p: Point) -> None: ...
    def debug_screenshot(self) -> Image.Image: ...


class NotepadOps(Protocol):
    def close_existing(self) -> None: ...
    def snapshot(self) -> frozenset[int]: ...
    def wait_for_launch(self, before: frozenset[int]) -> int: ...
    def prepare_blank(self, hwnd: int) -> None: ...
    def type_text(self, hwnd: int, text: str) -> None: ...
    def save_as(self, hwnd: int, path: Path) -> None: ...
    def close_document(self, hwnd: int) -> None: ...


class WorkflowError(RuntimeError):
    def __init__(self, post_id: int | None, stage: str, message: str, debug_image: Path | None = None):
        where = f"post {post_id}, step '{stage}'" if post_id is not None else f"step '{stage}'"
        super().__init__(f"{where}: {message}")
        self.post_id, self.stage, self.detail, self.debug_image = post_id, stage, message, debug_image


@dataclass
class PostResult:
    post_id: int
    path: Path
    click_point: Point
    search_calls: int
    seconds: float


class LiveScreen:
    """The real screen: Win+D, stable screenshots, clicks (all through the failsafe-protected input module)."""

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg

    def show_desktop(self) -> None:
        # Win+D toggles, so only press it when the desktop is not already in front.
        if is_desktop_foreground():
            return
        inputs.hotkey("win", "d")
        wait_for(is_desktop_foreground, self.cfg.desktop_timeout_s, "the desktop to come to the front (Win+D)")

    def fresh_screenshot(self) -> Image.Image:
        """Park the pointer (movement only), let tooltips fade, then capture once the screen is still."""
        inputs.park_cursor(neutral_cursor_spot(screen_size(), work_area()))
        time.sleep(self.cfg.park_settle_s)
        with timed("capture"):
            return wait_until_stable(self.cfg.stable_timeout_s)

    def click(self, p: Point) -> None:
        inputs.click(p)

    def double_click(self, p: Point) -> None:
        inputs.double_click(p)

    def debug_screenshot(self) -> Image.Image:
        return capture()


def matches_expected(path: Path, expected: str) -> bool:
    try:
        text = path.read_bytes().decode("utf-8-sig", errors="replace")
    except OSError:
        return False
    return text.replace("\r\n", "\n") == expected


class Workflow:
    def __init__(self, searcher: Searcher, screen: ScreenOps, notepad: NotepadOps, out_dir: Path,
                 description: str, cfg: Config = CONFIG, debug_dir: Path = DEBUG_DIR):
        self.searcher, self.screen, self.notepad = searcher, screen, notepad
        self.out_dir, self.description, self.cfg, self.debug_dir = out_dir, description, cfg, debug_dir
        self.stage = "start"

    # -- public --

    def run(self, posts: list[Post]) -> list[PostResult]:
        results = []
        for i, post in enumerate(posts, 1):
            log.info("=== post %d/%d (id %d) ===", i, len(posts), post.id)
            results.append(self.run_post(post))  # raises WorkflowError: the first failure stops everything
        return results

    def run_post(self, post: Post) -> PostResult:
        t0 = time.perf_counter()
        path = self.out_dir / post.filename
        try:
            self._enter("preflight")
            self.notepad.close_existing()

            self._enter("locate")
            located = self._locate()

            self._enter("launch")
            before = self.notepad.snapshot()
            with timed("launch"):
                self.screen.double_click(located.point)
                hwnd = self.notepad.wait_for_launch(before)

            self._enter("blank")
            self.notepad.prepare_blank(hwnd)

            self._enter("type")
            with timed("type"):
                self.notepad.type_text(hwnd, post.content)

            self._enter("save")
            with timed("save"):
                self.notepad.save_as(hwnd, path)

            self._enter("verify")
            self._verify_saved(path, post.content)

            self._enter("close")
            self.notepad.close_document(hwnd)
        except pyautogui.FailSafeException:
            raise  # the user pulled the failsafe: stop at once, add nothing
        except WorkflowError:
            raise
        except Exception as e:  # any other failure stops the run with the stage it happened in
            raise WorkflowError(post.id, self.stage, _describe(e), self._save_failure_image(post.id)) from e

        result = PostResult(post.id, path, located.point, located.calls, round(time.perf_counter() - t0, 1))
        log.info("post %d saved to %s (click point (%.0f, %.0f), %d search calls, %.1fs)", post.id, path,
                 located.point.x, located.point.y, located.calls, result.seconds)
        return result

    # -- steps --

    def _enter(self, stage: str) -> None:
        self.stage = stage
        log.info("[workflow] %s", stage)

    def _locate(self) -> Located:
        """Desktop -> fresh screenshot -> search, repeated after dismissing a verified pop-up control."""
        for attempt in range(self.cfg.popup_dismissals + 1):
            self.screen.show_desktop()
            shot = self.screen.fresh_screenshot()  # a new capture for every search; coordinates are never reused
            try:
                return self.searcher.find(shot, self.description)
            except PopupBlocked as pb:
                if attempt >= self.cfg.popup_dismissals:
                    raise GroundingError(f"still blocked by pop-ups after {attempt} dismissals: {pb}") from None
                log.warning("[workflow] dismissing pop-up via %r at (%.0f, %.0f)", pb.assessment.control_label,
                            pb.control_point.x, pb.control_point.y)
                self.screen.click(pb.control_point)  # verified close/dismiss control from this same screenshot
        raise AssertionError("unreachable")

    def _verify_saved(self, path: Path, expected: str) -> None:
        try:
            wait_for(lambda: matches_expected(path, expected), self.cfg.dialog_timeout_s,
                     f"{path.name} to contain the expected text")
        except WaitTimeout:
            actual = ""
            try:
                actual = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            except OSError:
                pass
            if not path.exists():
                raise RuntimeError(f"{path} was not created") from None
            raise RuntimeError(
                f"{path.name} does not contain the expected text (typed keys may have been dropped or the "
                f"document was not blank). Expected {len(expected)} chars, found {len(actual)}. "
                f"First difference at index {_first_diff(expected, actual)}.") from None

    def _save_failure_image(self, post_id: int) -> Path | None:
        try:
            return save_image(self.screen.debug_screenshot(), f"workflow_failed_post{post_id}_{self.stage}",
                              self.debug_dir)
        except Exception as e:
            log.warning("could not save a failure screenshot: %s", e)
            return None


def _describe(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    extra = [f"{label}: {getattr(e, attr)}" for attr, label in (("trace_path", "search trace"),
                                                                ("image_path", "search debug image"))
             if getattr(e, attr, None)]
    return msg + (f" ({'; '.join(extra)})" if extra else "")


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
