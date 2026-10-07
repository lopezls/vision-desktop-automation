"""Verifier: is the element at the predicted point really the target?

Follows the paper's result-checking prompt (Appendix C, Table 8, prompts/verifier.md). The
model is shown a crop around the predicted point with a red box drawn on the candidate, not
the whole screen, and answers is_target / target_elsewhere / target_not_found.

Anything other than a clean is_target is a failure: unparsable replies and API errors become
target_not_found so the program can never click on an unverified guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from .annotate import RED, draw_box
from .config import CONFIG, Config
from .llm import LLM, LLMError, extract_json
from .screen import crop_region, image_bounds, window_around
from .types import BBox, Point

log = logging.getLogger("vision_automation")

PROMPT_PATH = Path(__file__).parent / "prompts" / "verifier.md"

Result = Literal["is_target", "target_elsewhere", "target_not_found"]
RESULTS = ("is_target", "target_elsewhere", "target_not_found")
FALLBACK_BOX_HALF_PX = 24  # red box half-size (screen px) when no grounded box is available


@dataclass(frozen=True)
class Verdict:
    result: Result
    new_instruction: str | None = None
    reason: str = ""
    marked: Image.Image | None = field(default=None, compare=False, repr=False)  # what the model saw

    @property
    def is_target(self) -> bool:
        return self.result == "is_target"


class Verifier(Protocol):
    def verify(self, screenshot: Image.Image, point: Point, description: str,
               box: BBox | None = None) -> Verdict: ...


def parse_verdict(data: dict) -> Verdict:
    raw = data.get("result")
    result = str(raw).strip().strip("'\"").lower() if isinstance(raw, str) else ""
    ni = data.get("new_instruction")
    new_instruction = ni.strip() if isinstance(ni, str) and ni.strip() else None
    if result not in RESULTS:
        return Verdict("target_not_found", None, f"unrecognised result {raw!r}")
    return Verdict(result, new_instruction)  # type: ignore[arg-type]


def make_marked_crop(screenshot: Image.Image, point: Point, box: BBox | None, cfg: Config = CONFIG) -> Image.Image:
    """Crop around `point` and draw the red box on the candidate (the model sees only this).

    The window is verify_crop_size screen-px square (shifted, not shrunk, near screen edges),
    upscaled so its longer side is view_long_side. The red box is the grounder's box when it
    overlaps the window, otherwise a small square around the point.
    """
    window = window_around(point, cfg.verify_crop_size, image_bounds(screenshot))
    crop = crop_region(screenshot, window, target_long_side=cfg.view_long_side)
    parent_box = None
    if box is not None:
        parent_box = box.intersection(crop.box)  # clip to the window
    if parent_box is None:
        h = FALLBACK_BOX_HALF_PX
        parent_box = BBox(point.x - h, point.y - h, point.x + h, point.y + h).intersection(crop.box)
    if parent_box is None:
        raise ValueError(f"point ({point.x:.0f}, {point.y:.0f}) is outside the {screenshot.size} image")
    local = BBox(
        (parent_box.x0 - crop.box.x0) * crop.scale, (parent_box.y0 - crop.box.y0) * crop.scale,
        (parent_box.x1 - crop.box.x0) * crop.scale, (parent_box.y1 - crop.box.y0) * crop.scale,
    )
    return draw_box(crop.image, local, RED, 3)


class ClaudeVerifier:
    def __init__(self, llm: LLM, cfg: Config = CONFIG):
        self.llm = llm
        self.cfg = cfg
        self.template = PROMPT_PATH.read_text(encoding="utf-8")

    def verify(self, screenshot: Image.Image, point: Point, description: str,
               box: BBox | None = None) -> Verdict:
        marked = make_marked_crop(screenshot, point, box, self.cfg)
        prompt = self.template.replace("{instruction}", description)
        log.info("[verify] point=(%.0f, %.0f) instruction=%r crop=%dx%d", point.x, point.y,
                 description, marked.width, marked.height)
        try:
            text = self.llm.ask("verify", prompt, [marked])
            verdict = parse_verdict(extract_json(text))
        except (LLMError, ValueError) as e:
            if isinstance(e, ValueError):
                self.llm.note_parse_failure("verify", str(e))
            log.warning("[verify] failed, treating as not found: %s", e)
            return Verdict("target_not_found", None, f"verifier error: {e}", marked)
        if verdict.reason.startswith("unrecognised"):
            self.llm.note_parse_failure("verify", verdict.reason)
        verdict = Verdict(verdict.result, verdict.new_instruction, verdict.reason, marked)
        log.info("[verify] -> %s%s", verdict.result,
                 f" (new_instruction={verdict.new_instruction!r})" if verdict.new_instruction else "")
        return verdict
