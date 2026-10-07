"""Grounder: image + free-text query -> bounding box. No application-specific logic.

The query is whatever the caller passes (a planner hint such as "the icon column on the left",
or the final target description). The prompt in prompts/grounder.md is fully generic.

Coordinates: the model answers in PIXELS of the image it was shown (an earlier 0-1000
"normalized" convention was ignored by the model and silently mis-scaled results). To keep
that unambiguous we downscale the image ourselves to at most MAX_SENT_SIDE (so the API never
resizes it behind our back), tell the model the exact pixel size, and map the reply back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from .llm import LLM, LLMError
from .types import BBox

log = logging.getLogger("vision_automation")

PROMPT_PATH = Path(__file__).parent / "prompts" / "grounder.md"
MIN_BOX_PX = 4  # smaller than this on either side (in sent-image pixels) is degenerate
MAX_SENT_SIDE = 1568  # longest side we send; larger images are downscaled by us, not the API
OVERSHOOT_PX = 8  # tolerate a box that pokes slightly outside the image


@dataclass(frozen=True)
class GroundResult:
    found: bool
    box: BBox | None = None  # pixels of the image that was passed to ground()
    confidence: float | None = None
    note: str = ""
    invalid: bool = False  # the reply was malformed/implausible (counted as a parse failure)


class Grounder(Protocol):
    def ground(self, image: Image.Image, query: str) -> GroundResult: ...


def prepare_for_model(image: Image.Image) -> tuple[Image.Image, float]:
    """Downscale so the longer side <= MAX_SENT_SIDE. Returns (image to send, sent/original scale)."""
    long_side = max(image.width, image.height)
    if long_side <= MAX_SENT_SIDE:
        return image, 1.0
    scale = MAX_SENT_SIDE / long_side
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    sent = image.resize(size, Image.LANCZOS)
    return sent, sent.width / image.width


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_ground_reply(data: dict, sent_w: int, sent_h: int, scale: float = 1.0) -> GroundResult:
    """Validate a model reply (pixels of the SENT image) and map it to original-image pixels.

    Anything malformed or implausible becomes found=False rather than an exception, so the
    search treats it as a failed candidate instead of crashing.
    """
    note = str(data.get("note") or "")
    conf = data.get("confidence")
    conf = float(conf) if _number(conf) else None
    if not data.get("found"):
        return GroundResult(False, None, conf, note or "model reports not found")

    raw = data.get("box")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4 and all(_number(v) for v in raw)):
        return GroundResult(False, None, conf, f"malformed box {raw!r}", invalid=True)
    x0, x1 = sorted((raw[0], raw[2]))
    y0, y1 = sorted((raw[1], raw[3]))
    if (x0 < -OVERSHOOT_PX or y0 < -OVERSHOOT_PX
            or x1 > sent_w + OVERSHOOT_PX or y1 > sent_h + OVERSHOOT_PX):
        return GroundResult(False, None, conf, f"box outside the {sent_w}x{sent_h} image {raw!r}", invalid=True)

    sent_box = BBox(x0, y0, x1, y1).clamp(BBox(0, 0, sent_w, sent_h))
    if sent_box.width < MIN_BOX_PX or sent_box.height < MIN_BOX_PX:
        return GroundResult(False, None, conf, f"degenerate box {raw!r}", invalid=True)
    box = BBox(sent_box.x0 / scale, sent_box.y0 / scale, sent_box.x1 / scale, sent_box.y1 / scale)
    return GroundResult(True, box, conf, note)


class ClaudeGrounder:
    def __init__(self, llm: LLM, stage: str = "ground"):
        self.llm = llm
        self.stage = stage
        self.template = PROMPT_PATH.read_text(encoding="utf-8")

    def ground(self, image: Image.Image, query: str) -> GroundResult:
        sent, scale = prepare_for_model(image)
        system = self.template.replace("{width}", str(sent.width)).replace("{height}", str(sent.height))
        prompt = f"Description of the element to find: {query}"
        log.info("[ground] query=%r image=%dx%d sent=%dx%d", query, image.width, image.height,
                 sent.width, sent.height)
        try:
            data = self.llm.ask_json(self.stage, prompt, [sent], system=system)
        except LLMError as e:
            log.warning("[ground] failed: %s", e)
            return GroundResult(False, None, None, f"grounder error: {e}")
        result = parse_ground_reply(data, sent.width, sent.height, scale)
        if result.invalid:
            self.llm.note_parse_failure(self.stage, result.note)
        if result.found:
            c = result.box.center
            log.info("[ground] %r -> box=%s center=(%.0f, %.0f) conf=%s note=%r", query,
                     result.box.as_ints(), c.x, c.y, result.confidence, result.note)
        else:
            log.info("[ground] %r -> not found (%s)", query, result.note)
        return result
