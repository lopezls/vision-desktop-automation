"""Planner: pop-up check and position inference (text only, no coordinates).

Position inference follows the paper's Appendix C prompt (Table 7); see prompts/planner.md and
the change notes in PRESENCE_NOTE_* below. The popup check is not in the paper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from .llm import LLM, LLMError

log = logging.getLogger("vision_automation")

PROMPT_DIR = Path(__file__).parent / "prompts"

# Verbatim from Appendix C for the full-screenshot level.
PRESENCE_NOTE_FULL = "The target UI element is guaranteed to be present in the screenshot."
# Changed for crop levels: a crop may legitimately not contain the target, and the paper's
# guarantee would push the model to invent one instead of answering "No target".
PRESENCE_NOTE_CROP = (
    "The target UI element may or may not be present in this cropped screenshot. "
    "If it is not, output \"No target\"."
)

HintKind = Literal["element", "area", "neighbor"]


@dataclass(frozen=True)
class Hint:
    kind: HintKind
    text: str
    rank: int  # 0 = most likely, in the order the model listed them


@dataclass(frozen=True)
class PositionInference:
    hints: list[Hint] = field(default_factory=list)
    no_target: bool = False
    raw: str = ""

    def of_kind(self, *kinds: HintKind) -> list[Hint]:
        return [h for h in self.hints if h.kind in kinds]


# ---- tolerant parsing of the tagged reply ----------------------------------------------

_TAGS = "element|area|neighbor"
_CLOSED = re.compile(rf"<\s*({_TAGS})\s*>(.*?)<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED = re.compile(rf"<\s*({_TAGS})\s*>([^<\n]+)", re.IGNORECASE)
_NO_TARGET = re.compile(r"\bno\s+target\b", re.IGNORECASE)


def parse_position_reply(text: str) -> PositionInference:
    """Extract <element>/<area>/<neighbor> items in order of appearance.

    Tolerates case differences, whitespace inside tags, missing closing tags and surrounding
    prose. A reply with no parsable items that says "No target" (or has nothing usable at all)
    is a failed level, not an error.
    """
    found = [(m.start(), m.group(1).lower(), m.group(2)) for m in _CLOSED.finditer(text)]
    closed_spans = [(m.start(), m.end()) for m in _CLOSED.finditer(text)]
    for m in _UNCLOSED.finditer(text):
        if not any(s <= m.start() < e for s, e in closed_spans):
            found.append((m.start(), m.group(1).lower(), m.group(2)))
    found.sort(key=lambda t: t[0])

    hints: list[Hint] = []
    seen: set[tuple[str, str]] = set()
    for _, kind, raw in found:
        cleaned = " ".join(raw.split()).strip(" .,;:\"'")
        key = (kind, cleaned.lower())
        if not cleaned or key in seen:
            continue
        seen.add(key)
        hints.append(Hint(kind, cleaned, len(hints)))  # type: ignore[arg-type]

    no_target = not hints  # nothing usable: either an explicit "No target" or garbage
    if no_target and not _NO_TARGET.search(text):
        log.warning("[plan] reply had no tagged items and no 'No target': %r", text[:200])
    return PositionInference(hints=hints, no_target=no_target, raw=text)


# ---- popup assessment -------------------------------------------------------------------

PopupAction = Literal["none", "close", "abort"]

# Exact (normalized) labels the program is allowed to click on a pop-up. Anything else,
# including "OK", "Got it", "Yes", "Allow", "Accept", is refused, even if the model likes it.
SAFE_DISMISS_LABELS = frozenset({
    "x", "×", "✕", "✖", "╳", "close", "dismiss", "cancel",
    "not now", "no thanks", "no thank you", "maybe later", "remind me later",
})


def normalize_label(label: str | None) -> str:
    return " ".join((label or "").lower().replace("✕", "✕").strip(" .!\"'[]()").split())


def is_safe_dismiss_label(label: str | None) -> bool:
    return normalize_label(label) in SAFE_DISMISS_LABELS


@dataclass(frozen=True)
class PopupAssessment:
    blocked: bool
    action: PopupAction = "none"
    popup: str = ""
    control_label: str | None = None
    control_description: str | None = None
    reason: str = ""


def parse_popup_reply(data: dict) -> PopupAssessment:
    """Validate the model's popup reply and enforce the dismiss-only rule in code."""
    blocked = data.get("blocked") is True
    action = data.get("action")
    popup = str(data.get("popup") or "")
    reason = str(data.get("reason") or "")
    label = data.get("control_label")
    label = label if isinstance(label, str) and label.strip() else None
    desc = data.get("control_description")
    desc = desc if isinstance(desc, str) and desc.strip() else None

    if not blocked:
        return PopupAssessment(False, "none", popup, reason=reason or "nothing blocking")
    if action == "close":
        if not is_safe_dismiss_label(label):
            return PopupAssessment(True, "abort", popup, label, desc,
                                   f"refusing to click {label!r}: not a close/dismiss control. {reason}")
        if not desc:
            return PopupAssessment(True, "abort", popup, label, desc,
                                   f"no description of where the {label!r} control is. {reason}")
        return PopupAssessment(True, "close", popup, label, desc, reason)
    # blocked but action is "abort", "none" or garbage: never click anything
    return PopupAssessment(True, "abort", popup, label, desc, reason or "blocked and no safe way to dismiss")


# ---- planner ----------------------------------------------------------------------------

class Planner(Protocol):
    def check_popup(self, screenshot: Image.Image, description: str) -> PopupAssessment: ...
    def infer_positions(self, image: Image.Image, description: str, *, is_crop: bool = False) -> PositionInference: ...


class ClaudePlanner:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.position_prompt = (PROMPT_DIR / "planner.md").read_text(encoding="utf-8")
        self.popup_prompt = (PROMPT_DIR / "popup.md").read_text(encoding="utf-8")

    def check_popup(self, screenshot: Image.Image, description: str) -> PopupAssessment:
        prompt = self.popup_prompt.replace("{instruction}", description)
        try:
            data = self.llm.ask_json("popup", prompt, [screenshot])
        except LLMError as e:
            # Can't tell: refuse to proceed rather than click blind.
            log.warning("[popup] check failed: %s", e)
            return PopupAssessment(True, "abort", reason=f"popup check failed: {e}")
        result = parse_popup_reply(data)
        log.info("[popup] blocked=%s action=%s label=%r popup=%r reason=%s", result.blocked,
                 result.action, result.control_label, result.popup, result.reason)
        return result

    def infer_positions(self, image: Image.Image, description: str, *, is_crop: bool = False) -> PositionInference:
        note = PRESENCE_NOTE_CROP if is_crop else PRESENCE_NOTE_FULL
        prompt = self.position_prompt.replace("{presence_note}", note).replace("{instruction}", description)
        try:
            text = self.llm.ask("plan", prompt, [image])
        except LLMError as e:
            log.warning("[plan] failed: %s", e)
            return PositionInference(no_target=True, raw=f"error: {e}")
        result = parse_position_reply(text)
        if result.no_target and not _NO_TARGET.search(text):
            self.llm.note_parse_failure("plan", f"no tagged items and no 'No target': {text[:150]!r}")
        if result.no_target:
            log.info("[plan] 'No target' / nothing usable at this level")
        else:
            log.info("[plan] hints: %s", "; ".join(f"{h.kind}:{h.text}" for h in result.hints))
        return result
