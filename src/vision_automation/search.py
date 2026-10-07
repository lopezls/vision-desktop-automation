"""Recursive visual search (ScreenSeekeR-style, arXiv 2504.07981 Algorithm 1) with verification.

find(screenshot, description):
  1. pop-up check (own addition): a blocking pop-up raises PopupBlocked carrying a *verified*
     dismiss-control point (this module never clicks; the caller decides), or GroundingError.
  2. _search(region): if the region is small enough (direct_ground_size) or max_depth is reached,
     ground the target directly and verify. Otherwise: planner hints (text) -> ground each
     area/neighbor hint -> dilate -> centrality score -> NMS -> recurse into the best candidates,
     backtracking to the next candidate when a branch fails.
  3. Guards: a per-find call cap, a stall guard (a candidate that is not meaningfully smaller than
     its parent, or revisits an already-tried region, is skipped), a depth limit. If a level runs out
     of candidates and its region is <= fallback_direct_size, the target is grounded there directly.
When the description says "not the taskbar", the taskbar band (screen minus the Windows work area) is
excluded geometrically: hints are grounded normally, then any box or candidate centered in the band is
rejected, candidates are clipped to the work area, and a final point in the band is refused.
Every find writes a JSON trace (hints, grounded boxes, scores, candidates, verdicts, parse
failures, stage timings) to debug/traces/, and failures also save an annotated screenshot.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image

from .annotate import RED, YELLOW, draw_box, draw_marker, save_image
from .config import CONFIG, DEBUG_DIR, TRACE_DIR, Config
from .grounder import Grounder
from .llm import reset_stage_times, stage_times
from .planner import Hint, Planner, PopupAssessment
from .scoring import dilate, iou, nms, score_candidates
from .screen import crop_region, image_bounds, work_area
from .types import BBox, Point
from .verifier import Verdict, Verifier

log = logging.getLogger("vision_automation")

REVISIT_IOU = 0.8
RESERVED_CALLS = 2  # keep room for a direct ground + verify when truncating hints at the cap


# ---- results and errors -----------------------------------------------------------------

@dataclass
class Located:
    point: Point  # click target, full-screenshot pixels
    box: BBox
    verdict: Verdict
    depth: int = 0
    calls: int = 0
    elapsed_s: float = 0.0
    trace_path: Path | None = None


class SearchError(RuntimeError):
    trace_path: Path | None = None
    image_path: Path | None = None
    calls: int = 0


class GroundingError(SearchError):
    """The target was not found and verified. Never fall back to guessing."""


class CallBudgetExceeded(GroundingError):
    """The per-find LLM call cap was reached."""


class PopupBlocked(SearchError):
    """A pop-up blocks the screen; `control_point` is a verified close/dismiss control to click."""

    def __init__(self, assessment: PopupAssessment, control_point: Point, control_box: BBox):
        super().__init__(f"pop-up {assessment.popup!r} blocks the screen; dismiss with "
                         f"{assessment.control_label!r} at ({control_point.x:.0f}, {control_point.y:.0f})")
        self.assessment = assessment
        self.control_point = control_point
        self.control_box = control_box


# ---- description-driven region exclusion ------------------------------------------------

_EXCLUDE_TASKBAR = re.compile(r"\bnot\b[^.;]{0,30}\btask\s?bar\b", re.IGNORECASE)
MIN_AREA_SIDE = 8  # px: a candidate clipped thinner than this is dropped


def excludes_taskbar(description: str) -> bool:
    """True for descriptions like 'the X icon (not the taskbar)' / 'not on the taskbar'.

    This only decides WHETHER the taskbar band is excluded. Which hints/boxes are affected is
    decided by geometry (see clip_to_work_area and Searcher), never by the words in a hint: a
    hint like "bottom-right, above the taskbar" is a perfectly good place to look.
    """
    return bool(_EXCLUDE_TASKBAR.search(description))


def clip_to_work_area(area: BBox, work: BBox | None) -> BBox | None:
    """Clip a candidate to the work area; None if its center is in the taskbar band or it is left too thin."""
    if work is None:
        return area
    if not work.contains(area.center):
        return None
    clipped = area.intersection(work)
    if clipped is None or clipped.width < MIN_AREA_SIDE or clipped.height < MIN_AREA_SIDE:
        return None
    return clipped


# ---- trace ------------------------------------------------------------------------------

def _b(box: BBox | None) -> list[int] | None:
    return list(box.as_ints()) if box is not None else None


class Trace:
    def __init__(self, description: str):
        self.description = description
        self.started = datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self.events: list[dict] = []

    def add(self, kind: str, **fields) -> None:
        self.events.append({"t": round(time.perf_counter() - self._t0, 2), "kind": kind, **fields})


@dataclass
class _Ctx:
    img: Image.Image
    cfg: Config
    trace: Trace
    calls: int = 0
    tried: list[tuple[BBox, str]] = field(default_factory=list)  # regions entered (for the failure image)
    rejected: list[Point] = field(default_factory=list)  # points the verifier rejected
    call_log_start: int = 0


@dataclass
class _State:
    """Per-target search state (the pop-up control search gets its own)."""
    description: str
    work: BBox | None = None  # when set, the taskbar band (outside this box) is excluded
    visited: list[BBox] = field(default_factory=list)


# ---- the searcher -----------------------------------------------------------------------

class Searcher:
    def __init__(self, planner: Planner, grounder: Grounder, verifier: Verifier, llm=None,
                 cfg: Config = CONFIG, trace_dir: Path = TRACE_DIR, debug_dir: Path = DEBUG_DIR,
                 work_area_fn: Callable[[tuple[int, int]], BBox | None] = work_area):
        self.planner, self.grounder, self.verifier, self.llm = planner, grounder, verifier, llm
        self.cfg, self.trace_dir, self.debug_dir = cfg, trace_dir, debug_dir
        self.work_area_fn = work_area_fn  # screenshot size -> work area (screen minus taskbar) or None

    # -- public --

    def find(self, screenshot: Image.Image, description: str, *, check_popup: bool = True) -> Located:
        reset_stage_times()
        t0 = time.perf_counter()
        pf_start = len(self.llm.parse_failures) if self.llm else 0
        ctx = _Ctx(screenshot, self.cfg, Trace(description),
                   call_log_start=len(getattr(self.llm, "call_log", [])) if self.llm else 0)
        log.info("[search] find %r (cap %d calls, direct size %dpx, max depth %d)", description,
                 self.cfg.max_llm_calls_per_find, self.cfg.direct_ground_size, self.cfg.max_depth)
        try:
            located = self._run(ctx, description, check_popup)
        except SearchError as e:
            outcome = "popup_blocked" if isinstance(e, PopupBlocked) else "failed"
            e.calls = ctx.calls
            e.trace_path = self._finish(ctx, outcome, t0, pf_start, error=str(e))
            if isinstance(e, GroundingError):
                e.image_path = self._save_failure_image(ctx)
            raise
        path = self._finish(ctx, "found", t0, pf_start, result=located)
        return replace(located, calls=ctx.calls, elapsed_s=round(time.perf_counter() - t0, 2), trace_path=path)

    # -- orchestration --

    def _run(self, ctx: _Ctx, description: str, check_popup: bool) -> Located:
        if check_popup:
            self._handle_popup(ctx, description)
        st = _State(description, work=self._taskbar_exclusion(ctx, description))
        located = self._search(ctx, st, image_bounds(ctx.img), 0)
        if located is None:
            raise GroundingError(f"no verified match for {description!r}")
        return located

    def _taskbar_exclusion(self, ctx: _Ctx, description: str) -> BBox | None:
        """The work area to confine the search to, or None when no exclusion applies/can be determined."""
        if not excludes_taskbar(description):
            return None
        work = self.work_area_fn(ctx.img.size)
        full = image_bounds(ctx.img)
        if work is None:
            ctx.trace.add("taskbar_exclusion", active=False, reason="work area unavailable for this image")
            return None
        if work.x0 <= full.x0 and work.y0 <= full.y0 and work.x1 >= full.x1 and work.y1 >= full.y1:
            ctx.trace.add("taskbar_exclusion", active=False, reason="no taskbar band reserved (auto-hide?)",
                          work_area=_b(work))
            return None
        ctx.trace.add("taskbar_exclusion", active=True, work_area=_b(work), screen=list(ctx.img.size))
        return work

    def _charge(self, ctx: _Ctx, n: int, what: str) -> None:
        cap = ctx.cfg.max_llm_calls_per_find
        if ctx.calls + n > cap:
            ctx.trace.add("call_cap", used=ctx.calls, wanted=n, cap=cap, stage=what)
            raise CallBudgetExceeded(f"LLM call cap ({cap}) reached before {what}")
        ctx.calls += n

    def _handle_popup(self, ctx: _Ctx, description: str) -> None:
        self._charge(ctx, 1, "popup")
        a = self.planner.check_popup(ctx.img, description)
        ctx.trace.add("popup", blocked=a.blocked, action=a.action, popup=a.popup,
                      control_label=a.control_label, control_description=a.control_description,
                      reason=a.reason)
        if not a.blocked:
            return
        if a.action != "close":
            raise GroundingError(f"a pop-up blocks the screen and has no safe dismiss control: "
                                 f"{a.popup!r} ({a.reason})")
        control = (f"the '{a.control_label}' close/dismiss control of the pop-up "
                   f"({a.popup}): {a.control_description}")
        ctx.trace.add("popup_control_search", description=control)
        found = self._search(ctx, _State(control), image_bounds(ctx.img), 0)
        if found is None:
            raise GroundingError(f"could not locate and verify the {a.control_label!r} control of pop-up {a.popup!r}")
        raise PopupBlocked(a, found.point, found.box)

    # -- recursion --

    def _search(self, ctx: _Ctx, st: _State, region: BBox, depth: int) -> Located | None:
        cfg = ctx.cfg
        long_side = max(region.width, region.height)
        if long_side <= cfg.direct_ground_size or depth >= cfg.max_depth:
            return self._direct(ctx, st, region, depth)

        crop = crop_region(ctx.img, region, target_long_side=max(round(long_side), cfg.view_long_side))
        self._charge(ctx, 1, "plan")
        inf = self.planner.infer_positions(crop.image, st.description, is_crop=depth > 0)
        ctx.trace.add("hints", depth=depth, region=_b(region), no_target=inf.no_target,
                      hints=[{"kind": h.kind, "text": h.text, "rank": h.rank} for h in inf.hints])
        if inf.no_target:
            ctx.trace.add("level_failed", depth=depth, region=_b(region), reason="'No target' or no usable hints")
            return None

        hints = inf.of_kind("area", "neighbor")  # planner order = descending probability; trust it
        allowed = cfg.max_llm_calls_per_find - ctx.calls - RESERVED_CALLS
        if allowed < 1:
            ctx.trace.add("call_cap", used=ctx.calls, cap=cfg.max_llm_calls_per_find, stage="hint grounding")
            raise CallBudgetExceeded(f"LLM call cap ({cfg.max_llm_calls_per_find}) leaves no room to ground hints")
        hints = hints[: min(cfg.max_hints_per_level, allowed)]
        if not hints:
            ctx.trace.add("level_failed", depth=depth, region=_b(region), reason="no area/neighbor hints left")
            return None

        self._charge(ctx, len(hints), "hint grounding")
        with ThreadPoolExecutor(max_workers=min(4, len(hints))) as pool:
            results = list(pool.map(lambda h: self.grounder.ground(crop.image, h.text), hints))

        boxes: list[BBox] = []
        for h, r in zip(hints, results):
            full = crop.box_to_parent(r.box).clamp(region) if r.found and r.box else None
            ok = full is not None and full.is_valid
            rejected = None
            if ok and st.work is not None and not st.work.contains(full.center):
                ok, rejected = False, "box center is inside the taskbar band"
            ctx.trace.add("grounded", depth=depth, hint=f"{h.kind}: {h.text}", found=ok,
                          box=_b(full) if full is not None and full.is_valid else None,
                          rejected=rejected, invalid_reply=r.invalid, note=r.note)
            if ok:
                boxes.append(full)
        if not boxes:
            ctx.trace.add("level_failed", depth=depth, region=_b(region), reason="no hint could be grounded")
            return None

        areas, kept_boxes = [], []
        for b in boxes:
            area = clip_to_work_area(self._candidate_area(b, region), st.work)
            if area is None:
                ctx.trace.add("candidate_skipped", depth=depth, box=_b(b), reason="candidate lies in the taskbar band")
                continue
            areas.append(area)
            kept_boxes.append(b)
        if not areas:
            ctx.trace.add("level_failed", depth=depth, region=_b(region), reason="every candidate was in the taskbar band")
            return None
        boxes = kept_boxes
        scores = score_candidates(areas, boxes, cfg.sigma)  # every grounded box votes, stalled candidates included

        # Stall guard BEFORE NMS: a candidate that is not meaningfully smaller than its region cannot be
        # searched, so it must not get to suppress candidates that can be. (A wide tooltip hint once made a
        # region-sized candidate the NMS winner, which then got skipped, leaving nothing to search.)
        usable: list[tuple[BBox, float]] = []
        for area, score in zip(areas, scores):
            if area.area >= cfg.stall_area_ratio * region.area:
                ctx.trace.add("candidate_skipped", depth=depth, area=_b(area), reason="stalled: not smaller than its parent")
            else:
                usable.append((area, score))
        kept = nms([a for a, _ in usable], [s for _, s in usable], cfg.nms_iou)
        ctx.trace.add("candidates", depth=depth,
                      all=[{"area": _b(a), "score": round(s, 3)} for a, s in zip(areas, scores)],
                      after_nms=[{"area": _b(a), "score": round(s, 3)} for a, s in kept])

        # Every candidate that survives NMS is tried, best first (Algorithm 1 has no per-level cap).
        # What bounds the search is the per-find call cap and the stall/revisit guards, not a count.
        for area, score in kept:
            if any(iou(area, v) > REVISIT_IOU for v in st.visited):
                ctx.trace.add("candidate_skipped", depth=depth, area=_b(area), reason="already searched")
                continue
            st.visited.append(area)
            ctx.tried.append((area, f"d{depth + 1} s={score:.2f}"))
            ctx.trace.add("descend", depth=depth, area=_b(area), score=round(score, 3))
            found = self._search(ctx, st, area, depth + 1)
            if found is not None:
                return found

        # Nothing left to descend into. A moderately sized region can still be grounded directly
        # (the verifier guards the result), which beats abandoning a branch that holds the target.
        if depth >= 1 and long_side <= cfg.fallback_direct_size:
            ctx.trace.add("fallback_direct", depth=depth, region=_b(region),
                          reason="no candidate left to descend into; region small enough to ground directly")
            return self._direct(ctx, st, region, depth)
        return None

    def _candidate_area(self, box: BBox, region: BBox) -> BBox:
        cfg = self.cfg
        if max(box.width, box.height) >= cfg.dilate_skip_long_side:
            return box  # already a big area (e.g. "the icon column"); growing it adds nothing
        return dilate(box, region, factor=cfg.dilate_factor, min_size=cfg.dilate_min_size,
                      max_ratio=cfg.dilate_max_ratio)

    def _direct(self, ctx: _Ctx, st: _State, region: BBox, depth: int) -> Located | None:
        cfg = ctx.cfg
        crop = crop_region(ctx.img, region, target_long_side=cfg.view_long_side)
        self._charge(ctx, 1, "direct ground")
        res = self.grounder.ground(crop.image, st.description)
        if not (res.found and res.box):
            ctx.trace.add("direct_ground", depth=depth, region=_b(region), found=False, invalid_reply=res.invalid, note=res.note)
            return None
        box = crop.box_to_parent(res.box).clamp(region)
        if not box.is_valid:
            ctx.trace.add("direct_ground", depth=depth, region=_b(region), found=False, note="box outside region")
            return None
        point = box.center
        if st.work is not None and not st.work.contains(point):
            ctx.trace.add("direct_ground", depth=depth, region=_b(region), found=True, box=_b(box),
                          point=[round(point.x), round(point.y)], rejected="point is inside the taskbar band",
                          note=res.note)
            return None
        ctx.trace.add("direct_ground", depth=depth, region=_b(region), found=True, box=_b(box),
                      point=[round(point.x), round(point.y)], note=res.note)

        self._charge(ctx, 1, "verify")
        verdict = self.verifier.verify(ctx.img, point, st.description, box)
        ctx.trace.add("verdict", depth=depth, point=[round(point.x), round(point.y)], box=_b(box),
                      result=verdict.result, new_instruction=verdict.new_instruction, reason=verdict.reason)
        if verdict.is_target:
            return Located(point, box, verdict, depth=depth)
        ctx.rejected.append(point)
        return None

    # -- artifacts --

    def _finish(self, ctx: _Ctx, outcome: str, t0: float, pf_start: int, *, error: str | None = None,
                result: Located | None = None) -> Path:
        failures = list(self.llm.parse_failures[pf_start:]) if self.llm else []
        call_log = list(getattr(self.llm, "call_log", [])[ctx.call_log_start:]) if self.llm else []
        slowest = max(call_log, key=lambda c: c["seconds"] or 0, default=None)
        doc = {
            "description": ctx.trace.description,
            "started": ctx.trace.started,
            "outcome": outcome,
            "error": error,
            "screenshot_size": list(ctx.img.size),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "search_calls": ctx.calls,
            "llm_calls_total": self.llm.calls if self.llm else None,
            "stage_times": stage_times(),
            "llm_call_log": call_log,  # per API call: seconds, retries (SDK backoff), tokens, request id
            "slowest_call": slowest,
            "parse_failures": {"count": len(failures), "details": failures},
            "config": asdict(ctx.cfg),
            "result": None if result is None else {
                "point": [round(result.point.x), round(result.point.y)], "box": _b(result.box), "depth": result.depth},
            "events": ctx.trace.events,
        }
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        base = self.trace_dir / f"trace_{datetime.now():%Y%m%d_%H%M%S}"
        path = base.with_suffix(".json")
        n = 1
        while path.exists():
            n += 1
            path = Path(f"{base}_{n}.json")
        path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        level = logging.WARNING if failures else logging.INFO
        log.log(level, "[search] %s: %d search calls, %d unusable model replies, %.1fs, trace %s",
                outcome, ctx.calls, len(failures), doc["elapsed_s"], path)
        return path

    def _save_failure_image(self, ctx: _Ctx) -> Path | None:
        try:
            img = ctx.img
            for box, label in ctx.tried:
                img = draw_box(img, box, YELLOW, 3, label=label)
            for p in ctx.rejected:
                img = draw_marker(img, p, RED)
            return save_image(img, "find_failed", self.debug_dir)
        except Exception as e:  # never let diagnostics mask the real error
            log.warning("could not save failure image: %s", e)
            return None
