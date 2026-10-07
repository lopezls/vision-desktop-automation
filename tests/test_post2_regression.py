"""Regression for the real post-2 failure (trace_20261006_221236).

The icon was at the center of the desktop. Six candidates survived NMS; the two that contained the
icon (wallpaper area, tooltip area) were ranked 4th and 5th, behind three overlapping left-side
candidates, and a per-level cap of 3 meant they were never searched. Now every candidate that
survives NMS is tried.

Part 1 replays the real grounded boxes through the real scoring/NMS. Part 2 replays the whole search
on a synthetic screen laid out like that desktop, with the target reachable ONLY through the
candidate that is ranked 4th, so it cannot be found unless candidates beyond the 3rd are searched.
"""

from dataclasses import replace

import pytest
from PIL import Image, ImageChops, ImageDraw

from vision_automation.config import CONFIG
from vision_automation.grounder import GroundResult
from vision_automation.planner import Hint, PopupAssessment, PositionInference
from vision_automation.scoring import dilate, nms, score_candidates
from vision_automation.search import Searcher
from vision_automation.types import BBox
from vision_automation.verifier import Verdict

SCREEN = BBox(0, 0, 1920, 1080)
ICON = (949, 436)  # where the Notepad icon really was (center of its box)

# The six grounded boxes from the trace's level 0, in the planner's hint order.
REAL_BOXES = [
    ("center of the desktop wallpaper, around (949, 440)", BBox(277, 102, 1652, 1032)),
    ('tooltip text "Location: notepad (C:\\Windows\\System32)', BBox(950, 465, 1190, 486)),
    ("Notion desktop icon", BBox(81, 407, 147, 475)),
    ("Dev Projects folder icon", BBox(78, 311, 149, 375)),
    ("left-side desktop icon columns", BBox(0, 10, 154, 976)),
    ("Mp3tag", BBox(81, 509, 147, 576)),
]
REAL_SCORES = [1.917, 1.397, 2.404, 1.391, 3.076, 2.025]  # as logged in the failed run


def candidate_areas(cfg=CONFIG):
    boxes = [b for _, b in REAL_BOXES]
    areas = [b if max(b.width, b.height) >= cfg.dilate_skip_long_side
             else dilate(b, SCREEN, factor=cfg.dilate_factor, min_size=cfg.dilate_min_size,
                         max_ratio=cfg.dilate_max_ratio) for b in boxes]
    return boxes, areas


# ---- part 1: the real candidate list -----------------------------------------------------

def test_scoring_reproduces_the_logged_scores():
    boxes, areas = candidate_areas()
    scores = score_candidates(areas, boxes, CONFIG.sigma)
    assert scores == pytest.approx(REAL_SCORES, abs=0.01)
    assert areas[1].as_ints() == (710, 356, 1430, 596)  # the tooltip-derived candidate from the trace


def test_old_settings_ranked_the_icons_candidates_4th_and_5th_of_6():
    boxes, areas = candidate_areas()
    scores = score_candidates(areas, boxes, CONFIG.sigma)
    ranked = nms(areas, scores, 0.5)  # the old nms_iou
    order = [a for a, _ in ranked]
    assert len(order) == 6
    containing = [i for i, a in enumerate(order) if a.contains(_pt(ICON))]
    assert containing == [3, 4]  # 4th and 5th: behind the cut-off of 3 that used to apply


def test_new_nms_merges_the_overlapping_left_side_candidates():
    boxes, areas = candidate_areas()
    scores = score_candidates(areas, boxes, CONFIG.sigma)
    kept = nms(areas, scores, CONFIG.nms_iou)
    assert CONFIG.nms_iou == pytest.approx(0.3)
    assert [s for _, s in kept] == pytest.approx([3.076, 2.404, 1.917, 1.397], abs=0.01)
    # both candidates that contain the icon survive, and the redundant Mp3tag / Dev Projects ones are gone
    assert sum(a.contains(_pt(ICON)) for a, _ in kept) == 2


def test_there_is_no_per_level_candidate_cap_any_more():
    assert not hasattr(CONFIG, "max_candidates_per_level")


def _pt(xy):
    from vision_automation.types import Point
    return Point(*xy)


# ---- part 2: the whole search on a synthetic copy of that desktop -----------------------

BG = (128, 128, 128)
COLOR = {  # query keyword -> exact color
    "left-side": (200, 200, 60), "Notion": (60, 200, 200), "Dev Projects": (200, 60, 200),
    "Mp3tag": (60, 60, 200), "wallpaper": (30, 120, 220), "tooltip": (250, 250, 250),
    "selection box": (255, 160, 0), "checkbox": (10, 40, 120), "arrow": (0, 255, 255),
    "text label": (255, 0, 255), "Notepad desktop icon": (230, 0, 0),
}
LEFT_COL = BBox(0, 10, 154, 976)
TARGET = BBox(925, 405, 975, 470)  # the icon itself (red)
# Pieces of the icon, as the real planner described them inside the tooltip-derived region
SELECTION = BBox(912, 405, 988, 475)
CHECKBOX = BBox(926, 406, 940, 420)
ARROW = BBox(925, 435, 946, 456)
LABEL = BBox(927, 460, 975, 473)
TOOLTIP = BBox(950, 465, 1190, 486)
WALLPAPER = BBox(277, 102, 1652, 1032)


def world():
    img = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(img)
    for box, key in [(LEFT_COL, "left-side"), (REAL_BOXES[2][1], "Notion"), (REAL_BOXES[3][1], "Dev Projects"),
                     (REAL_BOXES[5][1], "Mp3tag"), (WALLPAPER, "wallpaper"), (SELECTION, "selection box"),
                     (TARGET, "Notepad desktop icon"), (CHECKBOX, "checkbox"), (ARROW, "arrow"),
                     (TOOLTIP, "tooltip"), (LABEL, "text label")]:
        d.rectangle(box.as_ints(), fill=COLOR[key])
    return img


class Grounder:
    def __init__(self):
        self.queries = []

    def ground(self, image, query):
        self.queries.append(query)
        for key, color in COLOR.items():
            if key in query:
                mask = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, color))
                bbox = mask.convert("L").point(lambda v: 255 if v < 6 else 0).getbbox()
                if bbox:
                    return GroundResult(True, BBox(*bbox), 0.9, "fake")
        return GroundResult(False, None, None, "not visible")


class Planner:
    """Level 0: the real run's six hints. Crops: behaves like the real planner did on those crops."""

    def __init__(self):
        self.crop_sizes = []

    def check_popup(self, screenshot, description):
        return PopupAssessment(False)

    def infer_positions(self, image, description, *, is_crop=False):
        if not is_crop:
            return PositionInference([Hint("area" if i in (0, 4) else "neighbor", t, i)
                                      for i, (t, _) in enumerate(REAL_BOXES)])
        self.crop_sizes.append(image.size)
        if image.width == 800:  # the tooltip candidate's crop (720x240 scaled to 800 wide)
            # The real planner's hints for this crop (trace_20261006_222328), incl. the WIDE tooltip box
            # whose dilated candidate equals the whole region and used to win NMS, then get skipped.
            return PositionInference([
                Hint("area", "highlighted selection box around the notepad shortcut on the desktop", 0),
                Hint("neighbor", "blue selection checkbox in the icon's top-left corner", 1),
                Hint("neighbor", "white shortcut arrow overlay at the icon's bottom-left", 2),
                Hint("neighbor", 'notepad" text label beneath the icon', 3),
                Hint("area", "tooltip region just below the icon", 4),
                Hint("neighbor", "notepad text label", 5)])
        # left column and wallpaper crops: the real planner listed only <element> items there
        return PositionInference([Hint("element", "some desktop icon", 0)])


class Verifier:
    def __init__(self):
        self.points = []

    def verify(self, screenshot, point, description, box=None):
        self.points.append(point)
        return Verdict("is_target" if TARGET.contains(point) else "target_not_found")


def run_search(tmp_path, cfg=CONFIG):
    planner, grounder, verifier = Planner(), Grounder(), Verifier()
    s = Searcher(planner, grounder, verifier, None, cfg, tmp_path / "t", tmp_path / "d",
                 work_area_fn=lambda size: BBox(0, 0, 1920, 1032))
    return s, planner, grounder, verifier


def test_the_correct_candidate_ranked_4th_is_now_reached_and_the_icon_is_found(tmp_path):
    s, planner, grounder, verifier = run_search(tmp_path)
    loc = s.find(world(), "the Notepad desktop icon (not the taskbar)")

    assert TARGET.contains(loc.point) and abs(loc.point.x - 950) < 8 and abs(loc.point.y - 437) < 8
    import json
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    cand = next(e for e in doc["events"] if e["kind"] == "candidates" and e["depth"] == 0)
    assert [c["score"] for c in cand["after_nms"]] == pytest.approx([3.076, 2.404, 1.917, 1.397], abs=0.02)
    descents = [e["area"] for e in doc["events"] if e["kind"] == "descend" and e["depth"] == 0]
    # tried in rank order, ending with the 4th: the tooltip-derived candidate that contains the icon
    assert len(descents) == 4
    assert descents[0][:2] == [0, 10] and descents[3][:2] in ([709, 356], [710, 356])  # left column first, tooltip area 4th
    assert doc["search_calls"] <= CONFIG.max_llm_calls_per_find  # still bounded by the global cap


def test_it_would_have_missed_the_icon_with_the_old_cap_of_three(tmp_path):
    # Same search, but stop after 3 descents at level 0 (what max_candidates_per_level=3 did).
    s, *_ = run_search(tmp_path)
    original = s._search
    state = {"level0": 0}

    def capped(ctx, st, region, depth):
        if depth == 1:  # every call at depth 1 is a descent from level 0
            state["level0"] += 1
            if state["level0"] > 3:
                return None
        return original(ctx, st, region, depth)

    s._search = capped
    from vision_automation.search import GroundingError
    with pytest.raises(GroundingError):
        s.find(world(), "the Notepad desktop icon (not the taskbar)")


def test_a_search_that_cannot_find_the_icon_is_still_bounded_by_the_call_cap(tmp_path):
    s, _, _, verifier = run_search(tmp_path, cfg=replace(CONFIG, max_llm_calls_per_find=8))
    from vision_automation.search import CallBudgetExceeded
    verifier.verify = lambda *a, **k: Verdict("target_not_found")
    with pytest.raises(CallBudgetExceeded) as ei:
        s.find(world(), "the Notepad desktop icon (not the taskbar)")
    assert ei.value.calls <= 8


def test_a_region_sized_candidate_must_not_suppress_the_tight_candidates_that_can_be_searched(tmp_path):
    """The second failure the real post-2 screenshot exposed.

    In the tooltip-derived region the planner described the icon's selection box, checkbox, arrow and
    label plus the WIDE tooltip. The tooltip's dilated candidate equals the whole region, scores highest
    (every hint's center is inside it), and used to win NMS over the tight icon-centered candidates
    (IoU ~0.33 > 0.3); the stall guard then skipped it and nothing was left. The stall guard now runs
    before NMS, so the tight candidates survive.
    """
    import json
    s, *_ = run_search(tmp_path)
    loc = s.find(world(), "the Notepad desktop icon (not the taskbar)")
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    region = [709, 356, 1432, 596]
    level1 = [e for e in doc["events"] if e["kind"] == "candidates" and e["depth"] == 1]
    assert len(level1) == 1
    c = level1[0]
    # the region-sized candidate exists, has the top score, and is reported as stalled ...
    top = max(c["all"], key=lambda x: x["score"])
    assert top["area"][0] <= 710 and top["area"][2] >= 1430
    assert any(e["kind"] == "candidate_skipped" and e["depth"] == 1 and "stalled" in e["reason"]
               for e in doc["events"])
    # ... but a tight candidate around the icon survived NMS and led to the verified find
    assert c["after_nms"] and all(a["area"] != top["area"] for a in c["after_nms"])
    assert any(a["area"][0] <= 949 <= a["area"][2] and a["area"][1] <= 436 <= a["area"][3] for a in c["after_nms"])
    assert TARGET.contains(loc.point)
