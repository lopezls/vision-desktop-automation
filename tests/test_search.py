"""Search tests: the real Searcher against a synthetic screen with fake planner/grounder/verifier."""

import json
from dataclasses import replace

import pytest
from PIL import Image, ImageChops, ImageDraw

from vision_automation.config import CONFIG
from vision_automation.grounder import GroundResult
from vision_automation.planner import Hint, PopupAssessment, PositionInference
from vision_automation.search import (
    CallBudgetExceeded,
    GroundingError,
    PopupBlocked,
    Searcher,
    clip_to_work_area,
    excludes_taskbar,
)
from vision_automation.types import BBox
from vision_automation.screen import work_area
from vision_automation.verifier import Verdict

GRAY, GREEN, BLUE, RED = (128, 128, 128), (0, 200, 0), (0, 0, 220), (230, 0, 0)
# 'ORANGE' is really cyan: anti-aliased red/yellow edges blend into orange-ish pixels, which
# confused the colour-matching fake grounder.
YELLOW, ORANGE, PURPLE = (230, 230, 0), (0, 220, 220), (160, 0, 200)

GREEN_PANEL = BBox(1300, 100, 1700, 700)
BLUE_BOX = BBox(1420, 300, 1480, 360)
TARGET = BBox(1500, 300, 1560, 360)  # the red square we are looking for
YELLOW_PANEL = BBox(100, 700, 500, 1000)
ORANGE_BOX = BBox(130, 740, 180, 790)
PURPLE_BOX = BBox(200, 740, 250, 790)
DECOY = BBox(300, 800, 360, 860)  # also red, but not the target
TRAY = BBox(1500, 1040, 1700, 1070)  # "system tray": its center is inside the taskbar band
WORK = BBox(0, 0, 1920, 1032)  # screen minus a 48px taskbar, like the real 1080p desktop
TRAY_COLOR = (10, 10, 10)

COLORS = {"green panel": GREEN, "blue square": BLUE, "yellow panel": YELLOW, "orange square": ORANGE,
          "purple square": PURPLE, "red square": RED, "tray icons": TRAY_COLOR}
DESC = "the red square (not the taskbar)"


def world() -> Image.Image:
    img = Image.new("RGB", (1920, 1080), GRAY)
    d = ImageDraw.Draw(img)
    for box, color in [(GREEN_PANEL, GREEN), (BLUE_BOX, BLUE), (TARGET, RED), (YELLOW_PANEL, YELLOW),
                       (ORANGE_BOX, ORANGE), (PURPLE_BOX, PURPLE), (DECOY, RED), (TRAY, TRAY_COLOR)]:
        d.rectangle(box.as_ints(), fill=color)
    return img


class FakeGrounder:
    """Finds the bounding box of the colour named in the query, in the pixels of the image it is given."""

    def __init__(self, llm=None):
        self.queries, self.llm = [], llm

    def ground(self, image, query):
        self.queries.append((query, image.size))
        for key, color in COLORS.items():
            if key in query:
                mask = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, color))
                bbox = mask.convert("L").point(lambda v: 255 if v < 6 else 0).getbbox()
                if bbox:
                    return GroundResult(True, BBox(*bbox), 0.9, "fake")
        return GroundResult(False, None, None, "fake: not visible")


class FakePlanner:
    def __init__(self, hints, popup=None, empty_on_crop=False):
        self.hints, self.popup, self.empty_on_crop = hints, popup, empty_on_crop
        self.infer_calls = []

    def check_popup(self, screenshot, description):
        return self.popup or PopupAssessment(False)

    def infer_positions(self, image, description, *, is_crop=False):
        self.infer_calls.append((image.size, is_crop))
        if is_crop and self.empty_on_crop:
            return PositionInference(no_target=True)
        return PositionInference([Hint(k, t, i) for i, (k, t) in enumerate(self.hints)])


class FakeVerifier:
    def __init__(self, accept=(TARGET,)):
        self.accept, self.points = accept, []

    def verify(self, screenshot, point, description, box=None):
        self.points.append(point)
        ok = any(b.contains(point) for b in self.accept)
        return Verdict("is_target" if ok else "target_not_found")


DEFAULT_HINTS = [("area", "green panel"), ("neighbor", "blue square")]


def make(tmp_path, hints=DEFAULT_HINTS, cfg=CONFIG, planner=None, verifier=None, llm=None, work=WORK, **kw):
    planner = planner or FakePlanner(hints, **kw)
    grounder, verifier = FakeGrounder(llm), verifier or FakeVerifier()
    s = Searcher(planner, grounder, verifier, llm, cfg, trace_dir=tmp_path / "t", debug_dir=tmp_path / "d",
                 work_area_fn=lambda size: work)  # deterministic: never the real machine's taskbar
    return s, planner, grounder, verifier


# ---- happy path -------------------------------------------------------------------------

def test_finds_target_zooming_in_and_maps_back_to_screen_pixels(tmp_path):
    s, planner, grounder, verifier = make(tmp_path)
    loc = s.find(world(), DESC)
    assert TARGET.contains(loc.point)
    assert abs(loc.point.x - 1530) < 6 and abs(loc.point.y - 330) < 6
    assert loc.verdict.is_target and loc.depth >= 1
    # it recursed: the planner saw the full screen first, then a crop, never the full screen again
    assert planner.infer_calls[0] == ((1920, 1080), False) and planner.infer_calls[1][1] is True
    # and it grounded the final target directly in a small crop (<= direct size, upscaled to view size)
    direct = [q for q in grounder.queries if q[0] == DESC]
    assert len(direct) == 1 and max(direct[0][1]) == CONFIG.view_long_side


def test_trace_is_saved_with_hints_scores_candidates_verdicts_and_timings(tmp_path):
    s, *_ = make(tmp_path)
    loc = s.find(world(), DESC)
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in doc["events"]]
    for k in ("popup", "hints", "grounded", "candidates", "descend", "direct_ground", "verdict"):
        assert k in kinds, k
    cand = next(e for e in doc["events"] if e["kind"] == "candidates")
    assert all("score" in c and "area" in c for c in cand["all"]) and cand["after_nms"]
    assert doc["outcome"] == "found" and doc["result"]["point"] and doc["parse_failures"]["count"] == 0
    assert doc["search_calls"] == loc.calls and "config" in doc and "stage_times" in doc


def test_failed_search_still_saves_trace_and_failure_image(tmp_path):
    s, *_ = make(tmp_path, verifier=FakeVerifier(accept=()))
    with pytest.raises(GroundingError) as ei:
        s.find(world(), DESC)
    e = ei.value
    assert e.trace_path.exists() and e.image_path.exists() and e.calls > 0
    assert json.loads(e.trace_path.read_text(encoding="utf-8"))["outcome"] == "failed"


# ---- backtracking, guards ---------------------------------------------------------------

def test_backtracks_when_the_verifier_rejects_the_best_candidate(tmp_path):
    hints = [("area", "yellow panel"), ("neighbor", "orange square"), ("neighbor", "purple square"),
             ("area", "green panel")]
    cfg = replace(CONFIG, max_llm_calls_per_find=80, max_depth=2)
    s, _, _, verifier = make(tmp_path, hints, cfg)
    loc = s.find(world(), DESC)
    assert TARGET.contains(loc.point)
    assert any(DECOY.contains(p) for p in verifier.points)  # it really did go down the decoy branch first
    assert len(verifier.points) >= 2


def test_falls_back_to_direct_grounding_when_a_small_region_has_nothing_to_descend_into(tmp_path):
    # In the green panel crop the only hint that grounds is the panel itself (the whole crop), which
    # the stall guard skips; the 600px region is then grounded directly instead of dead-ending.
    s, *_ = make(tmp_path, hints=[("area", "green panel")])
    loc = s.find(world(), DESC)
    assert TARGET.contains(loc.point)
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in doc["events"]]
    assert "candidate_skipped" in kinds and "fallback_direct" in kinds


def test_fallback_can_be_disabled(tmp_path):
    s, *_ = make(tmp_path, hints=[("area", "green panel")], cfg=replace(CONFIG, fallback_direct_size=0))
    with pytest.raises(GroundingError):
        s.find(world(), DESC)


def test_call_cap_stops_the_search(tmp_path):
    cfg = replace(CONFIG, max_llm_calls_per_find=4)
    s, *_ = make(tmp_path, cfg=cfg)
    with pytest.raises(CallBudgetExceeded) as ei:
        s.find(world(), DESC)
    assert ei.value.calls <= 4
    doc = json.loads(ei.value.trace_path.read_text(encoding="utf-8"))
    assert doc["search_calls"] <= 4 and any(e["kind"] == "call_cap" for e in doc["events"])


def test_calls_never_exceed_the_cap_even_when_searching_fails(tmp_path):
    for cap in (3, 6, 10, 25):
        s, *_ = make(tmp_path, cfg=replace(CONFIG, max_llm_calls_per_find=cap), verifier=FakeVerifier(accept=()))
        with pytest.raises(GroundingError) as ei:
            s.find(world(), DESC)
        assert ei.value.calls <= cap


class _FullBoxGrounder(FakeGrounder):
    """Always claims the whole image for hint queries: a candidate that is never 'smaller'."""

    def ground(self, image, query):
        if query == DESC:
            return super().ground(image, query)
        self.queries.append((query, image.size))
        return GroundResult(True, BBox(0, 0, image.width, image.height), 0.5, "whole image")


def test_stall_guard_skips_candidates_that_do_not_shrink(tmp_path):
    planner, verifier = FakePlanner([("area", "green panel")]), FakeVerifier()
    grounder = _FullBoxGrounder()
    s = Searcher(planner, grounder, verifier, None, CONFIG, tmp_path / "t", tmp_path / "d")
    with pytest.raises(GroundingError) as ei:
        s.find(world(), DESC)
    assert len(planner.infer_calls) == 1  # never recursed into the whole-screen "candidate"
    doc = json.loads(ei.value.trace_path.read_text(encoding="utf-8"))
    assert any(e["kind"] == "candidate_skipped" and "stalled" in e["reason"] for e in doc["events"])


def test_no_target_on_a_crop_fails_that_branch_without_crashing(tmp_path):
    # small direct size so every candidate needs the planner (tiny candidates would skip it)
    s, planner, *_ = make(tmp_path, cfg=replace(CONFIG, direct_ground_size=100), empty_on_crop=True)
    with pytest.raises(GroundingError) as ei:
        s.find(world(), DESC)
    doc = json.loads(ei.value.trace_path.read_text(encoding="utf-8"))
    assert any(e["kind"] == "level_failed" and "No target" in e["reason"] for e in doc["events"])


def test_direct_ground_size_is_a_config_knob(tmp_path):
    cfg = replace(CONFIG, direct_ground_size=2000)  # whole screen counts as "small enough"
    s, planner, grounder, verifier = make(tmp_path, cfg=cfg)
    # grounding the whole 1920x1080 screen for "red square" spans target AND decoy -> verifier rejects
    with pytest.raises(GroundingError):
        s.find(world(), DESC)
    assert planner.infer_calls == [] and [q[0] for q in grounder.queries] == [DESC]


def test_max_depth_zero_grounds_directly(tmp_path):
    s, planner, grounder, _ = make(tmp_path, cfg=replace(CONFIG, max_depth=0))
    with pytest.raises(GroundingError):
        s.find(world(), DESC)
    assert planner.infer_calls == []


# ---- taskbar handling -------------------------------------------------------------------

@pytest.mark.parametrize("desc,expected", [
    ("the Notepad desktop icon (not the taskbar)", True),
    ("the Notepad icon, not on the taskbar", True),
    ("the Notepad icon NOT in the Task Bar", True),
    ("the Notepad icon on the taskbar", False),
    ("the Notepad desktop icon", False),
    ("the taskbar clock", False),
])
def test_excludes_taskbar(desc, expected):
    assert excludes_taskbar(desc) is expected


def test_hint_mentioning_the_taskbar_is_still_grounded_and_used(tmp_path):
    # The planner's real hint from a failed bottom-right run. The old word filter dropped it; geometry keeps it.
    hint = "green panel at the right edge of the desktop, bottom-right, above the taskbar"
    s, _, grounder, _ = make(tmp_path, hints=[("area", hint)])
    loc = s.find(world(), DESC)
    assert TARGET.contains(loc.point)
    assert any(q[0] == hint for q in grounder.queries)
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    grounded = next(e for e in doc["events"] if e["kind"] == "grounded")
    assert grounded["found"] and grounded["rejected"] is None
    assert not any(e["kind"] == "hints_skipped" for e in doc["events"])
    assert next(e for e in doc["events"] if e["kind"] == "taskbar_exclusion")["active"] is True


def test_hint_whose_box_is_centered_in_the_taskbar_band_is_rejected(tmp_path):
    hints = [("neighbor", "tray icons"), ("area", "green panel"), ("neighbor", "blue square")]
    s, _, grounder, _ = make(tmp_path, hints)
    loc = s.find(world(), DESC)
    assert TARGET.contains(loc.point)
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    tray = next(e for e in doc["events"] if e["kind"] == "grounded" and "tray" in e["hint"])
    assert tray["found"] is False and "taskbar band" in tray["rejected"]
    cands = next(e for e in doc["events"] if e["kind"] == "candidates")
    assert len(cands["all"]) == 2  # only the two non-taskbar boxes became candidates


def test_taskbar_band_is_only_excluded_when_the_description_says_so(tmp_path):
    hints = [("neighbor", "tray icons"), ("area", "green panel"), ("neighbor", "blue square")]
    s, *_ = make(tmp_path, hints)
    loc = s.find(world(), "the red square")  # no "not the taskbar"
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    assert not any(e["kind"] == "taskbar_exclusion" for e in doc["events"])
    tray = next(e for e in doc["events"] if e["kind"] == "grounded" and "tray" in e["hint"])
    assert tray["found"] is True and tray["rejected"] is None


def test_final_point_inside_the_band_is_refused(tmp_path):
    cfg = replace(CONFIG, direct_ground_size=2000)  # ground the target directly on the whole screen
    s, *_ = make(tmp_path, cfg=cfg, hints=[])
    with pytest.raises(GroundingError) as ei:
        s.find(world(), "the tray icons (not the taskbar)")
    doc = json.loads(ei.value.trace_path.read_text(encoding="utf-8"))
    ev = next(e for e in doc["events"] if e["kind"] == "direct_ground")
    assert "taskbar band" in ev["rejected"]


def test_no_exclusion_when_the_work_area_is_unavailable_or_the_whole_screen(tmp_path):
    for work in (None, BBox(0, 0, 1920, 1080)):
        s, *_ = make(tmp_path, hints=[("neighbor", "tray icons"), ("area", "green panel"),
                                      ("neighbor", "blue square")], work=work)
        loc = s.find(world(), DESC)
        doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
        ev = next(e for e in doc["events"] if e["kind"] == "taskbar_exclusion")
        assert ev["active"] is False
        tray = next(e for e in doc["events"] if e["kind"] == "grounded" and "tray" in e["hint"])
        assert tray["found"] is True  # nothing rejected without a known band


def test_clip_to_work_area():
    work = BBox(0, 0, 1920, 1032)
    assert clip_to_work_area(BBox(100, 100, 300, 300), work) == BBox(100, 100, 300, 300)
    assert clip_to_work_area(BBox(100, 900, 300, 1100), work) == BBox(100, 900, 300, 1032)  # center above: clipped
    assert clip_to_work_area(BBox(100, 1000, 300, 1200), work) is None  # center (200, 1100) is in the band
    assert clip_to_work_area(BBox(100, 1020, 300, 1034), work) == BBox(100, 1020, 300, 1032)  # center y=1027: kept
    assert clip_to_work_area(BBox(100, 1026, 300, 1038), work) is None  # center y=1032 is on the edge, but only 6px survive
    assert clip_to_work_area(BBox(100, 5, 300, 13), work) == BBox(100, 5, 300, 13)
    assert clip_to_work_area(BBox(0, 0, 50, 50), None) == BBox(0, 0, 50, 50)


def test_work_area_matches_the_live_screen_and_ignores_other_image_sizes():
    from vision_automation.screen import screen_size
    w = work_area()
    sw, sh = screen_size()
    assert w is not None and 0 < w.width <= sw and 0 < w.height <= sh
    assert work_area((sw + 1, sh)) is None


# ---- pop-ups (the searcher never clicks) -----------------------------------------------

def _popup(action="close", label="X", desc="orange square"):
    return PopupAssessment(True, action, "update banner", label, desc, "blocks the screen")


def test_popup_with_safe_control_raises_popupblocked_with_a_verified_point(tmp_path):
    cfg = replace(CONFIG, direct_ground_size=2000)
    s, *_ = make(tmp_path, cfg=cfg, popup=_popup(), verifier=FakeVerifier(accept=(ORANGE_BOX,)))
    with pytest.raises(PopupBlocked) as ei:
        s.find(world(), DESC)
    assert ORANGE_BOX.contains(ei.value.control_point)
    assert ei.value.assessment.control_label == "X"
    assert json.loads(ei.value.trace_path.read_text(encoding="utf-8"))["outcome"] == "popup_blocked"


def test_popup_without_safe_control_is_an_error_not_a_click(tmp_path):
    s, *_ = make(tmp_path, popup=_popup(action="abort"))
    with pytest.raises(GroundingError, match="no safe dismiss control"):
        s.find(world(), DESC)


def test_popup_control_that_cannot_be_verified_is_an_error(tmp_path):
    cfg = replace(CONFIG, direct_ground_size=2000)
    s, *_ = make(tmp_path, cfg=cfg, popup=_popup(), verifier=FakeVerifier(accept=()))
    with pytest.raises(GroundingError, match="could not locate and verify"):
        s.find(world(), DESC)


def test_popup_check_can_be_skipped(tmp_path):
    s, *_ = make(tmp_path, popup=_popup(action="abort"))
    loc = s.find(world(), DESC, check_popup=False)
    assert TARGET.contains(loc.point)


# ---- parse-failure accounting -----------------------------------------------------------

class _StubLLM:
    def __init__(self):
        self.calls, self.parse_failures = 0, []

    def note_parse_failure(self, stage, detail):
        self.parse_failures.append({"stage": stage, "detail": detail})


class _NoisyGrounder(FakeGrounder):
    def ground(self, image, query):
        self.llm.note_parse_failure("ground", "unparsable JSON: test")
        return super().ground(image, query)


def test_parse_failures_during_a_find_are_counted_in_the_trace(tmp_path):
    llm = _StubLLM()
    llm.note_parse_failure("plan", "from an earlier find")  # must not be attributed to this find
    planner = FakePlanner(DEFAULT_HINTS)
    s = Searcher(planner, _NoisyGrounder(llm), FakeVerifier(), llm, CONFIG, tmp_path / "t", tmp_path / "d")
    loc = s.find(world(), DESC)
    doc = json.loads(loc.trace_path.read_text(encoding="utf-8"))
    assert doc["parse_failures"]["count"] == len(llm.parse_failures) - 1 > 0
    assert all(d["stage"] == "ground" for d in doc["parse_failures"]["details"])
