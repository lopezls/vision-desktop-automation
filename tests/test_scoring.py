import math

import pytest

from vision_automation.scoring import centrality, dilate, iou, nms, score_candidates
from vision_automation.types import BBox, Point

SCREEN = BBox(0, 0, 1920, 1080)
SIGMA = 0.3


# ---- centrality formula ----------------------------------------------------------------

def test_center_of_area_scores_one():
    assert centrality(Point(150, 150), BBox(100, 100, 200, 200), SIGMA) == pytest.approx(1.0)


def test_corner_of_area_matches_formula():
    # x'=y'=0 -> exp(-(0.25+0.25) / (2*0.09))
    expected = math.exp(-0.5 / (2 * 0.3**2))
    assert centrality(Point(100, 100), BBox(100, 100, 200, 200), SIGMA) == pytest.approx(expected)
    assert expected == pytest.approx(0.0622, abs=1e-4)


def test_off_center_hand_computed():
    # area 200x100 at (0,0); point (150, 25) -> x'=0.75, y'=0.25 -> exp(-(0.0625+0.0625)/0.18)
    expected = math.exp(-0.125 / 0.18)
    assert centrality(Point(150, 25), BBox(0, 0, 200, 100), SIGMA) == pytest.approx(expected)


def test_normalization_uses_width_and_height_separately():
    # same relative position in two differently shaped areas scores the same
    a = centrality(Point(25, 25), BBox(0, 0, 100, 100), SIGMA)
    b = centrality(Point(100, 25), BBox(0, 0, 400, 100), SIGMA)
    assert a == pytest.approx(b)


def test_outside_scores_zero():
    assert centrality(Point(99, 150), BBox(100, 100, 200, 200), SIGMA) == 0.0
    assert centrality(Point(150, 201), BBox(100, 100, 200, 200), SIGMA) == 0.0


def test_closer_to_center_scores_higher():
    area = BBox(0, 0, 100, 100)
    assert centrality(Point(50, 50), area) > centrality(Point(70, 50), area) > centrality(Point(95, 95), area)


def test_degenerate_area_scores_zero():
    assert centrality(Point(5, 5), BBox(5, 5, 5, 20), SIGMA) == 0.0


# ---- score_candidates ------------------------------------------------------------------

def test_score_is_sum_over_all_box_centers():
    area = BBox(0, 0, 100, 100)
    boxes = [BBox(40, 40, 60, 60),      # center (50,50) -> 1.0
             BBox(0, 0, 20, 20),        # center (10,10)
             BBox(500, 500, 520, 520)]  # outside -> 0
    expected = 1.0 + centrality(Point(10, 10), area, SIGMA)
    assert score_candidates([area], boxes, SIGMA) == [pytest.approx(expected)]


def test_cluster_of_boxes_beats_lone_box():
    # two areas; three grounded boxes cluster near area A, one near B
    a = BBox(0, 0, 200, 200)
    b = BBox(1000, 600, 1200, 800)
    boxes = [BBox(90, 90, 110, 110), BBox(80, 100, 100, 120), BBox(110, 80, 130, 100), BBox(1090, 690, 1110, 710)]
    sa, sb = score_candidates([a, b], boxes, SIGMA)
    assert sa > sb > 0


def test_no_boxes_scores_zero():
    assert score_candidates([BBox(0, 0, 10, 10)], [], SIGMA) == [0.0]


# ---- dilation --------------------------------------------------------------------------

def test_dilate_grows_by_factor_and_stays_centered():
    out = dilate(BBox(900, 500, 1000, 600), SCREEN, factor=3, min_size=50, max_ratio=10)
    assert (out.width, out.height) == (300, 300)
    assert out.center == Point(950, 550)


def test_dilate_enforces_min_size():
    out = dilate(BBox(100, 100, 110, 110), SCREEN, factor=2, min_size=240, max_ratio=100)
    assert (out.width, out.height) == (240, 240)


def test_dilate_enforces_max_ratio():
    out = dilate(BBox(900, 500, 1000, 600), SCREEN, factor=20, min_size=50, max_ratio=4)
    assert (out.width, out.height) == (400, 400)


def test_dilate_min_size_beats_max_ratio():
    out = dilate(BBox(100, 100, 120, 120), SCREEN, factor=2, min_size=240, max_ratio=3)
    assert (out.width, out.height) == (240, 240)


def test_dilate_shifts_not_shrinks_at_corner():
    out = dilate(BBox(0, 0, 40, 40), SCREEN, factor=1, min_size=240, max_ratio=100)
    assert out == BBox(0, 0, 240, 240)
    out = dilate(BBox(1880, 1040, 1920, 1080), SCREEN, factor=1, min_size=240, max_ratio=100)
    assert out == BBox(1680, 840, 1920, 1080)


def test_dilate_never_exceeds_image():
    out = dilate(BBox(900, 400, 1000, 500), SCREEN, factor=100, min_size=50, max_ratio=1000)
    assert out == SCREEN


def test_dilate_wide_box_keeps_aspect_via_per_side_growth():
    out = dilate(BBox(800, 500, 1000, 540), SCREEN, factor=2, min_size=10, max_ratio=10)
    assert (out.width, out.height) == (400, 80)


def test_dilate_empty_box_raises():
    with pytest.raises(ValueError):
        dilate(BBox(10, 10, 10, 50), SCREEN, factor=2, min_size=10, max_ratio=5)


# ---- IoU / NMS -------------------------------------------------------------------------

def test_iou_values():
    assert iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)) == 1.0
    assert iou(BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)) == 0.0
    # 10x10 and 10x10 overlapping 5x10: inter 50, union 150
    assert iou(BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou(BBox(0, 0, 10, 10), BBox(10, 0, 20, 10)) == 0.0  # touching edges only


def test_nms_keeps_higher_score_of_overlapping_pair():
    a = BBox(0, 0, 100, 100)
    b = BBox(10, 10, 110, 110)  # IoU ~0.68 with a
    out = nms([a, b], [1.0, 2.5], iou_thresh=0.5)
    assert out == [(b, 2.5)]


def test_nms_keeps_disjoint_and_sorts_descending():
    a, b, c = BBox(0, 0, 50, 50), BBox(200, 0, 250, 50), BBox(400, 0, 450, 50)
    out = nms([a, b, c], [0.5, 3.0, 1.5], iou_thresh=0.5)
    assert out == [(b, 3.0), (c, 1.5), (a, 0.5)]


def test_nms_overlap_below_threshold_survives():
    a = BBox(0, 0, 100, 100)
    b = BBox(80, 0, 180, 100)  # IoU = 20*100/(20000-2000) = 0.111
    assert len(nms([a, b], [1.0, 0.9], iou_thresh=0.5)) == 2


def test_nms_chain_suppression_is_against_kept_only():
    # b overlaps a (suppressed by a); c overlaps only b -> c must survive since b was dropped
    a = BBox(0, 0, 100, 100)
    b = BBox(40, 0, 140, 100)   # IoU(a,b) = 60*100/14000 ~ 0.43 -> use thresh 0.4
    c = BBox(130, 0, 230, 100)  # IoU(b,c) small, IoU(a,c) = 0
    out = nms([a, b, c], [3.0, 2.0, 1.0], iou_thresh=0.4)
    assert [x for x, _ in out] == [a, c]


def test_nms_tie_keeps_earlier_candidate():
    a = BBox(0, 0, 100, 100)
    b = BBox(0, 0, 100, 100)
    assert nms([a, b], [1.0, 1.0], iou_thresh=0.5) == [(a, 1.0)]


def test_nms_length_mismatch_raises():
    with pytest.raises(ValueError):
        nms([BBox(0, 0, 1, 1)], [], 0.5)


def test_nms_empty():
    assert nms([], [], 0.5) == []
