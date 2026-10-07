"""Candidate-area scoring for the ScreenSeekeR-style search (pure geometry, no I/O).

Paper (ScreenSpot-Pro, arXiv 2504.07981, section 4.2 / Algorithm 1):
  - grounded boxes are dilated into larger candidate areas,
  - each candidate's score is the sum, over every grounded box center that falls inside it, of
        s = exp(-((x'-0.5)^2 + (y'-0.5)^2) / (2 sigma^2))        sigma = 0.3
    where (x', y') is the center normalized to the candidate's width/height (0 outside),
  - overlapping candidates are merged with NMS, keeping the higher score.
The dilation numbers and the NMS IoU threshold are not given in the paper; they live in config.
"""

from __future__ import annotations

import math
from typing import Sequence

from .types import BBox, Point


def iou(a: BBox, b: BBox) -> float:
    inter = a.intersection(b)
    if inter is None:
        return 0.0
    union = a.area + b.area - inter.area
    return inter.area / union if union > 0 else 0.0


def _grow_side(side: float, factor: float, min_size: float, max_ratio: float, limit: float) -> float:
    new = max(side * factor, min_size)
    new = min(new, max(side * max_ratio, min_size))  # cap growth, but never below min_size
    return min(new, limit)  # can't be larger than the image itself


def dilate(box: BBox, bounds: BBox, *, factor: float, min_size: float, max_ratio: float) -> BBox:
    """Expand a grounded box into a candidate search area.

    Each side grows by `factor`, at least to `min_size`, at most to `max_ratio` times its
    original length (min_size wins if the two conflict), and never beyond the image. The
    result is centered on the box, then shifted (not shrunk) to stay inside `bounds`.
    """
    if not box.is_valid:
        raise ValueError(f"cannot dilate empty box {box}")
    w = _grow_side(box.width, factor, min_size, max_ratio, bounds.width)
    h = _grow_side(box.height, factor, min_size, max_ratio, bounds.height)
    c = box.center
    x0 = min(max(c.x - w / 2, bounds.x0), bounds.x1 - w)
    y0 = min(max(c.y - h / 2, bounds.y0), bounds.y1 - h)
    return BBox(x0, y0, x0 + w, y0 + h)


def centrality(center: Point, area: BBox, sigma: float = 0.3) -> float:
    """Gaussian score for one grounded-box center against one candidate area."""
    if not area.is_valid or not area.contains(center):
        return 0.0
    xn = (center.x - area.x0) / area.width
    yn = (center.y - area.y0) / area.height
    return math.exp(-((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * sigma**2))


def score_candidates(areas: Sequence[BBox], boxes: Sequence[BBox], sigma: float = 0.3) -> list[float]:
    """Score of each area = sum of centrality over the centers of ALL grounded boxes."""
    centers = [b.center for b in boxes]
    return [sum(centrality(c, a, sigma) for c in centers) for a in areas]


def nms(areas: Sequence[BBox], scores: Sequence[float], iou_thresh: float) -> list[tuple[BBox, float]]:
    """Greedy non-maximum suppression. Returns (area, score) pairs, best first.

    A candidate is dropped when its IoU with an already-kept (higher-scoring) one exceeds
    iou_thresh. Ties keep the earlier (higher-ranked planner hint) candidate.
    """
    if len(areas) != len(scores):
        raise ValueError("areas and scores must have the same length")
    order = sorted(range(len(areas)), key=lambda i: (-scores[i], i))
    kept: list[int] = []
    for i in order:
        if all(iou(areas[i], areas[j]) <= iou_thresh for j in kept):
            kept.append(i)
    return [(areas[i], scores[i]) for i in kept]
