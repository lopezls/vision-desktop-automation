"""Small geometry types shared across modules. All coordinates are pixels unless noted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(self.width, 0) * max(self.height, 0)

    @property
    def center(self) -> Point:
        return Point((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def contains(self, p: Point) -> bool:
        return self.x0 <= p.x <= self.x1 and self.y0 <= p.y <= self.y1

    def intersection(self, other: BBox) -> BBox | None:
        b = BBox(max(self.x0, other.x0), max(self.y0, other.y0),
                 min(self.x1, other.x1), min(self.y1, other.y1))
        return b if b.is_valid else None

    def clamp(self, bounds: BBox) -> BBox:
        """Clip to bounds (may shrink the box)."""
        return BBox(max(self.x0, bounds.x0), max(self.y0, bounds.y0),
                    min(self.x1, bounds.x1), min(self.y1, bounds.y1))

    def offset(self, dx: float, dy: float) -> BBox:
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def as_ints(self) -> tuple[int, int, int, int]:
        return round(self.x0), round(self.y0), round(self.x1), round(self.y1)
