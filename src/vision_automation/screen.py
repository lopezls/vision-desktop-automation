"""Screenshot capture, crop/zoom, and coordinate mapping between image spaces.

Models answer in pixels of the image they were shown (see grounder.py). Everything here maps
crop-space coordinates back to the parent image (and ultimately the screen).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from PIL import Image, ImageGrab

from .types import BBox, Point


def set_dpi_aware() -> None:
    """Make coordinates physical pixels (matters if display scaling is not 100%)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screen_size() -> tuple[int, int]:
    set_dpi_aware()
    return (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def work_area(expected_size: tuple[int, int] | None = None) -> BBox | None:
    """Primary-monitor work area (the screen minus the taskbar and other app bars), in physical pixels.

    The taskbar band is everything outside this box, wherever the taskbar is docked. Returns the
    full screen when nothing is reserved (e.g. an auto-hidden taskbar), and None when the API
    fails or `expected_size` (a screenshot's size) does not match the live screen, in which case
    the live work area says nothing about that image.
    """
    set_dpi_aware()
    rect = _RECT()
    SPI_GETWORKAREA = 0x0030
    try:
        ok = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    if expected_size is not None and tuple(expected_size) != screen_size():
        return None
    return BBox(rect.left, rect.top, rect.right, rect.bottom)


def capture() -> Image.Image:
    """Full primary-screen screenshot in physical pixels."""
    set_dpi_aware()
    return ImageGrab.grab().convert("RGB")


# ---- crops -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Crop:
    """A zoomed sub-image of a parent image.

    `box` is the region in PARENT pixels (integers). `scale` is crop-pixels per parent-pixel
    (>1 means the crop was upscaled so small icons take up more of the model's view).
    """

    image: Image.Image
    box: BBox
    scale: float

    def point_to_parent(self, p: Point) -> Point:
        return Point(self.box.x0 + p.x / self.scale, self.box.y0 + p.y / self.scale)

    def box_to_parent(self, b: BBox) -> BBox:
        return BBox(
            self.box.x0 + b.x0 / self.scale, self.box.y0 + b.y0 / self.scale,
            self.box.x0 + b.x1 / self.scale, self.box.y0 + b.y1 / self.scale,
        )

    def point_from_parent(self, p: Point) -> Point:
        return Point((p.x - self.box.x0) * self.scale, (p.y - self.box.y0) * self.scale)


def image_bounds(img: Image.Image) -> BBox:
    return BBox(0, 0, img.width, img.height)


def crop_region(img: Image.Image, box: BBox, target_long_side: int | None = None) -> Crop:
    """Crop `box` (parent px) out of `img`, clamped to the image, optionally upscaled.

    If target_long_side is given, the crop's longer side is resized to exactly that many
    pixels (zoom in on small regions; shrink large ones). Raises ValueError for an empty box.
    """
    clamped = box.clamp(image_bounds(img))
    x0, y0, x1, y1 = clamped.as_ints()
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"empty crop region {box} for image {img.size}")
    region = img.crop((x0, y0, x1, y1))
    scale = 1.0
    if target_long_side:
        scale = target_long_side / max(region.width, region.height)
        new_size = (max(1, round(region.width * scale)), max(1, round(region.height * scale)))
        region = region.resize(new_size, Image.LANCZOS)
        # Recompute from the integer output size so mapping stays exact on both axes.
        scale = region.width / (x1 - x0)
    return Crop(image=region, box=BBox(x0, y0, x1, y1), scale=scale)


def window_around(p: Point, size: float, bounds: BBox) -> BBox:
    """A size x size window centered on p, shifted (not shrunk) to stay inside bounds.

    If bounds are smaller than size in an axis, the window covers bounds in that axis.
    """
    w = min(size, bounds.width)
    h = min(size, bounds.height)
    x0 = min(max(p.x - w / 2, bounds.x0), bounds.x1 - w)
    y0 = min(max(p.y - h / 2, bounds.y0), bounds.y1 - h)
    return BBox(x0, y0, x0 + w, y0 + h)


def point_in_image(p: Point, img: Image.Image) -> bool:
    return 0 <= p.x < img.width and 0 <= p.y < img.height
