"""Drawing helpers for debug images and the annotated deliverable screenshots."""

from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import DEBUG_DIR
from .types import BBox, Point

RED = (255, 32, 32)
GREEN = (0, 200, 90)
YELLOW = (255, 200, 0)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(img: Image.Image, box: BBox, color=RED, width: int = 3, label: str | None = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle(box.as_ints(), outline=color, width=width)
    if label:
        f = _font(16)
        x, y = round(box.x0), round(box.y0)
        tb = d.textbbox((x, max(0, y - 20)), label, font=f)
        d.rectangle(tb, fill=color)
        d.text((tb[0], tb[1]), label, fill="white", font=f)
    return out


def draw_marker(img: Image.Image, p: Point, color=RED, radius: int = 14, box_half: int = 28) -> Image.Image:
    """Red box + crosshair at p. Used both for the verifier's marked crop and final output."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    x, y = round(p.x), round(p.y)
    d.rectangle((x - box_half, y - box_half, x + box_half, y + box_half), outline=color, width=3)
    d.line((x - radius, y, x + radius, y), fill=color, width=2)
    d.line((x, y - radius, x, y + radius), fill=color, width=2)
    return out


def annotate_result(screenshot: Image.Image, point: Point, box: BBox | None = None,
                    label: str = "Notepad icon") -> Image.Image:
    out = screenshot.copy()
    if box is not None:
        out = draw_box(out, box, GREEN, 3)
    out = draw_marker(out, point)
    d = ImageDraw.Draw(out)
    text = f"{label}: click point ({round(point.x)}, {round(point.y)})"
    f = _font(22)
    tx = min(max(8, round(point.x) - 120), max(8, out.width - 420))
    ty = round(point.y) + 40
    if ty > out.height - 36:
        ty = round(point.y) - 70
    tb = d.textbbox((tx, ty), text, font=f)
    d.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0))
    d.text((tx, ty), text, fill=YELLOW, font=f)
    return out


def save_image(img: Image.Image, name: str, directory: Path = DEBUG_DIR, timestamp: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{datetime.now():%Y%m%d_%H%M%S}" if timestamp else name
    path = directory / f"{stem}.png"
    img.save(path)
    return path
