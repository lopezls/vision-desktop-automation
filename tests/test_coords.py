import pytest
from PIL import Image

from vision_automation.screen import crop_region
from vision_automation.types import BBox, Point


def test_crop_without_zoom_maps_by_offset():
    img = Image.new("RGB", (1920, 1080))
    c = crop_region(img, BBox(100, 200, 500, 600))
    assert c.scale == 1.0
    assert c.image.size == (400, 400)
    assert c.point_to_parent(Point(10, 20)) == Point(110, 220)


def test_crop_with_zoom_round_trips():
    img = Image.new("RGB", (1920, 1080))
    c = crop_region(img, BBox(1700, 900, 1900, 1050), target_long_side=800)
    assert c.image.width == 800 and c.scale == pytest.approx(4.0)
    parent = Point(1800, 1000)
    local = c.point_from_parent(parent)
    back = c.point_to_parent(local)
    assert back.x == pytest.approx(parent.x) and back.y == pytest.approx(parent.y)


def test_crop_center_maps_to_region_center():
    img = Image.new("RGB", (1920, 1080))
    c = crop_region(img, BBox(0, 0, 200, 100), target_long_side=1000)
    center = Point(c.image.width / 2, c.image.height / 2)
    assert c.point_to_parent(center) == Point(100, 50)


def test_crop_clamps_to_image_bounds():
    img = Image.new("RGB", (1920, 1080))
    c = crop_region(img, BBox(-50, -50, 100, 100))
    assert c.box == BBox(0, 0, 100, 100)
    c2 = crop_region(img, BBox(1800, 1000, 2500, 2500))
    assert c2.box == BBox(1800, 1000, 1920, 1080)


def test_crop_outside_image_raises():
    img = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError):
        crop_region(img, BBox(200, 200, 300, 300))


def test_nested_crops_compose():
    img = Image.new("RGB", (1920, 1080))
    outer = crop_region(img, BBox(1000, 500, 1600, 800), target_long_side=1200)  # scale 2
    inner = crop_region(outer.image, BBox(200, 100, 600, 300), target_long_side=800)  # scale 2
    p_inner = Point(400, 200)  # in inner crop pixels
    p_screen = outer.point_to_parent(inner.point_to_parent(p_inner))
    # inner px 400,200 -> outer px 200+200, 100+100 = 400,200 -> screen 1000+200, 500+100
    assert p_screen == Point(1200, 600)
