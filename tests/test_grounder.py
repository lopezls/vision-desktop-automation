import pytest
from PIL import Image

from vision_automation.grounder import MAX_SENT_SIDE, PROMPT_PATH, parse_ground_reply, prepare_for_model
from vision_automation.types import BBox


def test_valid_reply_is_pixels_of_sent_image():
    r = parse_ground_reply({"found": True, "box": [100, 200, 300, 400], "confidence": 0.9, "note": "x"}, 1000, 800)
    assert r.found and r.box == BBox(100, 200, 300, 400) and r.confidence == 0.9


def test_reply_is_mapped_back_through_the_downscale():
    # sent at 0.5x: a box at (100,200)-(300,400) in the sent image is (200,400)-(600,800) originally
    r = parse_ground_reply({"found": True, "box": [100, 200, 300, 400]}, 960, 540, scale=0.5)
    assert r.box == BBox(200, 400, 600, 800)


def test_pixel_values_above_1000_are_valid_for_big_images():
    # regression: the old 0-1000 check rejected real taskbar boxes like [1036, 1038, 1066, 1076]
    r = parse_ground_reply({"found": True, "box": [1036, 1038, 1066, 1076]}, 1568, 882 + 200)
    assert r.found


def test_not_found():
    r = parse_ground_reply({"found": False, "box": None}, 1920, 1080)
    assert not r.found and r.box is None


def test_malformed_box_is_not_found_not_a_crash():
    for bad in (None, [1, 2, 3], "box", [1, 2, 3, "x"], [True, 0, 5, 5]):
        assert not parse_ground_reply({"found": True, "box": bad}, 1920, 1080).found


def test_box_outside_image_rejected():
    assert not parse_ground_reply({"found": True, "box": [0, 0, 2500, 500]}, 1920, 1080).found
    assert not parse_ground_reply({"found": True, "box": [0, 0, 500, 1200]}, 1920, 1080).found


def test_degenerate_box_rejected():
    assert not parse_ground_reply({"found": True, "box": [500, 500, 500, 600]}, 1920, 1080).found


def test_swapped_corners_fixed_and_small_overshoot_clamped():
    r = parse_ground_reply({"found": True, "box": [300, 400, 100, 200]}, 1000, 1000)
    assert r.box == BBox(100, 200, 300, 400)
    r = parse_ground_reply({"found": True, "box": [900, 900, 1005, 1005]}, 1000, 1000)
    assert r.found and r.box.x1 == 1000 and r.box.y1 == 1000


def test_prepare_for_model_downscales_large_images_only():
    small = Image.new("RGB", (800, 600))
    assert prepare_for_model(small) == (small, 1.0)
    big = Image.new("RGB", (1920, 1080))
    sent, scale = prepare_for_model(big)
    assert max(sent.size) == MAX_SENT_SIDE and scale == pytest.approx(sent.width / 1920)


def test_grounder_prompt_is_generic_and_states_pixel_size():
    text = PROMPT_PATH.read_text(encoding="utf-8").lower()
    for word in ("notepad", "desktop", "icon column", "windows", "1000", "normalized"):
        assert word not in text
    assert "{width}" in text and "{height}" in text
