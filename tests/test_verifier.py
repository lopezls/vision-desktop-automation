import pytest
from PIL import Image

from vision_automation.annotate import RED
from vision_automation.config import CONFIG
from vision_automation.llm import extract_json
from vision_automation.screen import window_around
from vision_automation.types import BBox, Point
from vision_automation.verifier import (
    PROMPT_PATH,
    ClaudeVerifier,
    make_marked_crop,
    parse_verdict,
)

SCREEN = BBox(0, 0, 1920, 1080)


# ---- prompt wording (Table 8) -----------------------------------------------------------

def test_verifier_prompt_matches_table_8():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    for fragment in (
        "You are given a cropped screenshot. Your task is to evaluate whether the marked element in the red box matches the target described in my instruction.",
        "1. Analyze the screenshot by describing its visible content and functionalities.",
        "- 'is_target': The marked element is the target.",
        "- 'target_elsewhere': The marked element is not the target, but it exists elsewhere.",
        "- 'target_not_found': The marked element is not the target, and it does not exist.",
        "3. If the target exists, rewrite the instruction to make it clearer.",
        "After your analysis, provide the result in JSON format:",
        "- \"result\": (str) One of 'is_target', 'target_elsewhere', or 'target_not_found'.",
        "- \"new_instruction\": (str, default null) A clearer version of the instruction.",
        "Here is my instruction: {instruction}",
    ):
        assert fragment in text, fragment
    assert "notepad" not in text.lower()


# ---- verdict parsing --------------------------------------------------------------------

def test_parse_verdict_values():
    for r in ("is_target", "target_elsewhere", "target_not_found"):
        assert parse_verdict({"result": r}).result == r
    v = parse_verdict({"result": "target_elsewhere", "new_instruction": "  the icon labelled notepad "})
    assert v.new_instruction == "the icon labelled notepad"
    assert parse_verdict({"result": "'is_target'"}).is_target
    assert parse_verdict({"result": "IS_TARGET"}).is_target


def test_unrecognised_or_missing_result_is_never_a_match():
    for bad in ({}, {"result": None}, {"result": "yes"}, {"result": "is target"}, {"result": 1}, {"result": ""}):
        v = parse_verdict(bad)
        assert v.result == "target_not_found" and not v.is_target


def test_extract_json_takes_last_object_after_analysis():
    text = 'The crop shows {an icon} next to others.\n```json\n{"result": "is_target", "new_instruction": null}\n```'
    assert extract_json(text) == {"result": "is_target", "new_instruction": None}


def test_extract_json_ignores_nested_and_stray_braces():
    assert extract_json('x { y } {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        extract_json("no json here { really")


# ---- window / marked crop geometry ------------------------------------------------------

def test_window_centered_and_shifted_at_edges():
    assert window_around(Point(960, 540), 300, SCREEN) == BBox(810, 390, 1110, 690)
    assert window_around(Point(10, 10), 300, SCREEN) == BBox(0, 0, 300, 300)
    assert window_around(Point(1915, 1075), 300, SCREEN) == BBox(1620, 780, 1920, 1080)
    assert window_around(Point(50, 50), 300, BBox(0, 0, 200, 100)) == BBox(0, 0, 200, 100)


def _screen():
    return Image.new("RGB", (1920, 1080), (30, 90, 160))


def test_marked_crop_is_zoomed_window_with_red_box_on_the_candidate():
    box = BBox(470, 470, 530, 530)
    img = make_marked_crop(_screen(), Point(500, 500), box)
    assert img.size == (CONFIG.view_long_side, CONFIG.view_long_side)
    scale = CONFIG.view_long_side / CONFIG.verify_crop_size
    # window starts at (350, 350); box left edge is at (470-350)*scale
    left = round((470 - 350) * scale)
    assert img.getpixel((left + 1, 400)) == RED
    assert img.getpixel((400, 400)) == (30, 90, 160)  # box interior is untouched
    assert img.getpixel((5, 5)) == (30, 90, 160)      # nothing else is marked


def test_marked_crop_without_box_uses_small_square_at_the_point():
    img = make_marked_crop(_screen(), Point(500, 500), None)
    scale = CONFIG.view_long_side / CONFIG.verify_crop_size
    left = round((500 - 24 - 350) * scale)
    assert img.getpixel((left + 1, 400)) == RED


def test_marked_crop_near_screen_corner_keeps_full_size():
    img = make_marked_crop(_screen(), Point(8, 8), BBox(0, 0, 40, 40))
    assert img.size == (CONFIG.view_long_side, CONFIG.view_long_side)


def test_grounder_box_larger_than_window_is_clipped_not_dropped():
    img = make_marked_crop(_screen(), Point(500, 500), BBox(0, 0, 1000, 1000))
    assert img.getpixel((2, 400)) == RED  # clipped box edge sits on the crop's left border


def test_point_outside_image_raises():
    with pytest.raises(ValueError):
        make_marked_crop(Image.new("RGB", (100, 100)), Point(500, 500), None)


# ---- verifier behaviour with a fake LLM -------------------------------------------------

class _FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.images, self.prompts = reply, error, [], []

    def note_parse_failure(self, stage, detail):
        self.failures = getattr(self, 'failures', []) + [(stage, detail)]

    def ask(self, stage, prompt, images=None, system=None):
        self.images = images
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def test_verify_sends_only_the_crop_and_fills_the_instruction():
    llm = _FakeLLM('Looks like a folder.\n{"result": "target_elsewhere", "new_instruction": "x"}')
    v = ClaudeVerifier(llm).verify(_screen(), Point(500, 500), "the thing", BBox(470, 470, 530, 530))
    assert v.result == "target_elsewhere" and v.new_instruction == "x"
    assert len(llm.images) == 1 and llm.images[0].size == (CONFIG.view_long_side,) * 2
    assert llm.prompts[0].rstrip().endswith("Here is my instruction: the thing")
    assert v.marked is llm.images[0] or v.marked.size == llm.images[0].size


def test_api_error_or_garbage_reply_is_not_a_match():
    from vision_automation.llm import LLMError
    for llm in (_FakeLLM(error=LLMError("boom")), _FakeLLM("I think it is the target!")):
        v = ClaudeVerifier(llm).verify(_screen(), Point(500, 500), "t")
        assert not v.is_target and v.result == "target_not_found"


def test_extract_json_tolerates_trailing_commas():
    text = 'analysis...\n```json\n{\n  "result": "target_elsewhere",\n  "new_instruction": "Click it",\n}\n```'
    assert extract_json(text) == {"result": "target_elsewhere", "new_instruction": "Click it"}
    assert extract_json('{"box": [1, 2, 3, 4,], "found": true,}') == {"box": [1, 2, 3, 4], "found": True}
