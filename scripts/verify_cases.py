"""Live verifier check on a saved desktop screenshot (makes real API calls).

    uv run python scripts/verify_cases.py [path/to/screenshot.png]

Cases are positions on one specific screenshot (default: the newest debug/ground_input_*.png),
so the coordinates below only fit the screenshot from Step 4/5 testing (Notepad icon at
roughly (266, 436) with Notion to its left and Recycle Bin above). Edit CASES for another
desktop. Each case reports the verdict and saves the exact crop the model saw to debug/.
"""

import logging
import sys
from pathlib import Path

from PIL import Image

from vision_automation.annotate import save_image
from vision_automation.config import DEBUG_DIR
from vision_automation.llm import LLM
from vision_automation.types import BBox, Point
from vision_automation.verifier import ClaudeVerifier

TARGET = "the Notepad desktop icon (not the taskbar)"

# name, point, red box (or None), must the verdict be is_target?
CASES = [
    ("correct: Notepad icon, grounder box", Point(266, 436), BBox(236, 411, 295, 476), True),
    ("correct: Notepad icon, point only", Point(266, 436), None, True),
    ("neighbor: Notion icon next to Notepad", Point(114, 441), BBox(83, 408, 144, 475), False),
    ("neighbor: Recycle Bin icon", Point(266, 240), BBox(231, 206, 300, 274), False),
    ("document-style file icon: rx-remote.css", Point(113, 632), BBox(80, 606, 148, 672), False),
    ("document-style file icon: Assignmen... (point only)", Point(38, 632), None, False),
    ("empty wallpaper", Point(560, 800), None, False),
    ("taskbar Notepad icon (distractor)", Point(1183, 1056), BBox(1163, 1038, 1203, 1075), False),
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        found = sorted(DEBUG_DIR.glob("ground_input_*.png"))
        if not found:
            print("no debug/ground_input_*.png found; pass a screenshot path")
            return 2
        path = found[-1]
    img = Image.open(path).convert("RGB")
    print(f"Screenshot: {path} ({img.width}x{img.height})\nTarget: {TARGET!r}\n")

    verifier = ClaudeVerifier(LLM())
    failures = 0
    for name, point, box, expect_target in CASES:
        v = verifier.verify(img, point, TARGET, box)
        ok = v.is_target == expect_target
        failures += not ok
        crop_path = save_image(v.marked, "verify_case")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       verdict={v.result}  expected "
              f"{'is_target' if expect_target else 'NOT is_target'}  crop={crop_path.name}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
