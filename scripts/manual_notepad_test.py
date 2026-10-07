"""Step 1b: manual Notepad type-and-save test.

Open Notepad BY HAND first (blank note, click into the text area), then run:
    uv run python scripts/manual_notepad_test.py

You get a 5 second countdown to make sure Notepad is focused. The script types a sample
post, saves it as post_test.txt in Desktop/tjm-project, and closes Notepad.
FAILSAFE is on: slam the mouse into any screen corner to abort.
"""

import ctypes
import sys
import time

import pyautogui

from vision_automation.paths import project_dir

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1
TYPE_INTERVAL = 0.03  # 0.01 dropped a character in the first run

TITLE = "sunt aut facere repellat provident"
BODY = "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam"


def main() -> int:
    ctypes.windll.user32.SetProcessDPIAware()
    target = project_dir()
    filename = "post_test.txt"
    print(f"Will save to: {target / filename}")
    for n in range(5, 0, -1):
        print(f"Typing in {n}s... (click into the Notepad text area; mouse to a corner aborts)")
        time.sleep(1)

    expected = f"Title: {TITLE}\n\n{BODY}"
    pyautogui.write(expected, interval=TYPE_INTERVAL)
    time.sleep(0.5)

    pyautogui.hotkey("ctrl", "shift", "s")  # Save As
    time.sleep(1.5)
    pyautogui.write(str(target / filename), interval=0.01)  # full path in the filename box
    pyautogui.press("enter")
    time.sleep(1.5)
    # Overwrite prompt (only if the file already exists): Yes is Alt+Y
    # We do not blindly confirm; just report what to look for.
    saved = (target / filename).exists()
    print(f"File exists after save: {saved}")
    if saved:
        actual = (target / filename).read_text(encoding="utf-8").replace("\r\n", "\n")
        print("Content:\n" + actual)
        print("CONTENT MATCHES" if actual == expected else "CONTENT MISMATCH (dropped/garbled keys)")
        pyautogui.hotkey("alt", "f4")  # close Notepad
    else:
        print("Save did not complete (overwrite prompt or dialog differences?). Notepad left open.")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
