import pytest

from vision_automation.planner import (
    PRESENCE_NOTE_CROP,
    PRESENCE_NOTE_FULL,
    PROMPT_DIR,
    ClaudePlanner,
    is_safe_dismiss_label,
    parse_popup_reply,
    parse_position_reply,
)


# ---- position reply parsing ------------------------------------------------------------

def test_parses_the_papers_example_in_order():
    text = ("The <element>shortcut link</element> is most likely to be found in the "
            "<area>Settings window</area>, in the <area>tools panel</area>, next to the "
            "<neighbor>Search button</neighbor>.")
    r = parse_position_reply(text)
    assert [(h.kind, h.text) for h in r.hints] == [
        ("element", "shortcut link"), ("area", "Settings window"),
        ("area", "tools panel"), ("neighbor", "Search button")]
    assert [h.rank for h in r.hints] == [0, 1, 2, 3]
    assert not r.no_target


def test_of_kind_filters():
    r = parse_position_reply("<area>left icon column</area> <neighbor>Recycle Bin icon</neighbor> <element>x</element>")
    assert [h.text for h in r.of_kind("area", "neighbor")] == ["left icon column", "Recycle Bin icon"]


def test_tolerates_case_whitespace_and_newlines():
    r = parse_position_reply("<AREA> the left\n icon  column </Area>\n<Neighbor >Recycle Bin</ neighbor>")
    assert [(h.kind, h.text) for h in r.hints] == [("area", "the left icon column"), ("neighbor", "Recycle Bin")]


def test_tolerates_missing_closing_tags():
    r = parse_position_reply("Look in the <area>top-left desktop corner\nnext to the <neighbor>Recycle Bin icon")
    assert [(h.kind, h.text) for h in r.hints] == [("area", "top-left desktop corner"), ("neighbor", "Recycle Bin icon")]


def test_mixed_closed_and_unclosed_keep_document_order():
    r = parse_position_reply("<area>A</area> then <neighbor>B then <area>C</area>")
    assert [h.text for h in r.hints] == ["A", "B then", "C"]


def test_duplicates_dropped_empty_skipped():
    r = parse_position_reply("<area>left column</area><area>Left Column</area><area>  </area><neighbor></neighbor>")
    assert [h.text for h in r.hints] == ["left column"]


def test_no_target_is_a_failed_level_not_a_crash():
    for text in ('No target', '"No target"', 'No target.', 'I could not find it. No target.', '', 'blah blah'):
        r = parse_position_reply(text)
        assert r.no_target and r.hints == []


def test_hints_win_over_stray_no_target_text():
    r = parse_position_reply('Not "No target": it is in the <area>left column</area>.')
    assert not r.no_target and len(r.hints) == 1


# ---- prompt wording ---------------------------------------------------------------------

def test_planner_prompt_keeps_appendix_c_wording_and_tags():
    text = (PROMPT_DIR / "planner.md").read_text(encoding="utf-8")
    for fragment in (
        "I want to identify a UI element that best matches my instruction.",
        "list the UI elements that might appear next to the target",
        'please output "No target"',
        "List the possible regions in descending order of probability.",
        'References such as "Other icons" and "window" are NOT allowed.',
        "<element></element>", "<area></area>", "<neighbor></neighbor>",
        "<element>shortcut link</element>", "<area>Settings window</area>",
        "<area>tools panel</area>", "<neighbor>Search button</neighbor>",
        "Do not speculate about operations that could change the screenshot.",
        "Instruction: {instruction}",
    ):
        assert fragment in text, fragment
    assert "{presence_note}" in text
    assert "notepad" not in text.lower()


def test_presence_note_full_is_verbatim_and_crop_differs():
    assert PRESENCE_NOTE_FULL == "The target UI element is guaranteed to be present in the screenshot."
    assert "No target" in PRESENCE_NOTE_CROP and "guaranteed" not in PRESENCE_NOTE_CROP


class _FakeLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def note_parse_failure(self, stage, detail):
        self.failures = getattr(self, 'failures', []) + [(stage, detail)]

    def ask(self, stage, prompt, images=None, system=None):
        self.prompts.append(prompt)
        return self.text


def test_infer_positions_fills_presence_note_and_instruction():
    from PIL import Image
    llm = _FakeLLM("<area>left column</area>")
    p = ClaudePlanner(llm)
    img = Image.new("RGB", (10, 10))
    p.infer_positions(img, "the thing")
    p.infer_positions(img, "the thing", is_crop=True)
    assert PRESENCE_NOTE_FULL in llm.prompts[0] and PRESENCE_NOTE_CROP not in llm.prompts[0]
    assert PRESENCE_NOTE_CROP in llm.prompts[1] and "guaranteed" not in llm.prompts[1]
    assert llm.prompts[0].rstrip().endswith("Instruction: the thing")
    assert "{" not in llm.prompts[0]


# ---- popup safety -----------------------------------------------------------------------

@pytest.mark.parametrize("label", ["X", "x", "×", "✕", "Close", "close.", " Dismiss ", "Not now", "No thanks",
                                   "Maybe later", "Remind me later", "Cancel"])
def test_safe_labels_allowed(label):
    assert is_safe_dismiss_label(label)


@pytest.mark.parametrize("label", ["OK", "Ok", "Yes", "Allow", "Accept", "Install", "Update now", "Continue",
                                   "Run", "Sign in", "Restart", "Got it", "Close and install updates",
                                   "Don't allow", "", None, "Close all"])
def test_unsafe_labels_refused(label):
    assert not is_safe_dismiss_label(label)


def test_popup_close_with_safe_label_allowed():
    r = parse_popup_reply({"blocked": True, "action": "close", "popup": "update banner",
                           "control_label": "Not now", "control_description": "bottom-right of the banner"})
    assert r.action == "close" and r.control_label == "Not now"


def test_popup_close_with_unsafe_label_downgraded_to_abort():
    r = parse_popup_reply({"blocked": True, "action": "close", "control_label": "OK",
                           "control_description": "center of dialog"})
    assert r.action == "abort" and "refusing" in r.reason


def test_popup_close_without_location_aborts():
    r = parse_popup_reply({"blocked": True, "action": "close", "control_label": "X", "control_description": None})
    assert r.action == "abort"


def test_popup_not_blocked_never_clicks():
    r = parse_popup_reply({"blocked": False, "action": "close", "control_label": "X", "control_description": "corner"})
    assert not r.blocked and r.action == "none"


def test_popup_blocked_with_garbage_or_none_action_aborts():
    for action in (None, "none", "click_ok", 5):
        r = parse_popup_reply({"blocked": True, "action": action})
        assert r.action == "abort"


def test_popup_non_boolean_blocked_treated_as_not_blocked():
    assert not parse_popup_reply({"blocked": "yes", "action": "close"}).blocked
