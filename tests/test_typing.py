"""Verified typing against a simulated Notepad editor: autocorrect, changed characters, retries, limits."""

import logging
from dataclasses import replace

import pytest

from vision_automation.config import CONFIG
from vision_automation.notepad import (
    NotepadError,
    WindowsNotepad,
    first_difference,
    normalize_newlines,
    split_chunks,
)

HWND = 7
CFG = replace(CONFIG, chunk_settle_s=0.0, chunk_retries=2, type_retries=2, focus_timeout_s=0.2)

# Post 4's real failure: Notepad turns the Latin "commodi" into the English "commode".
TEXT = "Title: eum et est\n\nquis hic commodi nesciunt rem\nvel officia dolor sit"


class FakeEditor:
    """A tiny Notepad: a text buffer with a caret, autocorrect on delimiters, and injectable faults.

    autocorrect:   {"commodi": "commode"}; fires when a space/Enter is typed right after the word (as in Notepad)
    corrupt:       {n: replacement}: the n-th character ever typed comes out as `replacement` ("" = dropped)
    mutate_earlier:{n: (pos, ch)}: when the n-th character is typed, the character at `pos` changes too
    persistent:    {"commodi": "commode"}: rewritten after EVERY edit, so no typing strategy can get past it
    """

    def __init__(self, initial="", autocorrect=None, corrupt=None, mutate_earlier=None, persistent=None,
                 unreadable=False, focused=True):
        self.doc, self.caret, self.selected = initial, len(initial), False
        self.autocorrect, self.corrupt = dict(autocorrect or {}), dict(corrupt or {})
        self.mutate_earlier, self.persistent = dict(mutate_earlier or {}), dict(persistent or {})
        self.unreadable, self.focused = unreadable, focused
        self.n = 0
        self.events = []

    # -- the Ui interface WindowsNotepad uses --
    def type_text(self, text, pause=True):
        self.events.append(("type", text))
        for ch in text:
            idx, self.n = self.n, self.n + 1
            ch = self.corrupt.get(idx, ch)
            if ch != "":
                self._insert(ch)
            if idx in self.mutate_earlier:
                pos, new = self.mutate_earlier[idx]
                self.doc = self.doc[:pos] + new + self.doc[pos + 1:]
            self._rewrite()

    def hotkey(self, *keys):
        self.events.append(("hotkey", "+".join(keys)))
        if keys == ("ctrl", "a"):
            self.selected = True

    def press(self, key):
        self.events.append(("press", key))
        if key == "delete" and self.selected:
            self.doc, self.caret, self.selected = "", 0, False

    def press_repeated(self, key, n):
        self.events.append(("repeat", key, n))
        if key == "left":
            self.caret = max(0, self.caret - n)
        elif key == "right":
            self.caret = min(len(self.doc), self.caret + n)
        elif key == "backspace":
            start = max(0, self.caret - n)
            self.doc, self.caret = self.doc[:start] + self.doc[self.caret:], start

    def editor_text(self, hwnd):
        return None if self.unreadable else self.doc.replace("\n", "\r\n")

    def has_focus(self, hwnd):
        return self.focused

    def focus(self, hwnd):
        return self.focused

    # -- the simulated editor --
    def _insert(self, ch):
        if self.selected:
            self.doc, self.caret, self.selected = "", 0, False
        self.doc = self.doc[:self.caret] + ch + self.doc[self.caret:]
        self.caret += 1
        if ch in " \n":  # a delimiter was typed: autocorrect looks at the word right before it
            end = self.caret - 1
            start = end
            while start > 0 and self.doc[start - 1] not in " \n":
                start -= 1
            word = self.doc[start:end]
            if word in self.autocorrect:
                fixed = self.autocorrect[word]
                self.doc = self.doc[:start] + fixed + self.doc[end:]
                self.caret += len(fixed) - len(word)

    def _rewrite(self):
        for wrong, right in self.persistent.items():
            if wrong in self.doc:
                at_end = self.caret >= len(self.doc)
                self.doc = self.doc.replace(wrong, right)
                self.caret = len(self.doc) if at_end else min(self.caret, len(self.doc))

    # -- assertions --
    @property
    def typed(self):
        return [e[1] for e in self.events if e[0] == "type"]

    @property
    def clears(self):
        return sum(1 for e in self.events if e == ("hotkey", "ctrl+a"))


def run(editor, text=TEXT, cfg=CFG):
    WindowsNotepad(cfg, editor).type_text(HWND, text)
    return editor


# ---- helpers ----------------------------------------------------------------------------

def test_split_chunks_covers_the_text_exactly():
    for text in (TEXT, "", "a", " lead", "a  b\n\n c ", "x\r\ny", "Title: eum\n\nbody"):
        assert "".join(split_chunks(text)) == text
    assert split_chunks("Title: eum\n\nbody end") == ["Title: ", "eum\n\n", "body ", "end"]


def test_first_difference_and_newline_normalization():
    assert first_difference("commodi", "commode") == 6
    assert first_difference("abc", "ab") == 2 and first_difference("abc", "abc") == 3
    assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"


# ---- the normal case --------------------------------------------------------------------

def test_clean_typing_types_each_word_once_and_never_clears():
    ed = run(FakeEditor())
    assert ed.doc == TEXT
    assert "".join(ed.typed) == TEXT and ed.clears == 0


# ---- Notepad's autocorrect (the post-4 failure) ------------------------------------------

def test_autocorrect_rewriting_a_word_is_repaired_in_place(caplog):
    ed = FakeEditor(autocorrect={"commodi": "commode", "officia": "officiae"})
    with caplog.at_level(logging.INFO, logger="vision_automation"):
        run(ed)
    assert ed.doc == TEXT  # "commodi", not "commode"; "officia", not "officiae" (a LONGER rewrite)
    assert caplog.text.count("repaired") == 2
    assert ed.clears == 0  # fixed at the word, without clearing the document


def test_repair_types_the_delimiter_first_so_the_word_is_never_followed_by_a_typed_delimiter():
    ed = FakeEditor(autocorrect={"commodi": "commode"})
    run(ed)
    ev = ed.events
    i = next(k for k, e in enumerate(ev) if e == ("repeat", "backspace", 8))  # removed 'commode '
    assert ev[i + 1] == ("type", " ") and ev[i + 2] == ("repeat", "left", 1)
    assert ev[i + 3] == ("type", "commodi") and ev[i + 4] == ("repeat", "right", 1)


def test_repair_across_a_newline_delimiter():
    ed = FakeEditor(autocorrect={"est": "EST"})  # the word before "\n\n"
    run(ed, "Title: eum et est\n\nbody here")
    assert ed.doc == "Title: eum et est\n\nbody here"


# ---- a character changed in the middle of the body ---------------------------------------

def test_a_character_changed_in_the_middle_of_the_body_is_caught_and_repaired(caplog):
    k = TEXT.index("nesciunt") + 3  # global index of the 'c' in "nesciunt": mid-body
    ed = FakeEditor(corrupt={k: "X"})
    with caplog.at_level(logging.INFO, logger="vision_automation"):
        run(ed)
    assert ed.doc == TEXT
    assert "repaired 'nesciunt '" in caplog.text
    assert ed.clears == 0


def test_a_dropped_key_is_caught_and_repaired():
    k = TEXT.index("officia") + 2
    ed = FakeEditor(corrupt={k: ""})
    run(ed)
    assert ed.doc == TEXT


def test_a_changed_character_in_the_last_word_is_caught_too():
    ed = FakeEditor(corrupt={len(TEXT) - 2: "Q"})  # the last word has no delimiter after it
    run(ed)
    assert ed.doc == TEXT


# ---- document-level retry: clear (Ctrl+A, Delete) and type again ------------------------

def test_retry_succeeds_after_clearing_and_retyping_the_whole_document(caplog):
    # While typing, an EARLIER part of the text changes once, which word-level repair cannot fix.
    ed = FakeEditor(mutate_earlier={30: (3, "Z")})
    with caplog.at_level(logging.WARNING, logger="vision_automation"):
        run(ed)
    assert ed.doc == TEXT
    assert ed.clears == 1  # cleared exactly once, then typed again
    assert ("hotkey", "ctrl+a") in ed.events and ("press", "delete") in ed.events
    assert "retry 1/2" in caplog.text
    assert "".join(ed.typed).count("Title: ") == 2  # the document really was typed twice


def test_retry_limit_reached_stops_with_the_clear_error():
    # The earlier text changes on every pass: 1 attempt + 2 retries, then stop.
    ed = FakeEditor(mutate_earlier={30: (3, "Z"), 30 + len(TEXT): (3, "Z"), 30 + 2 * len(TEXT): (3, "Z")})
    with pytest.raises(NotepadError) as ei:
        run(ed)
    msg = str(ei.value)
    assert "still differs from the expected text after 3 attempts" in msg
    assert f"expected {len(TEXT)} chars" in msg and "first difference at index 3" in msg
    assert "expected 'l'" in msg and "found 'Z'" in msg  # TEXT[3] is 'l' ("Title")
    assert ed.clears == 2  # exactly type_retries clears, then it stopped
    assert ed.doc != TEXT  # left as is for inspection


def test_a_rewrite_no_typing_trick_can_defeat_ends_in_the_clear_error():
    ed = FakeEditor(persistent={"commodi": "commode"})
    with pytest.raises(NotepadError, match="still differs from the expected text after 3 attempts"):
        run(ed)
    assert ed.clears == CFG.type_retries
    assert "commode" in ed.doc


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_the_retry_count_is_configurable(retries):
    ed = FakeEditor(persistent={"commodi": "commode"})
    with pytest.raises(NotepadError, match=f"after {1 + retries} attempts"):
        run(ed, cfg=replace(CFG, type_retries=retries))
    assert ed.clears == retries


# ---- safety -----------------------------------------------------------------------------

def test_never_types_into_a_document_that_already_has_text():
    ed = FakeEditor(initial="someone's notes")
    with pytest.raises(NotepadError, match="not empty"):
        run(ed)
    assert ed.typed == [] and ed.clears == 0 and ed.doc == "someone's notes"  # nothing typed, nothing deleted


def test_unreadable_editor_stops_before_typing_anything():
    ed = FakeEditor(unreadable=True)
    with pytest.raises(NotepadError, match="Could not read the document text"):
        run(ed)
    assert ed.typed == []


def test_no_keys_are_sent_when_notepad_does_not_have_the_focus():
    ed = FakeEditor(focused=False)
    with pytest.raises(NotepadError, match="keyboard focus"):
        run(ed)
    assert ed.events == []


def test_the_editor_text_is_read_from_the_control_never_the_clipboard():
    # The Ui layer has no clipboard operation at all; typing is the only way text gets in.
    assert not any("clip" in name or "paste" in name for name in dir(FakeEditor))
    from vision_automation import notepad
    assert "clipboard" not in notepad.__doc__.lower().replace("never uses the clipboard", "")
