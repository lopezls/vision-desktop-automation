"""CLI defaults. These only parse arguments; nothing here touches the screen, the keyboard or the API."""

import pytest

from vision_automation.cli import build_parser
from vision_automation.config import POST_COUNT


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_run_defaults_to_the_assignments_ten_posts():
    assert POST_COUNT == 10
    assert parse("run").posts == 10


@pytest.mark.parametrize("n", [1, 3, 10, 25])
def test_posts_flag_still_overrides_the_default(n):
    assert parse("run", "--posts", str(n)).posts == n


def test_run_help_states_the_new_default(capsys):
    with pytest.raises(SystemExit):
        parse("run", "--help")
    text = " ".join(capsys.readouterr().out.split())  # argparse wraps lines
    assert "default 10" in text
    assert "default 1;" not in text and "default 1)" not in text


def test_other_run_defaults_are_unchanged():
    args = parse("run")
    assert args.countdown == 5 and args.description is None


def test_non_integer_posts_is_rejected():
    with pytest.raises(SystemExit):
        parse("run", "--posts", "many")


def test_other_commands_still_parse():
    assert parse("check").cmd == "check"
    assert parse("locate", "--name", "icon_center").name == "icon_center"
