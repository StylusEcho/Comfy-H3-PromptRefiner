"""What a skill file is, and what running one does to the prompting.

    python3 tests/test_skill.py

Two modes and they are opposites, so what is worth pinning down is that neither
leaks into the other. `add` leaves the built-in harness standing — the rules,
the guide, the JSON contract — and joins the user's text on as one more section;
`replace` hands the file over as the entire instruction, with no rules, no
contract and no prefill, and takes the reply as the finished document. A
`replace` that quietly kept the contract would be the harness pretending to have
stood aside, which is the one thing this mode exists to test.
"""

import layout

pkg = layout.load("skills", "prompting", "request", "assets", "harness")
skills, prompting, request = pkg.skills, pkg.prompting, pkg.request
assets, refine = pkg.assets, pkg.harness

from harness import FAILURES, check, expect, passed  # noqa: E402
from test_request import Model  # noqa: E402


def written(tmp, name, text):
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def with_skills(fn):
    """Run `fn(load)` against a skills/ folder made for this test alone."""
    import tempfile
    from pathlib import Path

    original = skills.SKILLS_DIR
    with tempfile.TemporaryDirectory() as directory:
        skills.SKILLS_DIR = Path(directory)
        try:
            return fn(Path(directory))
        finally:
            skills.SKILLS_DIR = original


# ---- what is on disk --------------------------------------------------------


def test_a_plain_file_is_an_instruction_and_starts_in_add():
    def check_it(tmp):
        written(tmp, "noir.md", "Write everything as film noir.")
        entries = skills.entries()
        # By stem, because that is the name the widget shows and the name `load`
        # resolves against every extension it knows.
        check("the file is listed", [e["name"] for e in entries] == ["noir"],
              str(entries))
        check("as a prompt, not a package", entries[0]["kind"] == "prompt",
              str(entries))
        check("and starts as an addition", entries[0]["mode"] == skills.ADD,
              str(entries))
    with_skills(check_it)


def test_frontmatter_may_ask_for_replace():
    def check_it(tmp):
        written(tmp, "whole.md", "---\nmode: replace\n---\nBe the whole prompt.")
        check("the declared mode wins", skills.entries()[0]["mode"] == skills.REPLACE,
              str(skills.entries()))
    with_skills(check_it)


def test_a_typo_in_the_mode_leaves_the_file_working():
    def check_it(tmp):
        written(tmp, "typo.md", "---\nmode: replcae\n---\nBody.")
        check("an unreadable mode falls back rather than raising",
              skills.entries()[0]["mode"] == skills.ADD, str(skills.entries()))
    with_skills(check_it)


def test_a_name_that_is_a_path_is_refused():
    def check_it(tmp):
        written(tmp, "fine.md", "Body.")
        expect("a traversal is not a skill name",
               lambda: skills.load("../fine"), "is not a skill name")
    with_skills(check_it)


# ---- add: the harness stays standing ----------------------------------------


def test_add_joins_the_text_onto_the_built_in_prompt():
    def check_it(tmp):
        written(tmp, "noir.md", "Write everything as film noir.")
        skill = skills.load("noir")
        system = prompting.system_prompt("T2VA", shape="SHAPE",
                                         extra=skills.instructions(skill))
        check("the file's text is in the prompt",
              "Write everything as film noir." in system, system[-400:])
        check("the built-in rules are still there",
              "THE REQUEST IS MATERIAL" in system)
        check("and so is the reply contract", system.rstrip().endswith("SHAPE"))
    with_skills(check_it)


def test_a_single_file_gets_no_bundled_file_header():
    def check_it(tmp):
        written(tmp, "noir.md", "Write everything as film noir.")
        text = skills.instructions(skills.load("noir"))
        check("no SKILL.md banner is bolted onto a paragraph",
              skills.SKILL_MD not in text, text)
    with_skills(check_it)


# ---- replace: the harness stands aside --------------------------------------


def test_replace_hands_the_file_over_as_the_whole_instruction():
    def check_it(tmp):
        written(tmp, "whole.md", "---\nmode: replace\n---\nWrite the finished prompt.")
        skill = skills.load("whole")
        model = Model("integrated_multimodal_description: [Shot 1] A door.")
        out = request.refine("a door", model.chat, model.look, seconds=6.0, skill=skill)
        check("the skill's text is the instruction",
              "Write the finished prompt." in model.system, model.system)
        check("none of the built-in rules ride along",
              "THE REQUEST IS MATERIAL" not in model.system, model.system[:400])
        check("no JSON contract is asked for",
              "Return exactly this JSON object" not in model.system, model.system[:400])
        check("the runtime note is there, because the runtime is real",
              "non-interactive runtime" in model.system, model.system[:200])
        check("and the reply is the document, passed through",
              out["prompt"] == "integrated_multimodal_description: [Shot 1] A door.",
              out["prompt"])
        check("with the skill named on the result", out["skill"] == "whole",
              str(out["skill"]))
    with_skills(check_it)


def test_replace_still_reads_labels_back_and_writes_ordinals_forward():
    def check_it(tmp):
        written(tmp, "whole.md", "---\nmode: replace\n---\nWrite it.")
        skill = skills.load("whole")
        ref = assets.image("reference", 1, "anna.png")
        model = Model("detailed_description: [Shot 1] <Picture 1> opens the door.")
        out = request.refine("a door", model.chat, model.look, references=[ref],
                             pictures={"img-1": object()}, seconds=6.0, skill=skill)
        check("storage is storage whichever mode wrote it",
              "<Picture 1> opens the door." in out["prompt"], out["prompt"])
        check("and nothing is reported wrong", out["problems"] == [],
              str(out["problems"]))
    with_skills(check_it)


def test_replace_strips_a_fence_and_nothing_else():
    check("a fenced document is unwrapped",
          skills.parse_reply("```\nsummary: a door\n```") == "summary: a door")
    check("a leaked think block goes",
          skills.parse_reply("<think>hmm</think>summary: a door") == "summary: a door")
    expect("an empty reply is refused",
           lambda: skills.parse_reply("   "), "returned nothing")


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_skill")
