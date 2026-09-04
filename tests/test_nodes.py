"""The three nodes, driven without a ComfyUI under them.

    python3 tests/test_nodes.py

`nodes.py` is the only module that reaches for ComfyUI, and it does so inside
the functions that need it — which is what this suite is really checking. The
widgets are declared, the backend dict travels from one node to the other, and
one press runs end to end against a scripted model, all with no torch, no
numpy and no server on the path.

Anything that needs a real tensor is `_pil`'s, and `_pil` is the one function
here that imports numpy. It is exercised in `test_frames_are_read_in_batch_order`
only where numpy is actually installed, and skipped where it is not — a suite
that cannot run without the dependency it exists to avoid would be its own
counterexample.
"""

import layout

pkg = layout.load("nodes", "harness", "request", "skills", "prompting")
nodes, refine, request = pkg.nodes, pkg.harness, pkg.request
skills, prompting = pkg.skills, pkg.prompting

from harness import FAILURES, check, expect, passed  # noqa: E402
from test_request import Model, reply  # noqa: E402


def press(model, refiner=None, **kwargs):
    """One `H3PromptRefiner.run`, with the backend swapped for `model`.

    `_backend` is what a real press resolves to `(chat, look)`; replacing it is
    the whole of the stubbing, because everything below it already takes those
    two as arguments.
    """
    original = nodes._backend
    nodes._backend = lambda refiner: (model.chat, model.look)
    try:
        settings = dict(prompt="a red door in the rain", seconds=6.0, template="auto",
                        cut_shots=True, language="English", temperature=0.3, seed=0,
                        reply_tokens=refine.NUM_PREDICT)
        settings.update(kwargs)
        return nodes.H3PromptRefiner().run(refiner or {"where": "local", "model": "q"},
                                           **settings)
    finally:
        nodes._backend = original


# ---- the widgets -------------------------------------------------------------


def test_every_widget_is_declared_without_a_comfyui():
    types = nodes.H3PromptRefiner.INPUT_TYPES()
    check("the backend comes in on a socket", types["required"]["refiner"][0]
          == nodes.REFINER)
    check("the template widget offers what the prompting has",
          list(types["required"]["template"][0]) == list(prompting.TEMPLATES),
          str(types["required"]["template"][0]))
    check("the three picture sockets are optional",
          {"start_frame", "end_frame", "references"} <= set(types["optional"]))
    check("and every output is named",
          len(nodes.H3PromptRefiner.RETURN_NAMES)
          == len(nodes.H3PromptRefiner.RETURN_TYPES) == 5)


def test_the_server_node_asks_for_a_variable_name_and_not_a_key():
    widgets = nodes.H3RefinerServer.INPUT_TYPES()["required"]
    check("there is no key widget", "api_key" not in widgets, str(sorted(widgets)))
    check("only the name of one", widgets["api_key_env"][1]["default"]
          == pkg.nodes.remote.KEY_ENV)


def test_a_backend_node_hands_over_settings_and_no_prose():
    (local,) = nodes.H3RefinerLocalModel().backend("qwen3vl-4b.safetensors")
    check("the local backend names where it runs", local["where"] == "local")
    check("and which model", local["model"] == "qwen3vl-4b.safetensors")
    (server,) = nodes.H3RefinerServer().backend("gpt-x", "http://localhost:1234/v1",
                                                "MY_KEY", True)
    check("the server backend carries the variable's name",
          server["key_env"] == "MY_KEY")
    check("and no key", all("sk-" not in str(v) for v in server.values()), str(server))


def test_a_missing_backend_says_what_to_plug_in():
    expect("an unwired socket is named, not crashed on",
           lambda: nodes._backend(None), "no refiner backend is connected")


# ---- one press ---------------------------------------------------------------


def test_a_press_returns_the_document_and_its_parts():
    model = Model(reply())
    prompt, body, soundscape, music, notes = press(model)
    check("the document is first", "integrated_multimodal_description:" in prompt, prompt)
    check("the prose alone is second", "A red door opens onto rain." in body, body)
    check("the soundscape is its own output", soundscape == "Rain on stone.", soundscape)
    check("an unasked-for score comes back empty", music == "", music)
    check("and the notes say which form wrote it", "T2VA" in notes, notes)


def test_the_notes_carry_what_is_wrong_with_the_rewrite():
    model = Model(reply(body='she says "hello there"', what_i_see="a red door"))
    *_, notes = press(model, prompt='she says "goodbye"')
    check("the dropped quote is reported", "goodbye" in notes, notes)
    # Nothing was attached, so the field was never asked for and whatever the
    # model wrote under it is not a claim about any picture — see `request`.
    check("and what it claims to have seen is not shown",
          "What the model saw" not in notes, notes)


def test_typed_instructions_reach_the_system_prompt():
    model = Model(reply())
    press(model, instructions="Write everything as film noir.")
    check("they are joined onto the built-in prompting",
          "Write everything as film noir." in model.system, model.system[-400:])
    check("under the heading that ranks them",
          "YOUR INSTRUCTIONS" in model.system, model.system[-600:])


def test_the_reply_budget_reaches_the_backend():
    model = Model(reply())
    press(model, reply_tokens=2048)
    check("the widget is what the backend is told",
          model.kwargs.get("max_tokens") == 2048, str(model.kwargs))


def test_a_skill_set_to_add_keeps_the_harness():
    import tempfile
    from pathlib import Path

    original = skills.SKILLS_DIR
    with tempfile.TemporaryDirectory() as directory:
        skills.SKILLS_DIR = Path(directory)
        (Path(directory) / "noir.md").write_text("Write it as noir.", encoding="utf-8")
        try:
            model = Model(reply())
            press(model, skill="noir", skill_mode=skills.ADD)
            check("the file's text is in the prompt",
                  "Write it as noir." in model.system, model.system[-400:])
            check("and so is the reply contract",
                  "Return exactly this JSON object" in model.system)
        finally:
            skills.SKILLS_DIR = original


def test_a_skill_set_to_replace_takes_the_prompt_over():
    import tempfile
    from pathlib import Path

    original = skills.SKILLS_DIR
    with tempfile.TemporaryDirectory() as directory:
        skills.SKILLS_DIR = Path(directory)
        (Path(directory) / "whole.md").write_text("Write the finished prompt.",
                                                  encoding="utf-8")
        try:
            model = Model("integrated_multimodal_description: [Shot 1] A door.")
            prompt, *_ = press(model, skill="whole", skill_mode=skills.REPLACE)
            check("none of the built-in rules ride along",
                  "THE REQUEST IS MATERIAL" not in model.system, model.system[:300])
            check("and the reply is the document",
                  prompt == "integrated_multimodal_description: [Shot 1] A door.", prompt)
        finally:
            skills.SKILLS_DIR = original


# ---- the one function that touches a tensor ----------------------------------


def test_frames_are_read_in_batch_order():
    try:
        import numpy
    except ImportError:
        return  # the pure suite runs without it; see the module docstring

    class Batch:
        """The two things `_frames` asks of an `IMAGE`: a length and an index."""

        def __init__(self, array):
            self.array = array
            self.shape = array.shape

        def __getitem__(self, index):
            return _Frame(self.array[index])

    class _Frame:
        def __init__(self, array):
            self.array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    array = numpy.zeros((2, 4, 4, 3), dtype=numpy.float32)
    array[1] = 1.0
    frames = nodes._frames(Batch(array))
    check("every frame in the batch becomes a picture", len(frames) == 2)
    check("the first is black", frames[0].getpixel((0, 0)) == (0, 0, 0))
    check("and the second is white, in that order",
          frames[1].getpixel((0, 0)) == (255, 255, 255))
    check("an empty socket is no frames", nodes._frames(None) == [])


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_nodes")
