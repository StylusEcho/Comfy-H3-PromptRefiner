"""One press, end to end, against a model that answers from a script.

    python3 tests/test_request.py

`request.refine` takes the backend as two functions, which is what makes this
possible: the whole path — plan the attachments, build the glossary, write the
system prompt, ask, read the reply, check it, assemble the document — runs here
with a `chat` that returns a string from this file and a `look` that hands the
picture straight back. No torch, no ComfyUI, no server.

What is worth holding down here rather than in `test_refine.py` is everything
that only exists once the halves are joined: that the prompt leaving the node
carries ordinals and not handles, that a picture attached but never described
raises a note rather than an error, that the instruction line matches what was
plugged in, and that a skill takes the whole prompt over when it is set to.
"""

import layout

pkg = layout.load("request", "harness", "assets", "contextir", "prompting")
request, refine, assets = pkg.request, pkg.harness, pkg.assets
contextir = pkg.contextir

from harness import FAILURES, check, expect, passed  # noqa: E402


class Model:
    """A backend that says what this test told it to, and records what it was asked."""

    def __init__(self, reply):
        self.reply = reply
        self.system = self.message = None
        self.images = []
        self.kwargs = {}

    def chat(self, model, system, message, images=(), **kwargs):
        self.system, self.message, self.images = system, message, list(images)
        self.kwargs = kwargs
        return self.reply

    @staticmethod
    def look(picture):
        return picture


def reply(body="A red door opens onto rain.", **extra):
    import json

    out = {"shots": [{"body": body}], "overall_soundscape": "Rain on stone.",
           "non_diegetic_music": ""}
    out.update(extra)
    return json.dumps(out)


def run(request_text="a red door in the rain", model=None, **kwargs):
    model = model or Model(reply())
    return model, request.refine(request_text, model.chat, model.look, **kwargs)


# ---- the document that comes out --------------------------------------------


def test_a_bare_request_composes_the_base_form():
    _, out = run(seconds=6.0)
    check("the mode is read off what is attached", out["mode"] == "T2VA", out["mode"])
    check("the body is wrapped in the base field",
          "integrated_multimodal_description: [Shot 1] A red door" in out["prompt"],
          out["prompt"])
    check("the soundscape rides in its own field",
          "overall_soundscape: Rain on stone." in out["prompt"], out["prompt"])
    check("an unasked-for score becomes the guide's N/A",
          f"non_diegetic_music: {contextir.NO_MUSIC}" in out["prompt"], out["prompt"])
    check("and nothing is wrong with it", out["problems"] == [], str(out["problems"]))


def test_a_start_frame_writes_the_alignment_line_first():
    first = assets.image("first_frame", 1, "start.png")
    _, out = run(first_frame=first, pictures={"img-1": object()},
                 seconds=6.0, model=Model(reply(what_i_see="a red door")))
    check("the mode follows the socket", out["mode"] == "I2VA", out["mode"])
    check("the instruction line opens the document",
          out["prompt"].startswith("For the target video, at 0.00 seconds"), out["prompt"])


def test_an_end_frame_names_the_shot_that_arrives_at_it():
    last = assets.image("last_frame", 1, "end.png")
    model = Model(reply(what_i_see="a door"))
    _, out = run(last_frame=last, pictures={"img-1": object()}, seconds=8.0, model=model)
    check("the mode follows the socket", out["mode"] == "L2VA", out["mode"])
    check("the alignment line names the duration and the last shot",
          "(from [Shot 1]) aligns with the 8.00-second mark" in out["prompt"],
          out["prompt"])


def test_the_prompt_that_leaves_carries_ordinals_not_handles():
    ref = assets.image("reference", 1, "anna.png")
    model = Model(reply(body="@img-1 stands in the doorway.", what_i_see="anna",
                        subject_definitions="@img-1 is a woman.", summary="s",
                        retention_analysis="r"))
    _, out = run(references=[ref], pictures={"img-1": object()}, seconds=6.0, model=model)
    check("the mode is the reference form", out["mode"] == "REF2VA", out["mode"])
    check("no handle survives into the prompt", "@img-1" not in out["prompt"],
          out["prompt"])
    check("the label is written instead", "<Picture 1> stands in the doorway"
          in out["prompt"], out["prompt"])
    check("the reference sections are written too",
          "subject_definitions: <Picture 1> is a woman." in out["prompt"], out["prompt"])
    check("and the body takes the reference form's field",
          "detailed_description:" in out["prompt"], out["prompt"])


def test_the_description_output_keeps_its_own_shape():
    _, out = run(seconds=6.0)
    # It carries its shot marker — `join_shots` writes those, and they are what
    # a second shot's cut time would hang off — but none of the field names
    # `compose` puts around it.
    check("the prose alone has no field names",
          out["body"] == "[Shot 1] A red door opens onto rain.", out["body"])
    check("which the document does have",
          "integrated_multimodal_description:" in out["prompt"], out["prompt"])


# ---- what the model is shown ------------------------------------------------


def test_the_pictures_are_numbered_against_the_lines_they_belong_to():
    refs = [assets.image("reference", 1, "a.png"), assets.image("reference", 2, "b.png")]
    model = Model(reply(body="@img-1 and @img-2 meet.", what_i_see="two doors",
                        subject_definitions="d", summary="s", retention_analysis="r"))
    run(references=refs, pictures={"img-1": "A", "img-2": "B"}, seconds=6.0, model=model)
    check("both pictures ride along", model.images == ["A", "B"], str(model.images))
    check("each is marked against its own handle",
          "@img-1 (becomes <Picture 1>) [image 1]" in model.message, model.message)
    check("and the second against the second",
          "@img-2 (becomes <Picture 2>) [image 2]" in model.message, model.message)


def test_a_reference_with_no_picture_says_so():
    ref = assets.image("reference", 1, "sound.wav")
    model = Model(reply(body="@img-1 plays.", subject_definitions="d", summary="s",
                        retention_analysis="r"))
    run(references=[ref], pictures={}, seconds=6.0, model=model)
    check("the glossary line says there is no picture",
          "no picture of it is attached" in model.message, model.message)
    check("and nothing was attached", model.images == [], str(model.images))


def test_the_cap_holds_and_is_reported():
    refs = [assets.image("reference", n, f"{n}.png")
            for n in range(1, request.MAX_IMAGES + 3)]
    pictures = {ref.handle: ref.handle for ref in refs}
    cited = " ".join("@" + ref.handle for ref in refs)
    model = Model(reply(body=cited, what_i_see="many doors", subject_definitions="d",
                        summary="s", retention_analysis="r"))
    _, out = run(references=refs, pictures=pictures, seconds=6.0, model=model)
    check("no more than the cap is attached", len(model.images) == request.MAX_IMAGES,
          str(len(model.images)))
    check("and the ones left out are reported",
          any("not shown to the model" in p for p in out["problems"]),
          str(out["problems"]))


def test_the_model_may_be_asked_to_cut_the_request_itself():
    import json

    model = Model(json.dumps({
        "shots": [{"at_seconds": 0, "body": "The door opens."},
                  {"at_seconds": 4, "body": "Rain falls on the step."}],
        "overall_soundscape": "Rain.", "non_diegetic_music": ""}))
    _, out = run(seconds=9.0, model=model)
    check("the cut rule was asked for", "SHOTS AND CUTS" in model.system)
    check("both shots are in one description",
          "[Shot 1] The door opens. [Shot 2] At 00:04.000, Rain falls" in out["body"],
          out["body"])


def test_cut_shots_off_asks_for_one_body():
    model = Model(reply())
    _, out = run(seconds=9.0, cut_shots=False, model=model)
    check("no cut rule is written", "SHOTS AND CUTS" not in model.system)
    check("and one shot comes back", out["body"] == "A red door opens onto rain.")


# ---- the advisory notes -----------------------------------------------------


def test_a_picture_the_model_never_described_is_reported():
    first = assets.image("first_frame", 1, "start.png")
    # No `what_i_see` in this reply, with a picture attached.
    _, out = run(first_frame=first, pictures={"img-1": object()}, seconds=6.0)
    check("writing past the pictures is reported",
          any("did not say what it saw" in p for p in out["problems"]),
          str(out["problems"]))


def test_what_it_saw_is_dropped_where_nothing_was_attached():
    # The field was never asked for, so anything under it is the model writing
    # the field its worked example had rather than the one it was given.
    _, out = run(seconds=6.0, model=Model(reply(what_i_see="a door I imagined")))
    check("the readout is empty", out["seen"] == "", out["seen"])


def test_a_reference_the_rewrite_never_names_is_reported():
    ref = assets.image("reference", 1, "anna.png")
    model = Model(reply(body="A door opens.", what_i_see="anna",
                        subject_definitions="d", summary="s", retention_analysis="r"))
    _, out = run(references=[ref], pictures={"img-1": object()}, seconds=6.0, model=model)
    check("the uncited reference is named",
          any("never mentions @img-1" in p for p in out["problems"]),
          str(out["problems"]))


def test_a_dropped_quote_is_reported():
    _, out = run('she says "the bridge is out"', seconds=6.0)
    check("the words that did not survive are quoted back",
          any("the bridge is out" in p for p in out["problems"]), str(out["problems"]))


def test_a_pin_across_the_reference_boundary_is_honoured_and_noted():
    model = Model(reply(subject_definitions="d", summary="s", retention_analysis="r"))
    _, out = run(seconds=6.0, template="REF2VA", model=model)
    check("the pin is honoured", out["mode"] == "REF2VA", out["mode"])
    check("what was derived is still reported", out["derived"] == "T2VA", out["derived"])
    check("and the cost is said out loud",
          any("REF2VA template is pinned" in p for p in out["problems"]),
          str(out["problems"]))


# ---- the failures that are refusals -----------------------------------------


def test_an_empty_request_is_refused():
    expect("nothing to refine says so", lambda: run(""), "nothing to refine")


def test_a_reply_that_is_not_the_contract_is_refused():
    expect("prose instead of an object is refused",
           lambda: run(model=Model("I'd be happy to help!")),
           "did not return JSON")


def test_a_rewrite_citing_something_unattached_is_reported_not_composed_wrong():
    model = Model(reply(body="@img-3 opens."))
    # `check` reports it, and then `substitute` refuses to write an ordinal for
    # a picture that is not there: a prompt naming <Picture 3> with two attached
    # is the one failure that produces a wrong video rather than an error.
    expect("a handle with nothing behind it stops the compose",
           lambda: run(seconds=6.0, model=model), "no such asset is attached")


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_request")
