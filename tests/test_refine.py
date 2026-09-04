"""What the refiner promises about the model's reply.

    python3 tests/test_refine.py

Runs standalone, with no torch and no ComfyUI. Everything here is the boundary
between a language model's output and a prompt that will be encoded, which is
the part worth pinning down: the model is the one component of this pack that
cannot be relied on to do what it was asked, so what happens when it does not is
the contract.

The load-bearing one is `normalize_handles` followed by `assets.substitute`. The
model is asked for `@img-1` and shown the label it will be given, and it reaches
for the label anyway, because the guide it has just read is written entirely in
ordinals. Converting back and then forward again is what keeps one
representation in the middle. Getting it wrong is silent: the prompt still
composes, and points at the wrong picture.
"""

import layout

pkg = layout.load("harness", "prompting", "assets", "contextir")
refine, prompting = pkg.harness, pkg.prompting
assets, contextir = pkg.assets, pkg.contextir

from harness import FAILURES, check, expect, passed  # noqa: E402


# ---- handles and labels -----------------------------------------------------

LABELS = {"img-1": "<Picture 1>", "img-2": "<Picture 2>", "vid-1": "<Video 1>"}


def test_normalize_reads_labels_back():
    check("a written label becomes its handle",
          refine.normalize_handles("<Picture 2> is the start", LABELS)
          == "@img-2 is the start")
    check("spacing inside a label is tolerated",
          refine.normalize_handles("<  Picture   1 >", LABELS) == "@img-1")
    check("a video label reads back too",
          refine.normalize_handles("<Video 1> continues", LABELS) == "@vid-1 continues")


def test_normalize_leaves_a_label_with_nothing_behind_it():
    # Silently deleting it would hide the one failure that produces a wrong
    # video rather than an error. `check` is what reports it.
    check("a label no asset will be given is left as written",
          refine.normalize_handles("<Picture 9> matters", LABELS)
          == "<Picture 9> matters")


def test_subject_labels_are_left_alone_without_a_cast():
    # `<Subject N>` in a reference rewrite is the model's own invention, defined
    # inside its own sections and pointing at nothing outside the rewrite.
    check("a subject label survives normalisation",
          refine.normalize_handles("<Subject 1> walks", LABELS) == "<Subject 1> walks")


def test_substitute_writes_the_ordinal_back():
    handles = set(LABELS)
    check("a handle becomes its label",
          assets.substitute("@img-2 opens", LABELS, handles) == "<Picture 2> opens")
    check("prose that only looks like a handle survives",
          assets.substitute("meet me @ 5", LABELS, handles) == "meet me @ 5")
    expect("a handle with nothing attached is refused",
           lambda: assets.substitute("@img-9 opens", LABELS, handles),
           "no such asset is attached")


def test_the_round_trip_is_the_identity():
    written = "<Picture 1> and @img-2 and <Video 1>"
    back = refine.normalize_handles(written, LABELS)
    check("everything is handles in the middle", back == "@img-1 and @img-2 and @vid-1")
    check("and ordinals again at the end",
          assets.substitute(back, LABELS, set(LABELS))
          == "<Picture 1> and <Picture 2> and <Video 1>")


# ---- the advisory checks ----------------------------------------------------


def test_check_reports_a_citation_pointing_at_nothing():
    problems = refine.check("@img-4 is on the left", set(LABELS), LABELS)
    check("an unattached handle is reported", len(problems) == 1, str(problems))
    check("and names itself", "@img-4" in problems[0], str(problems))


def test_check_reports_a_stray_label():
    problems = refine.check("<Picture 7> is behind", set(LABELS), LABELS)
    check("a label no asset takes is reported", len(problems) == 1, str(problems))


def test_check_passes_a_clean_rewrite():
    check("nothing wrong is nothing said",
          refine.check("@img-1 opens on <Picture 2>", set(LABELS), LABELS) == [])


def test_uncited_finds_a_reference_nothing_points_at():
    refs = {"img-1", "img-2"}
    check("a reference never named comes back",
          refine.uncited("@img-1 alone", refs, LABELS) == ["img-2"])
    check("a reference named by its label counts as cited",
          refine.uncited("@img-1 and <Picture 2>", refs, LABELS) == [])


def test_dropped_quotes_holds_the_one_mechanical_promise():
    request = 'she says "the bridge is out"'
    check("a quoted span that vanished is reported",
          refine.dropped_quotes([request], "she warns him about the bridge")
          == ["the bridge is out"])
    check("casing and spacing are forgiven",
          refine.dropped_quotes([request], 'She says, "The  bridge is out."') == [])


# ---- cutting one request into shots -----------------------------------------


def test_shot_limit_is_the_duration_over_the_floor():
    check("a six second clip is three shots at most", refine.shot_limit(6) == 3)
    check("under two shots' worth there is no choice to offer",
          refine.shot_limit(3) == 1)
    check("and the ceiling holds", refine.shot_limit(600) == refine.MAX_SHOTS)


def test_plan_cuts_makes_the_times_monotonic_and_fit():
    planned = refine.plan_cuts(["a", "b", "c"], [9, 2, 4], 12.0)
    times = [at for at, _ in planned]
    check("the first shot starts at zero", times[0] == 0.0, str(times))
    check("every later cut clears the one before it",
          all(b - a >= refine.MIN_SHOT_S for a, b in zip(times, times[1:])), str(times))
    check("and leaves a shot's worth of video after it",
          times[-1] <= 12.0 - refine.MIN_SHOT_S, str(times))


def test_plan_cuts_merges_a_shot_with_no_room_left():
    # Its prose is the only copy of that part of the description, so it is
    # merged into the shot before it rather than dropped.
    planned = refine.plan_cuts(["a", "b", "c"], [0, 2, 4], 3.0)
    check("a shot that cannot fit joins the one before it", len(planned) == 1,
          str(planned))
    check("and its prose survives", planned[0][1] == "a b c", str(planned))


# ---- the reply --------------------------------------------------------------


def test_json_object_forgives_transport_noise():
    check("a fenced object is read",
          refine.json_object('```json\n{"a": 1}\n```') == {"a": 1})
    check("a leaked think block is read past",
          refine.json_object('<think>hmm</think>{"a": 1}') == {"a": 1})
    check("a sentence in front of the object is read past",
          refine.json_object('Here you go: {"a": 1}') == {"a": 1})


def test_json_object_refuses_what_is_not_an_object():
    expect("prose with no object in it is refused",
           lambda: refine.json_object("I cannot do that"),
           "did not return JSON")
    expect("an array is refused",
           lambda: refine.json_object("[1, 2]"), "did not return JSON")


REPLY = """\
{"what_i_see": "a red door",
 "shots": [{"body": "A red door opens onto rain."}],
 "overall_soundscape": "Rain on stone.",
 "non_diegetic_music": ""}"""


def test_parse_reply_reads_the_contract():
    parsed = prompting.parse_reply(REPLY, "I2VA")
    check("the body comes back", parsed["shots"] == ["A red door opens onto rain."])
    check("the soundscape comes back", parsed["soundscape"] == "Rain on stone.")
    check("an empty score stays empty", parsed["music"] == "")
    check("and what it saw comes back", parsed["seen"] == "a red door")


def test_parse_reply_refuses_a_shot_count_nobody_asked_for():
    two = '{"shots": [{"body": "a"}, {"body": "b"}]}'
    expect("two bodies where one was asked for is refused",
           lambda: prompting.parse_reply(two, "T2VA"), "asked for one shot")
    check("...and is fine where the model was choosing",
          len(prompting.parse_reply(two, "T2VA", cuts=3)["shots"]) == 2)


def test_parse_reply_carries_the_reference_sections():
    reply = ('{"subject_definitions": "d", "summary": "s", "retention_analysis": "r",'
             ' "shots": [{"body": "b"}]}')
    parsed = prompting.parse_reply(reply, "REF2VA")
    check("all three sections come back",
          parsed["sections"] == {"subject_definitions": "d", "summary": "s",
                                 "retention_analysis": "r"}, str(parsed))
    check("and a base mode asks for none",
          "sections" not in prompting.parse_reply(reply, "T2VA"))


def test_join_shots_writes_the_markers_and_the_times():
    body = prompting.join_shots(["first", "second"], [0, 4], 8.0)
    check("shot one carries no timestamp", body.startswith("[Shot 1] first"), body)
    check("shot two opens on its cut time", "[Shot 2] At 00:04.000, second" in body, body)


def test_join_shots_takes_back_markers_the_model_wrote_itself():
    body = prompting.join_shots(["[Shot 1] first", "[Shot 2] At 00:04.000, second"],
                                [0, 4], 8.0)
    check("a marker the model wrote is not doubled",
          body.count("[Shot 1]") == 1 and body.count("[Shot 2]") == 1, body)
    check("nor is a cut time", body.count("At 00:04.000") == 1, body)


# ---- what is attached, and what it is called --------------------------------


def test_derive_mode_reads_the_shape():
    first = assets.image("first_frame", 1)
    last = assets.image("last_frame", 2)
    ref = assets.image("reference", 1)
    check("nothing attached is T2VA", assets.derive_mode(None, None, []) == "T2VA")
    check("a start frame is I2VA", assets.derive_mode(first, None, []) == "I2VA")
    check("an end frame is L2VA", assets.derive_mode(None, last, []) == "L2VA")
    check("both is FL2VA", assets.derive_mode(first, last, []) == "FL2VA")
    check("a reference is REF2VA", assets.derive_mode(None, None, [ref]) == "REF2VA")
    check("...whatever else is attached alongside it",
          assets.derive_mode(first, last, [ref]) == "REF2VA")


def test_references_keep_their_ordinals_when_a_frame_rides_along():
    refs = [assets.image("reference", 1), assets.image("reference", 2)]
    first = assets.image("first_frame", 3)
    alone, labels_alone = assets.plan(references=refs)
    both, labels_both = assets.plan(first_frame=first, references=refs)
    check("a reference's label does not move when a frame is added",
          labels_alone["img-1"] == labels_both["img-1"] == "<Picture 1>",
          str(labels_both))
    check("and the frame takes the ordinal after them",
          labels_both["img-3"] == "<Picture 3>", str(labels_both))
    check("the frames are presented last",
          [a.handle for a in both] == ["img-1", "img-2", "img-3"], str(both))


def test_a_lone_end_frame_is_picture_one():
    last = assets.image("last_frame", 1)
    _, labels = assets.plan(last_frame=last)
    check("with no references the end frame opens the presentation",
          labels["img-1"] == "<Picture 1>", str(labels))


def test_choose_template_honours_a_pin():
    check("auto follows what is attached",
          prompting.choose_template("auto", "I2VA") == ("I2VA", False))
    check("a pin replaces it, and says it was forced",
          prompting.choose_template("REF2VA", "I2VA") == ("REF2VA", True))
    expect("a template that does not exist is refused",
           lambda: prompting.choose_template("T2V", "T2VA"), "unknown refine template")


def test_pin_note_fires_only_across_the_reference_boundary():
    check("two base templates need no note",
          prompting.pin_note("I2VA", "T2VA") is None)
    check("crossing into the reference form does",
          "REF2VA template is pinned" in prompting.pin_note("REF2VA", "T2VA"))
    check("and crossing out of it does",
          "T2VA template is pinned" in prompting.pin_note("T2VA", "REF2VA"))


# ---- the request that goes out ----------------------------------------------


def test_the_system_prompt_carries_the_mode_and_its_template():
    system = prompting.system_prompt("I2VA", shape="SHAPE", cuts=3)
    check("the mode is stated", "This request is I2VA." in system)
    check("its template is in there", prompting.MODE_TEMPLATE["I2VA"] in system)
    check("the craft notes are in there", prompting.CRAFT in system)
    check("the cut rule is there where cuts were asked for", "SHOTS AND CUTS" in system)
    check("and the contract is last", system.rstrip().endswith("SHAPE"), system[-80:])


def test_the_cut_rule_is_left_out_where_there_is_nothing_to_divide():
    check("one shot means no cut rule",
          "SHOTS AND CUTS" not in prompting.system_prompt("T2VA", cuts=1))


def test_user_instructions_outrank_the_craft_and_not_the_contract():
    system = prompting.system_prompt("T2VA", shape="SHAPE", extra="write it in haiku")
    check("the instructions are in there", "write it in haiku" in system)
    check("they are ranked above the craft",
          system.index("YOUR INSTRUCTIONS") > system.index(prompting.CRAFT))
    check("and below the reply contract",
          system.index("YOUR INSTRUCTIONS") < system.index("SHAPE"))


def test_the_reply_shape_asks_what_it_saw_only_with_a_picture_attached():
    check("with a picture, the field is asked for and named",
          refine.SEEN_FIELD in prompting.reply_shape("I2VA", shown=("img-1",)))
    check("with none, it is not",
          refine.SEEN_FIELD not in prompting.reply_shape("T2VA"))
    check("the handles are written into the instruction",
          "@img-1" in prompting.reply_shape("I2VA", shown=("img-1",)))


def test_the_reply_shape_asks_for_the_sections_only_in_the_reference_form():
    check("REF2VA asks for all three",
          all(name in prompting.reply_shape("REF2VA") for name in
              ("subject_definitions", "summary", "retention_analysis")))
    check("a base mode asks for none",
          "subject_definitions" not in prompting.reply_shape("I2VA"))


def test_the_user_message_fences_the_request():
    message = prompting.user_message("a dog on a beach", seconds=6)
    check("the request is behind a fence",
          "<request>\na dog on a beach\n</request>" in message, message)
    check("and the duration is stated", "6.00 seconds" in message, message)


def test_the_user_message_names_the_pictures_it_is_attaching():
    slots = [prompting.slot_row(assets.image("reference", 1, "anna.png"), "<Picture 1>")]
    slots[0]["image"] = 1
    message = prompting.user_message("a portrait", shown=("img-1",), slots=slots)
    check("the one picture is named by its handle", "@img-1" in message, message)
    check("the glossary line carries the filename", "anna.png" in message, message)
    check("and the label it will be given", "<Picture 1>" in message, message)
    check("with the instruction to look at it", "[image 1]" in message, message)


def test_an_empty_request_is_refused_before_a_model_is_loaded():
    expect("nothing to refine says so",
           lambda: prompting.user_message("   "), "nothing to refine")


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_refine")
