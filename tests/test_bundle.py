"""Reading a `H3_REFS` bundle from the Fantastic MiniMax H3 Prompt Builder.

    python3 tests/test_bundle.py

Nothing is imported from that pack and it need not be installed: ComfyUI matches
sockets by type *name*, so declaring an input of type `H3_REFS` is the whole of
the interoperation. What that buys has to be paid for here instead — the shape
of the bundle and the order inside it are a contract this suite is the only
witness to.

The load-bearing claim is the ordinal walk. The bundle's order is what its
Reference Splitter fans out into H3's own node, so it is what decides the
`<Picture 2>` the *sampler* assigns; `assets.plan` has to walk it the same way or
the prompt cites one tensor and the model is handed another. Nothing errors when
that goes wrong — you get a video of the wrong reference.

H3's presentation order, from `comfy_extras/nodes_minimax_h3.py` and matched by
both packs: pictures, then videos with each paired soundtrack's `<Audio j>`
immediately *before* its `<Video k>`, then standalone audio. This node adds the
keyframes after all of it, which is where upstream Continuity put them too.
"""

import layout

pkg = layout.load("nodes", "assets", "request", "harness", "prompting")
nodes, assets, request = pkg.nodes, pkg.assets, pkg.request
refine, prompting = pkg.harness, pkg.prompting

from harness import FAILURES, check, expect, passed  # noqa: E402
from test_request import Model, reply  # noqa: E402


class Clip(list):
    """The two things a frame batch is asked for here: a length and an index.

    `_still` takes the middle frame, so a clip has to be indexable and have a
    length. Its "frames" are strings, and `_pil` is never reached because
    `nodes._still` is stubbed for these tests — what is under test is which
    frame is chosen and where it lands, not the tensor arithmetic, which
    `test_nodes` covers.
    """


def bundle(pictures=0, videos=0, tracks=(), audios=0, items=None):
    """A bundle shaped like the loader's, with strings standing in for tensors."""
    return {
        # A loaded picture is an IMAGE batch of one, so it reads exactly as a
        # one-frame clip does — which is why `_still` serves both.
        "pictures": [Clip([f"P{n}"]) for n in range(1, pictures + 1)],
        "videos": [Clip([f"V{n}a", f"V{n}b", f"V{n}c"]) for n in range(1, videos + 1)],
        # Index-aligned with `videos`, None where a clip has no paired soundtrack.
        "video_audios": [({"waveform": n} if n in tracks else None)
                         for n in range(1, videos + 1)],
        "audios": [{"waveform": f"A{n}"} for n in range(1, audios + 1)],
        "items": items if items is not None else [],
    }


def stub_still(fn):
    """Run `fn` with `_still` returning the middle element rather than a PIL image."""
    def run(*args):
        original = nodes._still
        nodes._still = lambda clip: (clip[len(clip) // 2] if clip is not None
                                     and len(clip) else None)
        try:
            return fn(*args)
        finally:
            nodes._still = original
    return run


# ---- the walk ----------------------------------------------------------------


@stub_still
def test_the_ordinals_are_h3s_presentation_order():
    refs, videos, audios, _ = nodes._from_bundle(
        bundle(pictures=2, videos=2, tracks=(1,), audios=1))
    _, labels = assets.plan(references=refs, videos=videos, audios=audios)

    check("pictures come first", labels["img-1"] == "<Picture 1>", str(labels))
    check("and are numbered in bundle order", labels["img-2"] == "<Picture 2>", str(labels))
    # Video 1 has a paired soundtrack, so it takes <Audio 1> before <Video 1>.
    check("a paired soundtrack takes the first audio ordinal",
          labels["vid-1:sound"] == "<Audio 1>", str(labels))
    check("videos are numbered among themselves",
          (labels["vid-1"], labels["vid-2"]) == ("<Video 1>", "<Video 2>"), str(labels))
    check("a clip with no soundtrack claims no audio ordinal",
          "vid-2:sound" not in labels, str(labels))
    check("and standalone audio continues the audio count",
          labels["aud-1"] == "<Audio 2>", str(labels))


@stub_still
def test_the_keyframes_trail_the_whole_bundle():
    refs, videos, audios, _ = nodes._from_bundle(bundle(pictures=2, videos=1))
    first = assets.image("first_frame", len(refs) + 1, "start")
    _, labels = assets.plan(first_frame=first, references=refs, videos=videos,
                            audios=audios)
    check("a reference keeps the ordinal it would have had alone",
          labels["img-1"] == "<Picture 1>", str(labels))
    check("and the keyframe takes the one after the pictures",
          labels["img-3"] == "<Picture 3>", str(labels))
    check("which a clip does not consume", labels["vid-1"] == "<Video 1>", str(labels))


@stub_still
def test_a_soundtracks_label_survives_normalisation_and_substitution():
    # It has no handle of its own — nothing points at it separately — so its
    # label is filed under a key the reverse map skips. Writing <Audio 1> for it
    # has to be neither rewritten nor reported.
    refs, videos, audios, _ = nodes._from_bundle(bundle(videos=1, tracks=(1,)))
    _, labels = assets.plan(references=refs, videos=videos, audios=audios)
    handles = {"vid-1"}
    written = "<Video 1> plays, and <Audio 1> is its soundtrack."
    back = refine.normalize_handles(written, labels)
    check("the clip reads back to its handle", "@vid-1 plays" in back, back)
    check("the soundtrack's label is left alone", "<Audio 1>" in back, back)
    check("and nothing is reported wrong",
          refine.check(back, handles, labels) == [],
          str(refine.check(back, handles, labels)))
    check("so it survives to the prompt",
          assets.substitute(back, labels, handles) == written,
          assets.substitute(back, labels, handles))


# ---- what the model is told --------------------------------------------------


@stub_still
def test_a_clip_is_shown_one_frame_from_its_middle():
    _, _, _, pictures = nodes._from_bundle(bundle(videos=1))
    check("the middle frame is the one attached", pictures["vid-1"] == "V1b",
          str(pictures))


@stub_still
def test_an_audio_reference_contributes_nothing_to_look_at():
    _, _, _, pictures = nodes._from_bundle(bundle(audios=2))
    check("no picture rides for a sound", pictures == {}, str(pictures))


@stub_still
def test_the_glossary_says_what_each_kind_is():
    refs, videos, audios, pictures = nodes._from_bundle(
        bundle(pictures=1, videos=1, tracks=(1,), audios=1))
    model = Model(reply(body="@img-1 and @vid-1 and @aud-1.", what_i_see="things",
                        subject_definitions="d", summary="s", retention_analysis="r"))
    request.refine("a scene", model.chat, model.look, references=refs, videos=videos,
                   audios=audios, pictures=pictures, seconds=6.0)

    check("the picture is named and labelled",
          "@img-1 (becomes <Picture 1>) [image 1]" in model.message, model.message)
    check("the clip carries its own soundtrack's label",
          "its soundtrack rides along as <Audio 1>" in model.message, model.message)
    check("the standalone sound says it cannot be heard",
          "you cannot hear it" in model.message, model.message)
    check("and only the two pictures ride along",
          model.images == ["P1", "V1b"], str(model.images))


@stub_still
def test_a_bundle_makes_the_request_a_reference_generation():
    refs, videos, audios, pictures = nodes._from_bundle(bundle(audios=1))
    model = Model(reply(body="@aud-1 plays.", subject_definitions="d", summary="s",
                        retention_analysis="r"))
    _, out = (model, request.refine("a scene", model.chat, model.look, references=refs,
                                    videos=videos, audios=audios, pictures=pictures,
                                    seconds=6.0))
    check("a sound alone is still something cited", out["mode"] == "REF2VA", out["mode"])
    check("so the six-section form is written",
          "detailed_description:" in out["prompt"], out["prompt"])


# ---- the names ---------------------------------------------------------------


@stub_still
def test_filenames_are_read_off_the_item_list():
    items = [{"kind": "picture", "name": "anna.png"},
             {"kind": "video", "name": "walk.mp4", "has_audio": True,
              "audio_mode": "paired"}]
    _, _, _, _ = nodes._from_bundle(bundle(pictures=1, videos=1, tracks=(1,),
                                           items=items))
    names = nodes._names(bundle(pictures=1, videos=1, tracks=(1,), items=items))
    check("the picture's name is found", names["pictures"] == ["anna.png"], str(names))
    check("and the clip's", names["videos"] == ["walk.mp4"], str(names))


@stub_still
def test_an_item_switched_off_is_not_counted():
    items = [{"kind": "picture", "name": "off.png", "enabled": False},
             {"kind": "picture", "name": "on.png"}]
    names = nodes._names(bundle(pictures=1, items=items))
    check("the numbering closes up around it", names["pictures"] == ["on.png"],
          str(names))


@stub_still
def test_names_that_do_not_line_up_are_dropped_rather_than_guessed():
    # A name against the wrong file is worse than no name, so a disagreement
    # between the item list and the tensors falls back to positional names.
    items = [{"kind": "picture", "name": "one.png"}]
    refs, _, _, _ = nodes._from_bundle(bundle(pictures=2, items=items))
    check("the fallback names by position",
          [a.filename for a in refs] == ["picture 1", "picture 2"],
          str([a.filename for a in refs]))


@stub_still
def test_a_bundle_with_no_items_still_reads():
    refs, videos, audios, _ = nodes._from_bundle(bundle(pictures=1, videos=1, audios=1))
    check("everything arrives", (len(refs), len(videos), len(audios)) == (1, 1, 1))
    check("named by what it is", refs[0].filename == "picture 1", refs[0].filename)


def test_something_that_is_not_a_bundle_is_refused_by_name():
    expect("a wrong wire says so",
           lambda: nodes._from_bundle(["not", "a", "bundle"]),
           "did not receive a MiniMax H3 reference bundle")


# ---- the two reference inputs together ---------------------------------------


@stub_still
def test_the_bundles_pictures_lead_and_the_loose_socket_follows():
    # The Prompt Builder wrote its prose against the bundle's own numbering, so
    # a loose picture cutting in front of it would renumber every citation.
    class Batch(list):
        shape = (1,)

    original = nodes._frames
    nodes._frames = lambda image: (["LOOSE"] if image is not None else [])
    try:
        run = nodes.H3PromptRefiner().run
        model = Model(reply(body="@img-1 and @img-2.", what_i_see="two",
                            subject_definitions="d", summary="s",
                            retention_analysis="r"))
        backend = nodes._backend
        nodes._backend = lambda refiner: (model.chat, model.look)
        try:
            run({"where": "local", "model": "q"}, prompt="a scene", seconds=6.0,
                template="auto", cut_shots=True, language="English", temperature=0.3,
                seed=0, reply_tokens=refine.NUM_PREDICT,
                references=Batch(["x"]), reference_bundle=bundle(pictures=1))
        finally:
            nodes._backend = backend
    finally:
        nodes._frames = original

    check("the bundle's picture is <Picture 1>",
          "@img-1 (becomes <Picture 1>)" in model.message, model.message)
    check("and the loose one follows it",
          "@img-2 (becomes <Picture 2>)" in model.message, model.message)
    check("in that order on the wire", model.images == ["P1", "LOOSE"],
          str(model.images))


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_bundle")
