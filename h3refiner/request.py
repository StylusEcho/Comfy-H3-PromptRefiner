"""One press of the refiner: look at what is attached, ask, read the reply back.

Upstream this was `creator/refine_routes.py` — an aiohttp endpoint that took the
Creator node's whole state blob, compiled it to find out what the request was,
and handed prose back for the frontend to write into the same blob. It was a
server round trip rather than a step inside `execute` because the rewrite had to
be visible and editable before a five-minute sampler pass, because it had to be
stored or the same queue would produce different prompts on consecutive runs,
and because the media it looks at was addressed by filename in the input folder.

None of those three hold for a node. The rewrite is an output socket, so it is
visible wherever the user cares to route it; ComfyUI's cache is what makes two
runs of the same graph produce the same prose; and the pictures arrive as
tensors on an input, so nothing needs to open a file. So the round trip is gone
and this is an ordinary function, called from `nodes.py` inside `execute`, with
no aiohttp and no ComfyUI in it.

What survived unchanged is the part that was never about transport: which
template writes the rewrite, how many pictures one call may look at, binding
each picture to the glossary line it belongs to, and the four advisory checks
run over the prose that comes back — a citation pointing at nothing, a label no
asset will be given, an attached reference the rewrite never mentions, and a
span the request put in quotation marks that did not survive. Those are reported
rather than raised: `notes` comes out of the node beside the prompt, which is a
better place to resolve them than a refusal on prose that is one word away from
being right.
"""

from . import assets, contextir, harness, prompting, skills

# What one call will look at. Every image rides in the context window for the
# whole generation, and each of a few thousand output tokens attends over all of
# them again. Upstream's number, for upstream's reason: a request with a dozen
# references would otherwise fill the window with pictures and leave no room for
# the guide.
MAX_IMAGES = 16

# Rewriting is a fidelity task, not an ideation one: the default leans cold so
# that named things survive.
TEMPERATURE = 0.3


def _glossary(ordered, labels, pictures, limit=MAX_IMAGES):
    """The glossary lines, and the pictures that actually ride with them.

    Binding each picture to the line it belongs to is explicit rather than
    positional — `_number` upstream, and for the same reason. Several attached
    things have no picture of their own: an audio reference, a clip taken for
    its soundtrack alone, an image socket that was left empty. Counting the
    pictures and stamping the number onto the slot that produced each one is
    what stops an audio clip in the middle shifting every later picture onto the
    wrong handle.

    The cap is applied here rather than afterwards, so a slot past it says it
    was not shown rather than pointing at an image no longer in the message.

    -> `(slots, [picture], (handle per picture,), how many were dropped)`. The
    handles come back because the instruction that asks the model to describe
    the pictures names them: told only how many there are, a model reaching for
    a handle takes the one its worked example used, and every example is written
    in `@img-N`.
    """
    slots, kept, shown, dropped = [], [], [], 0
    for asset in ordered:
        slot = prompting.slot_row(asset, labels.get(asset.handle))
        picture = pictures.get(asset.handle)
        if picture is None:
            if asset.kind != "audio" and asset.track != "sound":
                # It should have had one and does not. Said rather than left
                # silent, because the glossary line stays either way and a handle
                # the model believes it can see is worse than one it knows it
                # cannot.
                slot["note"] = "no picture of it is attached"
        elif len(kept) >= limit:
            dropped += 1
            slot["note"] = (f"not shown to the model — one call looks at at most "
                            f"{limit} images")
        else:
            kept.append(picture)
            shown.append(asset.handle)
            slot["image"] = len(kept)
        slots.append(slot)
    return slots, kept, tuple(shown), dropped


def _dropped_note(dropped):
    if not dropped:
        return []
    return [f"{dropped} attached file{'s were' if dropped != 1 else ' was'} not shown "
            f"to the model — one call looks at at most {MAX_IMAGES}"]


def _out(mode, derived, body, soundscape, music, seen, problems, seconds, labels,
         sections=None, skill=None):
    """The finished result, with the document assembled around the prose.

    `contextir.compose` is what turns the fields into the sectioned document H3
    was trained to read, and it is handed the shot count off the body's own
    markers: `instruction`'s `(from [Shot N])` is about where the end frame
    lands, so a rewrite the model cut into three shots has to say the third.
    """
    prompt = contextir.compose(
        mode, body, soundscape=soundscape, music=music, seconds=seconds,
        shots=max(1, contextir.count_shots(body)), sections=sections)
    return {
        "prompt": prompt,
        "body": body,
        "soundscape": soundscape,
        "music": music,
        "seen": seen,
        # Which template actually wrote this, and what the attachments said it
        # was. Both, because "which form is this prose in" should be readable off
        # the result rather than deduced from what was plugged in.
        "mode": mode,
        "derived": derived,
        "skill": skill,
        "labels": dict(labels),
        "problems": problems,
    }


def _run_skill(skill, chat, look, model, request, labels, handles, refs, slots,
               pictures, dropped, seconds, mode, language, temperature, seed,
               max_tokens):
    """The replace path: the loaded skill is the whole instruction.

    Nothing of the harness rides along — no rules, no guide, no JSON contract, no
    prefill — and nothing is assembled around what comes back either. The reply
    *is* the document: instruction line, field names, shot markers and
    timestamps are the skill's to write, which is the bet this mode exists to
    take. So `compose` is not called here, and a skill that leaves a field out
    has left it out rather than had `N/A` written in for it.

    The one thing that still happens to the text is the label round trip, which
    is not the harness talking: a skill's guide is written in `<Picture N>` and
    the model reaches for it, so it is read back to handles, checked against
    what is attached, and written forward again — the same one representation
    every other path keeps.
    """
    shot = {"text": request, "slots": slots}
    content = chat(
        model,
        skills.system_prompt(skill),
        skills.user_message(shot, seconds=seconds, images=len(pictures), mode=mode,
                            language=language),
        [look(picture) for picture in pictures],
        temperature=temperature, seed=seed, max_tokens=max_tokens,
        prefill="",
    )
    written = harness.normalize_handles(skills.parse_reply(content), labels)

    problems = _dropped_note(dropped)
    problems += ["The rewrite " + p for p in harness.check(written, handles, labels)]
    for handle in harness.uncited(written, refs, labels):
        problems.append(
            f"the rewrite never mentions @{handle} — the file is still attached, "
            f"but nothing in the prompt will point at it. Refine again, or write "
            f"it in yourself."
        )
    for span in harness.dropped_quotes([request], written):
        problems.append(
            f'the request quotes "{span}" and the rewrite never writes it — those '
            f'exact words will not reach the video model. Refine again, or edit '
            f'them in.'
        )

    # The document carries its own audio sections, so the two fields stay empty
    # rather than duplicating them outside it, and there is no `seen` readout —
    # the skill's contract has no such field, and inventing one would be the
    # harness leaking back in.
    document = assets.substitute(written, labels, handles)
    return {"prompt": document, "body": document, "soundscape": "", "music": "",
            "seen": "", "mode": mode, "derived": mode, "skill": skill["name"],
            "labels": dict(labels), "problems": problems}


def refine(request, chat, look, model="", first_frame=None, last_frame=None,
           references=(), pictures=None, seconds=0.0, template="auto",
           language="English", temperature=TEMPERATURE, seed=-1, max_tokens=None,
           extra="", skill=None, cut_shots=True):
    """One rewrite. -> the result dict `_out` describes.

    `chat` and `look` are the backend: one generation, and whatever that backend
    wants a picture as — a tensor in-process, a `data:` URL on the wire. They are
    passed in rather than chosen here so that this module needs neither torch nor
    a network, and so the tests can drive the whole path against a canned reply.

    `pictures` is `handle -> PIL image`, for the assets that have one.

    `cut_shots` is whether the model may divide the request into several shots
    for itself. There is nothing else to divide one request — no cards, no
    durations anybody set — so without it the answer is a single uncut shot
    however long the clip runs. `harness.shot_limit` is what turns the duration
    into a ceiling, and a clip too short for two shots is not asked whatever this
    says.
    """
    request = str(request or "").strip()
    if not request:
        raise harness.RefineError("there is nothing to refine — write a prompt first")

    pictures = dict(pictures or {})
    ordered, labels = assets.plan(first_frame, last_frame, references)
    handles = {asset.handle for asset in ordered}
    refs = {asset.handle for asset in ordered if asset.role == "reference"}

    derived = assets.derive_mode(first_frame, last_frame, references)
    mode, forced = prompting.choose_template(template, derived)

    slots, pictures, shown, dropped = _glossary(ordered, labels, pictures)

    if skill:
        return _run_skill(skill, chat, look, model, request, labels, handles, refs,
                          slots, pictures, dropped, seconds, mode, language,
                          temperature, seed, max_tokens)

    cuts = harness.shot_limit(seconds) if cut_shots else 0

    # ComfyUI's generation loop samples plain logits — nothing constrains the
    # reply to a shape — so the shape is written into the instruction as words
    # and the reply is started mid-object.
    shape = prompting.reply_shape(mode, cuts=cuts, shown=shown)
    system = prompting.system_prompt(mode, language or "English", shape=shape,
                                     cuts=cuts, extra=extra)
    message = prompting.user_message(request, seconds=seconds, shown=shown,
                                     slots=slots, mode=mode)
    content = chat(model, system, message, [look(picture) for picture in pictures],
                   temperature=temperature, seed=seed, max_tokens=max_tokens)
    parsed = prompting.parse_reply(content, mode, cuts=cuts)

    # Several shots came back where one description was asked about, which is the
    # whole point of `cuts` — they are one description with cuts in it, and
    # `join_shots` is what writes the markers and the times around them.
    if "cuts" in parsed:
        body = prompting.join_shots(parsed["shots"], parsed["cuts"], seconds)
    else:
        body = parsed["shots"][0]

    problems = []
    # A pin is honoured, never refused — but where the pinned form and the
    # attachments stop describing each other, that costs something worth saying.
    note = prompting.pin_note(mode, derived) if forced else None
    if note:
        problems.append(note)
    problems += _dropped_note(dropped)
    # Asked for whenever a picture rides along, so its absence is a model that
    # wrote the rewrite without ever attending to the images — which is exactly
    # the failure the field was added to make visible.
    if pictures and not parsed.get("seen"):
        problems.append(
            "the model did not say what it saw in the attached images, so it may "
            "have written past them — check the rewrite against your frames"
        )
    # ...and with no picture attached the field was never asked for, so anything
    # under it is the model writing the field its worked example had rather than
    # the one it was given.
    seen = parsed["seen"] if pictures else ""

    # Back to handles first: the model has just read a guide written entirely in
    # ordinals, so it reaches for them however it is asked. Normalising means
    # `check` and `uncited` see one representation, and `assets.substitute` is
    # the single place an ordinal is written.
    fields = {"body": body,
              "soundscape": parsed["soundscape"],
              "music": parsed["music"]}
    sections = {name: text for name, text in (parsed.get("sections") or {}).items()
                if text}
    fields.update(sections)
    fields = {name: harness.normalize_handles(text, labels)
              for name, text in fields.items()}

    for name, text in fields.items():
        where = "the rewrite" if name == "body" else f"the rewrite's {name}"
        for problem in harness.check(text, handles, labels):
            problems.append(f"{where} {problem}")

    # Every field joined, because a reference legitimately lives in only one of
    # them: H3's reference form defines an image inside `subject_definitions`,
    # folds it into a `<Subject N>`, and never names it again.
    everything = "\n".join(fields.values())
    for handle in harness.uncited(everything, refs, labels):
        problems.append(
            f"the rewrite never mentions @{handle} — the file is still attached, "
            f"but nothing in the prompt will point at it. Refine again, or write "
            f"it in yourself."
        )
    # The one fidelity promise that can be checked mechanically rather than
    # trusted to the system prompt: quotation marks in a request are the user
    # dictating exact words, and a quoted span either survives or it does not.
    for span in harness.dropped_quotes([request], everything):
        problems.append(
            f'the request quotes "{span}" and the rewrite never writes it — those '
            f'exact words will not reach the video model. Refine again, or edit '
            f'them in.'
        )

    written = {name: assets.substitute(text, labels, handles,
                                       "the rewrite" if name == "body"
                                       else f"the rewrite's {name}")
               for name, text in fields.items()}
    return _out(mode, derived, written.pop("body"), written.pop("soundscape"),
                written.pop("music"), seen, problems, seconds, labels,
                sections={name: written[name] for name in sections})
