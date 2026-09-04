"""The document H3 was trained to read, assembled around the model's prose.

Lifted from ComfyUI-Continuity's `creator/families/h3/contextir.py`, trimmed to
the half a prompt refiner needs. H3 is two models: the hosted half rewrites what
the user typed into a labelled, sectioned intermediate representation, and the
open weights were only ever trained on that output. This module puts the
*skeleton* back — the field names, the instruction line, the `[Shot N]` markers,
the written form of every cut time — because all of that is mechanical.
`prompting.py` is what asks a vision LLM for the prose that goes inside it.

What did not come across is everything that was about the Creator node's own
bookkeeping: the reference and retention line builders that walked a compiled
asset plan, the task-type derivation, the cast's subject sections. Those
described a timeline being compiled to a payload, and this node compiles
nothing — it emits text. `compose` still accepts the six-section form under
`sections=`, because the refiner writes those itself in REF2VA.
"""

import re


# The three base-mode fields, in the order the guide emits them.
BODY_FIELD = "integrated_multimodal_description"


SOUNDSCAPE_FIELD = "overall_soundscape"


MUSIC_FIELD = "non_diegetic_music"


# Ref2VA is a different, six-section form, and its body field is
# `detailed_description`. Upstream derived its other three sections from the
# Creator node's chips and cast so that a reference prompt was complete whether
# or not anybody ran a refiner; here they come from the refiner alone, which is
# the arrangement `REF_SECTIONS` was written for in the first place.
REF_BODY_FIELD = "detailed_description"


BODY_FIELDS = (BODY_FIELD, REF_BODY_FIELD)


# The three sections the reference form has that the base form does not, in the
# order the guide emits them. The refiner writes all three when it is asked for
# a REF2VA rewrite; `compose` emits the reference form when it is handed any of
# them and the base form when it is not.
REF_SECTIONS = ("subject_definitions", "summary", "retention_analysis")


# The modes whose body belongs in `integrated_multimodal_description`.
BASE_MODES = ("T2VA", "I2VA", "L2VA", "FL2VA")


# `[Shot 1]`, `[Shot 12]` — the marker the description is segmented by.
SHOT_RE = re.compile(r"\[Shot\s+\d+\]")


# The guide's value for "there is deliberately none of this". A blank
# `non_diegetic_music` used to emit nothing at all, which reads to the model as
# a free hand rather than as a decision; `N/A` is the decision written down, and
# is the single most-cited community fix for a soundtrack nobody asked for.
NO_MUSIC = "N/A"


def count_shots(body):
    """How many shots a description holds — what `instruction`'s `Shot N` is."""
    return len(SHOT_RE.findall(body or ""))


# `[Shot 3]` with its number, for `appears_in`.
_SHOT_NUMBER_RE = re.compile(r"\[Shot\s+(\d+)\]")


def appears_in(label, body):
    """`"[Shot 1], [Shot 3]"` — where `label` is written, or "" if nowhere.

    Derived from the finished description rather than declared, because it is
    derivable: the shots are numbered in the text and the label is in it or it
    is not. A body with no shot markers at all is one shot, and a generation is
    one shot unless it says otherwise — so the common case answers `[Shot 1]`
    without anyone having written a marker.

    Kept because the shot marker is this module's to know, and anything that
    wants to say where a label was written should be reading it off the finished
    description rather than tracking it alongside.
    """
    if label not in (body or ""):
        return ""
    shots = []
    current = 1
    for piece in re.split(r"(\[Shot\s+\d+\])", body):
        match = _SHOT_NUMBER_RE.fullmatch(piece)
        if match:
            current = int(match.group(1))
        elif label in piece and current not in shots:
            shots.append(current)
    return ", ".join(f"[Shot {n}]" for n in sorted(shots))


def has_field(text, name):
    """Whether `text` already carries a `name:` section, at the start of a line."""
    return re.search(rf"^[ \t]*{re.escape(name)}[ \t]*:", text or "", re.MULTILINE) is not None


def _has_instruction(text):
    """Whether `text` already opens with a keyframe-alignment instruction.

    Matched on the two documented openings rather than on a field name, because
    the instruction is a bare sentence with no `name:` marker to look for.
    """
    head = (text or "").lstrip()
    return head.startswith("For the target video,") or head.startswith("How the reference pictures align")


def shot_time(seconds):
    """`3.5` -> `"00:03.500"`, the cut-time format the guide writes.

    Section 4.2: every shot after the first opens with a strictly increasing cut
    time. This is the only place that format is spelled, so a change here moves
    every cut in a one-pass render.
    """
    total_ms = int(round(float(seconds) * 1000))
    minutes, rest = divmod(total_ms, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


# `At 00:03.500,` at the head of a shot — already written by hand, so not added.
CUT_TIME_RE = re.compile(r"^\s*At\s+\d{1,3}:\d{2}\.\d{3}\s*,")


def shot_body(shots):
    """`[(at_seconds, text), ...]` -> one `[Shot n]`-marked description.

    The guide's section 4.2 in one function: shot 1 carries no timestamp, every
    later shot opens with its cut time, and the prose after the comma is the
    user's own — including which of `the camera cuts to` / `the shot transitions
    to` they wanted. Inventing a transition verb here would be writing a line of
    their description for them, and the guide lists five to choose between.

    A card that already carries its own markers is passed through verbatim and
    counts for as many shots as it numbers, so writing two shots into one card
    does not knock the rest of the timeline out of step. Its numbers are checked
    against the position it actually occupies and refused if they disagree —
    refusing is not rewriting, and the alternative is a description with two
    `[Shot 2]`s in it that nothing would have complained about.
    """
    out = []
    number = 1
    for position, (at, text) in enumerate(shots, start=1):
        text = (text or "").strip()
        if not text:
            raise ValueError(
                f"shot {position} has no prompt — the shots of one pass are a "
                f"single description with cuts in it, so an empty one would leave "
                f"a cut with nothing on the far side of it"
            )

        own = [re.sub(r"\s+", " ", m) for m in SHOT_RE.findall(text)]
        if own:
            want = [f"[Shot {n}]" for n in range(number, number + len(own))]
            if own != want:
                raise ValueError(
                    f"shot {position} numbers its own shots {' '.join(own)}, but in this "
                    f"timeline it is {' '.join(want)} — renumber it, or drop the markers "
                    f"and let the timeline number the shots"
                )
            out.append(text)
            number += len(own)
            continue

        head = f"[Shot {number}]"
        if number > 1 and not CUT_TIME_RE.match(text):
            head += f" At {shot_time(at)},"
        out.append(f"{head} {text}")
        number += 1
    return " ".join(out)


def instruction(mode, seconds, shots=1):
    """The first line of the prompt for a keyframe mode, or None.

    Quoted from the official guide rather than paraphrased — including FL2VA's
    unbracketed `Picture 1`, which differs from the other two lines and is not a
    typo on this end. `S.SS` is the effective duration to exactly two decimals,
    so it must be the real frame-count-derived duration of the clip you are about
    to sample — set the node's `seconds` to that, not to a rounded figure.

    `shots` is how many shots the description holds. The end frame is reached by
    the *last* one — the guide writes `(from Shot N)` — which only differs from
    `Shot 1` in a one-pass render of several shots. The start frame is always
    Shot 1's, whatever follows it.

    T2VA has no instruction (there is no picture to align), and REF2VA states its
    alignment inside `retention_analysis` instead.
    """
    end = f"{float(seconds):.2f}"
    last = max(1, int(shots))
    if mode == "I2VA":
        return ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    if mode == "FL2VA":
        return ("How the reference pictures align with the target video — Picture 1 "
                "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot {last}) aligns with the {end}-second mark of the target video.")
    if mode == "L2VA":
        return ("How the reference pictures align with the target video — <Picture 1> "
                f"(from [Shot {last}]) aligns with the {end}-second mark of the target video.")
    return None


def compose(mode, body, soundscape="", music="", seconds=0.0, preamble="", shots=1,
            sections=None):
    """The user's prose -> the sectioned prompt the DiT was trained to read.

    `body` is what the user wrote, with `@handles` already substituted and any
    LoRA trigger words already in front of it — triggers belong inside the
    description, not above the instruction line, because the instruction has to
    be the prompt's first line.

    A blank `soundscape` or `music` emits nothing at all. `N/A` is the guide's
    value for "there is deliberately none of this", which is a real thing to say
    and a very different one from leaving the box empty, so it stays something
    the user types rather than something inferred from an empty string.

    `sections` is `REF_SECTIONS -> prose`, and it comes from the refiner: a
    REF2VA reply carries all three. Handed none of them, this emits the base
    form — a bare sentence wrapped in `integrated_multimodal_description` — and
    that is the right answer for a request with nothing to define. The
    six-section form is what Ref2VA was trained on, so a prompt with references
    in it wants the sections, which is what asking for a REF2VA rewrite gets.

    What no rule can derive is the guide's 350-500 words of shot description.
    This builds the document; the prose inside it is still the user's sentence
    until a refiner writes a better one.
    """
    body = (body or "").strip()
    soundscape = (soundscape or "").strip()
    music = (music or "").strip()
    sections = sections or {}

    out = []

    line = instruction(mode, seconds, shots)
    if line and not _has_instruction(body):
        out.append(line)

    # After the instruction, which has to be the first line, and before the
    # description — the same slot the reference form gives `subject_definitions`.
    preamble = (preamble or "").strip()
    if preamble:
        out.append(preamble)

    # Whether this prompt is written in the reference form, decided by whether
    # there is anything to declare rather than by which mode was derived.
    #
    # It used to be `mode == "REF2VA"`, and a cast in a text-only generation got
    # two of the sections with the base form's body field — a hybrid neither
    # guide describes. The mode is a statement about which slot the reference
    # fills, not about how the prompt is written: the two trainings share an
    # architecture, people run reference-form prompts against T2VA and get what
    # they asked for, and the weights do not police the field name. So a piece
    # that has something to define is written in the form built for defining
    # things, whatever it is about to be encoded as.
    #
    # A bare sentence with no cast and no references still gets the base form.
    # There is nothing to declare there, and `detailed_description:` with no
    # sections above it would be claiming a form the rest of which is missing —
    # which is the same mistake in the other direction.
    reference_form = any(str(sections.get(name) or "").strip() for name in REF_SECTIONS)

    # Each is skipped where the body already carries one, so a prompt somebody
    # has hand-written into full form is not given a second copy of a section it
    # already has.
    for name in (REF_SECTIONS if reference_form else ()):
        value = str(sections.get(name) or "").strip()
        if value and not has_field(body, name):
            out.append(f"{name}: {value}")

    if body:
        # Only wrapped when the body is plain prose. Anything already sectioned —
        # either form — is its own rewrite already.
        field = REF_BODY_FIELD if reference_form else BODY_FIELD
        if not any(has_field(body, f) for f in BODY_FIELDS):
            # The description is written shot by shot and every example opens on
            # a marker. A segment is one shot, so `[Shot 1]` is the whole of it —
            # unless the body already numbers its own, which is someone writing
            # several shots into one generation and knowing that they are.
            if not SHOT_RE.search(body):
                body = f"[Shot 1] {body}"
            body = f"{field}: {body}"
        out.append(body)

    if soundscape and not has_field(body, SOUNDSCAPE_FIELD):
        out.append(f"{SOUNDSCAPE_FIELD}: {soundscape}")
    # `NO_MUSIC` where the user named none: a missing field is a free hand and
    # this field is the one the model most reliably fills on its own. The
    # soundscape above has no such default — `N/A` there is a claim of total
    # silence, which is a real thing to mean and not one to infer from an empty
    # box, so a blank one still emits nothing.
    if (music or body) and not has_field(body, MUSIC_FIELD):
        # `body` guards the default and not the value: an empty request composes
        # to nothing, as it always has, and a piece that is only a music line is
        # still that line.
        out.append(f"{MUSIC_FIELD}: {music or NO_MUSIC}")

    return "\n\n".join(out)
