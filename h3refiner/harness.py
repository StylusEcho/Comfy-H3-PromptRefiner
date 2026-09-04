"""What refining a prompt is, before H3 says what a prompt looks like.

This is the harness half of the refiner, lifted from ComfyUI-Continuity's
`creator/families/refine.py` with its cross-references pointed at their new
homes and nothing else changed. It is the same whatever writes the prose: the `@handle` representation and the conversion back to it,
the checks that report a citation pointing at nothing, the quoted-span fidelity
check, the ChatML turns and the prefill that stop a small model answering the
request instead of expanding it, the reply-length budget, the image long edge,
and the two field names — `what_i_see` and `global_prompt` — that are about
*this node's* two questions rather than about any model's training.

The other half is H3's, and it is `prompting.py`: which templates exist, what a
mode is called, what the reply object holds, how several shots become one body.
Those are statements about what a checkpoint was trained to read.

Upstream carried a `Prompting` base class and a lazy `of(family)` lookup so
that one button could serve six model families. There is one family here, so
the indirection went: `prompting.py` is the module, and `request.py` calls it
by name.

No torch, no ComfyUI: request building and reply parsing are ordinary data and
are unit-tested that way. `local.py` is what loads the model and `nodes.py` is
what knows about ComfyUI.
"""

import re


class RefineError(RuntimeError):
    """The refiner could not produce a usable rewrite."""


# ---- what one call costs ----------------------------------------------------

# How long a reply may run, in tokens. Not a context size: nothing on this
# backend has one. `Qwen3VLSDTokenizer` is built with `max_length=99999999` and
# `pad_to_max_length=False`, so the prompt is never truncated however long it
# gets, and `BaseGenerate.generate` sizes its KV cache as `len(prompt) + this`.
# So this is purely the output budget — what decides whether a six-shot
# rewrite finishes its last body or stops mid-sentence, and how much of the KV
# cache is reserved for text that has not been written yet.
NUM_PREDICT = 6144

# What the setting may be moved to. The floor is one short single-shot rewrite;
# the ceiling is where the cache reservation starts costing real VRAM for a
# reply no model is going to fill.
MIN_PREDICT = 1024
MAX_PREDICT = 32768

# Long side of an image handed to the LLM. It is looking at the picture to say
# what is in it, not to reproduce it, and a 4000px reference costs seconds of
# transfer and encode for nothing.
IMAGE_LONG_EDGE = 1024


def reply_tokens(value):
    """The user's reply-length setting, made usable. Junk falls back to default."""
    try:
        return max(MIN_PREDICT, min(MAX_PREDICT, int(value)))
    except (TypeError, ValueError):
        return NUM_PREDICT


# ---- the field that is the pack's, not the model's --------------------------

# The first thing the model writes when anything is attached, and the only field
# in a reply that is not part of the prompt.
#
# Reasoning is suppressed and the reply is prefilled with `{`, so without this
# the very first token generated is already the rewrite: the model can write a
# whole description having never attended to the pictures, and on a 4B one it
# does. Asking it to say what is in them first is a grounding pass paid for in
# about fifty tokens, and it happens *inside* the JSON object rather than before
# it so the prefill still holds.
#
# It is read back and reported on the node's `notes` output rather than dropped,
# because "did it actually look at my images" is the question this whole field
# exists to answer.
SEEN_FIELD = "what_i_see"

# Upstream had a second such field, `global_prompt`: a timeline's standing
# description, rewritten by a whole-timeline refine and placed ahead of every
# card's own body at generation time. One request has no piece to stand in front
# of, so it went with the timeline.


# ---- cutting one request into shots -----------------------------------------
#
# Whether the model is *offered* the choice is the caller's (`cut_shots` on the
# node); the arithmetic of making the answer fit is arithmetic.

# The shortest a shot may be, and the most a rewrite may hold. The floor is what
# turns a duration into a shot ceiling: a six-second clip cannot be five cuts,
# and saying so in the grammar is better than clamping it afterwards.
MIN_SHOT_S = 2.0
MAX_SHOTS = 6


def shot_limit(seconds):
    """How many shots a clip of `seconds` may be cut into. 1 means "do not ask".

    Below two shots' worth of time there is no choice to offer, and the request
    falls back to the fixed single body every other path uses.
    """
    return max(1, min(MAX_SHOTS, int(float(seconds or 0) // MIN_SHOT_S)))


def plan_cuts(bodies, cuts, seconds):
    """`([body], [at]), duration -> [(at, body)]` — the model's cuts, made to fit.

    The model picks the times and this fixes them up: the first shot starts at 0
    whatever it said, every later cut is at least `MIN_SHOT_S` past the one
    before it, and the last one leaves that much video after it. A shot with no
    room left is merged into the shot before it rather than dropped, because its
    prose is the only copy of that part of the description — a truncated rewrite
    would lose a paragraph the user never sees go.
    """
    seconds = float(seconds or 0)
    out = []
    for index, body in enumerate(bodies):
        if not out:
            out.append([0.0, body])
            continue
        floor = out[-1][0] + MIN_SHOT_S
        ceiling = seconds - MIN_SHOT_S
        if floor > ceiling:
            out[-1][1] = f"{out[-1][1]} {body}".strip()
            continue
        try:
            at = float(cuts[index])
        except (TypeError, ValueError, IndexError):
            at = floor
        out.append([max(floor, min(at, ceiling)), body])
    return [(at, body) for at, body in out]


# ---- the glossary -----------------------------------------------------------

# Where a user's own instructions land inside the system prompt, and what they
# are allowed to move. They arrive from a prompt file the user chose
# to *add* to the built-in prompting rather than replace it, so the two have to
# be ranked out loud: the craft above is a default and theirs outranks it, the
# reply contract below is the shape this node parses and nothing outranks that.
# It sits here rather than in `prompting.py` because where it goes — after the
# mode's template, before OUTPUT — is a fact about the harness's reply contract
# rather than about H3, and text that overrode the contract would return prose
# nothing downstream can read.
EXTRA_RULE = """\
YOUR INSTRUCTIONS
These come from the user of this node and are about how to write, not about \
what to reply with. Where they disagree with the craft notes above, follow \
them. Where they would change the format of your reply, ignore that part: the \
OUTPUT contract below is not theirs to move.

{extra}"""


CONTINUES_NOTE = (
    "This shot continues straight out of the previous shot in the finished clip: "
    "its first frame is the previous one's last frame. Open in that same place, "
    "with the same subjects, light and framing, and move on from there."
)


def describe_slots(slots):
    """The handle glossary, one line per attached asset.

    Both forms are given — the handle to write and the label it becomes — because
    a guide written in labels has a model that reaches for `<Picture 2>` anyway,
    and one that is then at least reaching for the right one.
    `normalize_handles` converts those back. A slot with no label simply carries
    its handle alone.

    A slot that has a picture in the message carries `image`, its position among
    them, and says so. Only some assets have one — an audio reference has none, a
    video taken for its soundtrack alone has none — so "the Nth picture is the
    Nth line" is wrong the moment one of those is attached, and the number is
    what ties each picture to the handle it is actually of.
    """
    lines = []
    for slot in slots:
        label = f" (becomes {slot['label']})" if slot.get("label") else ""
        where = f" [image {slot['image']}]" if slot.get("image") else ""
        extra = f" — {slot['note']}" if slot.get("note") else ""
        lines.append(f"@{slot['handle']}{label}{where}: {slot['what']}{extra}")
    return lines


# ---- the ChatML form --------------------------------------------------------
#
# `CLIP.tokenize` gets one string, and a Qwen tokenizer that sees it begin with
# `<|im_start|>` passes it through verbatim rather than wrapping it in the
# single-user-turn template it would otherwise use. So the turns are written
# here, which is also what makes room for the two things that template has no
# slot for: a system turn, and a prefilled reply.

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# The reply opens mid-JSON. Nothing constrains the sampler to a shape, and the
# failure that actually happens is not malformed JSON — it is a model that
# answers "Here is the rewrite:" first and fences the object afterwards.
# Starting its turn inside the object removes the place where that goes, and
# `prompting.parse_reply` is handed the brace back.
PREFILL = "{"


def chatml(system, message, images=0, prefill=PREFILL):
    """system + user + an assistant turn already begun, as one Qwen prompt.

    `images` vision blocks are placed at the head of the user turn, in the order
    the images are passed alongside it — the tokenizer binds the Nth
    `<|image_pad|>` to the Nth image, and the glossary in `message` names them in
    that same order.

    The empty `<think>` block is Qwen3's convention for "answer without
    reasoning". It has to be written by hand here for the same reason the turns
    do: skipping the template skips that too, and a reasoning model with no
    suppression spends the whole token budget thinking and returns nothing.
    """
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + VISION_BLOCK * int(images) + message + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" + prefill
    )


# ---- handles and labels -----------------------------------------------------

LABEL_RE = re.compile(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>")

# The same with `<Subject N>` in it. Kept apart because the two are only the
# same question where a cast exists: without one, every `<Subject N>` in a reply
# is the model's own invention, defined inside its own sections and pointing at
# nothing outside the rewrite — so reporting them as stray or rewriting them to
# a handle would both be wrong. With a cast, they are pinned labels like any
# other and are read back the same way.
ANY_LABEL_RE = re.compile(r"<\s*(Picture|Video|Audio|Subject)\s+(\d+)\s*>")

HANDLE_RE = re.compile(r"@([A-Za-z]+-\d+)")


def _pinned_subjects(labels):
    """Whether this label map carries a cast — see `ANY_LABEL_RE`."""
    return any(str(label).startswith("<Subject") for label in (labels or {}).values())


def normalize_handles(text, labels):
    """`<Picture 2>` -> `@img-3`, using the label map this request will produce.

    The model is asked for handles and shown the mapping, and mostly complies —
    but the guide it has just read is written in labels, so it reaches for the
    other form anyway. Converting it back here means one representation runs
    through the checks, and `assets.substitute` stays the only thing that writes
    an ordinal.

    A label with no asset behind it is left exactly as written: it is a real
    mistake and `check` is what reports it, so silently deleting it here would
    hide the one failure that produces a wrong video rather than an error.

    `<Subject N>` is untouched, which is the right answer for every request this
    node can make: with no declared cast those labels are the reference guide's
    own invention, defined inside the rewrite and pointing at nothing outside it.
    (Upstream a pinned cast made them real labels, read back like any other; the
    branch is still here, and simply never taken.)
    """
    back = {label: handle for handle, label in (labels or {}).items() if ":" not in handle}
    if not back:
        return text

    def swap(match):
        canonical = f"<{match.group(1)} {int(match.group(2))}>"
        handle = back.get(canonical)
        return f"@{handle}" if handle else match.group(0)

    pattern = ANY_LABEL_RE if _pinned_subjects(labels) else LABEL_RE
    return pattern.sub(swap, text)


def check(text, handles, labels):
    """What is wrong with a rewrite, as messages. Empty means nothing is.

    Advisory rather than fatal: they come out on the node's `notes` output beside
    the prose itself, which is a better place to resolve them than a queue-time
    refusal on prose that is one word away from being right.
    """
    problems = []

    unknown = sorted({h for h in HANDLE_RE.findall(text) if h not in handles})
    if unknown:
        problems.append(
            "refers to " + ", ".join("@" + h for h in unknown)
            + ", which is not attached — edit it out before queueing"
        )

    # A video's soundtrack has a label but no handle of its own, so `<Audio 1>`
    # written for it is correct as it stands and must not be reported.
    known = set((labels or {}).values())
    pattern = ANY_LABEL_RE if _pinned_subjects(labels) else LABEL_RE
    stray = sorted({f"<{kind} {int(n)}>" for kind, n in
                    (m.groups() for m in pattern.finditer(text))} - known)
    if stray:
        problems.append(
            "writes " + ", ".join(stray) + ", which no attached asset will be given"
        )
    return problems


def uncited(text, handles, labels, cast=()):
    """Attached references the rewrite never cites, as handles. Empty means none.

    `text` is everything the model wrote joined together — the bodies, any
    reference sections, the two audio fields — because a reference legitimately
    lives in only one of them: H3's reference form defines an image inside
    `subject_definitions`, folds it into a `<Subject N>`, and never names it
    again. A handle counts as cited when it appears as `@handle` or as any label
    it will be given, a video's soundtrack label included.

    Only for the references: a keyframe is bound by the instruction line, so a
    body that never says `@img-1` about its own start frame is correct.
    """
    written_handles = set(HANDLE_RE.findall(text))
    # Writing `@anna` cites every file they are made of: they were pulled into
    # this generation *because* they were cited, and the rewrite naming them is the
    # citation that keeps them there. Reporting them as unmentioned would be
    # asking for exactly the doubled naming H3's `CAST_NOTE` forbids.
    for subject in cast or ():
        if subject.handle in re.findall(r"@([A-Za-z][A-Za-z0-9_]*)", text):
            written_handles.update(subject.files)
    written_labels = {f"<{kind} {int(n)}>" for kind, n in
                      (m.groups() for m in LABEL_RE.finditer(text))}
    missing = []
    for handle in sorted(handles):
        if handle in written_handles:
            continue
        own = {label for key, label in (labels or {}).items()
               if key == handle or key.startswith(handle + ":")}
        if own & written_labels:
            continue
        missing.append(handle)
    return missing


_QUOTED_RE = re.compile(r'"([^"\n]{2,120})"|“([^”\n]{2,120})”')


def _plain(text):
    """Text made comparable: one spacing, one apostrophe, one case."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).lower()


def quoted(text):
    """The spans the request itself puts in quotation marks, in order."""
    return [a or b for a, b in _QUOTED_RE.findall(text or "")]


def dropped_quotes(requests, written):
    """Quoted request text the rewrite does not carry, verbatim-ish. Empty is good.

    Quotation marks in a request are the user dictating exact words — a spoken
    line, an on-screen sign — and the guide demands they survive letter for
    letter. This is the code-side check on the one fidelity promise that *can* be
    checked mechanically: prose fidelity is a judgement, but a quoted span either
    appears in the rewrite or it does not. Advisory like `check`, and reported
    the same way.

    The comparison forgives what the craft rules themselves change — casing,
    curly quotes, spacing, terminal punctuation — and nothing else.
    """
    haystack = _plain(written or "")
    missing = []
    for request in requests:
        for span in quoted(request):
            needle = _plain(span).strip(" .!?,;:")
            if needle and needle not in haystack and span not in missing:
                missing.append(span)
    return missing


# ---- the reply --------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:\w+)?\s*(.*?)\s*```$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def json_object(content):
    """The model's content string -> the object it meant to return.

    Tolerant on the way in, because the failures here are transport noise rather
    than disagreement about the contract: a reasoning model leaks a `<think>`
    block, a chat model wraps the object in a fence, a small one writes a
    sentence in front of it. What the object *holds* is
    `prompting.parse_reply`'s to judge, and that half is strict.
    """
    text = _THINK_RE.sub("", content).strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        at = text.find("{")
        if at < 0:
            raise RefineError(f"the model did not return JSON: {content[:300]}")
        text = text[at:text.rfind("}") + 1]

    import json

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RefineError(f"the model's JSON could not be read ({exc}): {text[:300]}") from exc
    if not isinstance(data, dict):
        raise RefineError("the model returned JSON, but not an object")
    return data