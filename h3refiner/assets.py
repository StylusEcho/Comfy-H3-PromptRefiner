"""What is attached, what it is called, and which mode that makes this.

Upstream this was scattered through a two-thousand-line compiler that turned a
whole timeline into a ComfyUI graph: `compile.Asset`, `compile._derive_mode`,
`compile.plan_references`, `compile._keyframe_labels`,
`compile._trailing_frame_labels`, `compile._substitute`. A node with three image
sockets needs the same six answers and none of the rest, so they are here, in a
module with no ComfyUI in it.

**Handles are the representation, labels are the output.** The refiner is asked
for `@img-1` and shown the label each handle will be given, because a rewrite
written in ordinals goes stale the moment a picture is added or removed — that
was true of a stored blob upstream and it is true of prose a user keeps in a
text box here. `substitute` is the last step, and it is what puts `<Picture 2>`
into the prompt that leaves this node.

**The ordinals are the presentation order, and it is not the socket order.**
References are presented to the tokenizer first and keep the ordinals they would
have had on their own; start and end frames trail them (see `plan`). That is
what makes a reference-only prompt byte-identical whether or not a keyframe was
attached alongside it, and it is the order the encoder is fed in.
"""

import re
from dataclasses import dataclass


class AttachError(ValueError):
    """What is attached cannot be described as one H3 request."""


# What the tokenizer is handed, in the guide's own citation forms. `Video` and
# `Audio` are here because the glossary and the reply can carry them: this node
# has no socket for a clip, but a request that names one by hand still reads.
_CITE = {"image": "Picture", "video": "Video", "audio": "Audio"}

# The prefix a handle takes, per kind — `@img-1`, `@vid-1`, `@aud-1`.
_PREFIX = {"image": "img", "video": "vid", "audio": "aud"}

# The same shape `compile.HANDLE_RE` matched, so prose written against the
# Creator node reads here unchanged.
HANDLE_RE = re.compile(r"@([A-Za-z]+-\d+)")

# What a reference image may be narrowed to. `full` is the whole picture and is
# the default; the four others are `prompting._TAKES_NOTE`'s keys, and picking
# one is what tells the refiner to define the subject alone and retain nothing
# else from that picture.
TAKES = ("full", "person", "object", "scene", "style")


@dataclass
class Asset:
    """One attached file, as the glossary and the label plan see it.

    A trimmed `compile.Asset`: the fields this node can actually set. `filename`
    is shown to the model in the glossary line — a name is a real clue about
    what a picture holds — and is otherwise unused, so a socket with no file
    behind it passes the label it was given instead.
    """

    handle: str          # "img-1", "vid-1", "aud-1" — what the request types after @
    kind: str            # image | video | audio
    role: str            # reference | first_frame | last_frame
    filename: str = ""   # what to call it in the glossary
    track: str = None    # video only: "picture", "picture+sound", "sound"
    takes: str = "full"  # reference: one of TAKES; what of it is the reference


def image(role, ordinal, filename="", takes="full"):
    """One image asset, handled by its position among the images."""
    if takes not in TAKES:
        raise AttachError(f"unknown reference scope {takes!r} — one of {', '.join(TAKES)}")
    return Asset(handle=f"{_PREFIX['image']}-{ordinal}", kind="image", role=role,
                 filename=filename, takes=takes)


def derive_mode(first_frame, last_frame, references):
    """What this request is, in H3's own name for it.

    The shape is read here — something opens, something closes, something is
    cited — exactly as `compile._derive_mode` reads it. References and frames do
    not lock each other out: Ref2VA is the superset training and it is what a
    mixed request runs on, so anything with a reference in it is REF2VA whatever
    else is attached.
    """
    if references:
        return "REF2VA"
    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


def plan(first_frame=None, last_frame=None, references=()):
    """The attachments -> `([asset], {handle: label})`, in presentation order.

    One ordered walk, because the ordinals in the prompt and the tensors in the
    payload have to agree and there is only one order to read them from. It is
    `compile.plan_references` followed by `compile._trailing_frame_labels`: every
    reference takes the `<Picture N>` it would have had on its own, and the
    frames take the next ordinals after them.

    With no references at all, the frames are the whole presentation and the
    walk collapses to `compile._keyframe_labels` — start frame `<Picture 1>`, end
    frame `<Picture 2>` where both are attached, and `<Picture 1>` where the end
    frame is the only one.

    The list is the order to hand the pictures to the encoder in. The node's own
    sockets are read in this order too, so what the refiner is shown as
    `[image 3]` is the third picture in the message.
    """
    ordered = list(references) + [a for a in (first_frame, last_frame) if a is not None]
    labels, counts = {}, {"image": 0, "video": 0, "audio": 0}
    for asset in ordered:
        counts[asset.kind] += 1
        labels[asset.handle] = f"<{_CITE[asset.kind]} {counts[asset.kind]}>"
    return ordered, labels


def substitute(text, labels, handles, where="the rewrite"):
    """Replace every `@handle` with the label it was given.

    Only handles that name something attached are touched, so ordinary prose
    ("meet me @ 5") survives. A handle-shaped token with no asset behind it is an
    error rather than a silent pass-through: it means the prompt refers to a
    picture that will not be in the payload, and `<Picture 4>` written for a
    third picture points at nothing.

    `where` names the field, because this runs over more than the body: a REF2VA
    rewrite cites its references inside `subject_definitions` and
    `retention_analysis` and never again, so those are substituted with the same
    labels.
    """
    dangling = sorted({h for h in HANDLE_RE.findall(text or "") if h not in handles})
    if dangling:
        raise AttachError(
            f"{where} references " + ", ".join("@" + h for h in dangling)
            + " but no such asset is attached"
        )
    return HANDLE_RE.sub(lambda m: labels.get(m.group(1), m.group(0)), text or "")
