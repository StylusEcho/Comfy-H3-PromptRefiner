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


def video(ordinal, filename="", sound=False):
    """One reference video, and whether its soundtrack rides with it.

    `sound` is the pairing the H3 node makes: a clip presented with its own audio
    takes an `<Audio j>` immediately before its `<Video k>`. Said as a `track`
    because that is the field `prompting.slot_row` reads to decide what the
    glossary line calls it.
    """
    return Asset(handle=f"{_PREFIX['video']}-{ordinal}", kind="video", role="reference",
                 filename=filename, track="picture+sound" if sound else "picture")


def audio(ordinal, filename=""):
    """One standalone audio reference. Nothing can be shown of it."""
    return Asset(handle=f"{_PREFIX['audio']}-{ordinal}", kind="audio", role="reference",
                 filename=filename)


def derive_mode(first_frame, last_frame, references, videos=(), audios=()):
    """What this request is, in H3's own name for it.

    The shape is read here — something opens, something closes, something is
    cited — exactly as `compile._derive_mode` reads it. References and frames do
    not lock each other out: Ref2VA is the superset training and it is what a
    mixed request runs on, so anything cited is REF2VA whatever else is attached.
    """
    if references or videos or audios:
        return "REF2VA"
    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


def plan(first_frame=None, last_frame=None, references=(), videos=(), audios=()):
    """The attachments -> `([asset], {handle: label})`, in presentation order.

    One ordered walk, because the ordinals in the prompt and the tensors handed
    to the sampler have to agree and there is only one order to read them from.
    It is `compile.plan_references` followed by `compile._trailing_frame_labels`,
    and it is the order H3's own node presents in (`comfy_extras/
    nodes_minimax_h3.py`): pictures, then videos, then standalone audio, with the
    keyframes trailing all of it.

    **A video with a soundtrack is two citations.** Its `<Audio j>` is emitted
    immediately *before* its `<Video k>`, which is the presentation order the
    tokenizer expects. The soundtrack has no handle of its own — nothing points
    at it separately — so its label is filed under `"<video handle>:sound"`, a
    key `normalize_handles` skips when it builds the reverse map and `check`
    still counts as a label something will be given.

    With nothing but keyframes, the walk collapses to `compile._keyframe_labels`
    — start frame `<Picture 1>`, end frame `<Picture 2>` where both are attached,
    and `<Picture 1>` where the end frame is the only one.

    The list is also the order to hand the pictures to the refiner in, so what it
    is shown as `[image 3]` is the third picture in the message.
    """
    ordered, labels = [], {}
    picture = video = audio = 0

    for asset in references:
        picture += 1
        labels[asset.handle] = f"<Picture {picture}>"
        ordered.append(asset)
    for asset in videos:
        if asset.track == "picture+sound":
            audio += 1
            labels[f"{asset.handle}:sound"] = f"<Audio {audio}>"
        video += 1
        labels[asset.handle] = f"<Video {video}>"
        ordered.append(asset)
    for asset in audios:
        audio += 1
        labels[asset.handle] = f"<Audio {audio}>"
        ordered.append(asset)
    for asset in (first_frame, last_frame):
        if asset is not None:
            picture += 1
            labels[asset.handle] = f"<Picture {picture}>"
            ordered.append(asset)
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
