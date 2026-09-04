"""The nodes themselves: where ComfyUI meets everything above it.

Three of them, and the split is the same one `request.py` describes. Two say
*where the model runs* and hold nothing else — `H3RefinerLocalModel` for the
text encoder loaded in this process, `H3RefinerServer` for the OpenAI-compatible
server somebody already keeps resident — and both output an `H3_REFINER`, which
is a backend and its settings and no prose at all. `H3PromptRefiner` is the one
that has the prompt box, takes an `H3_REFINER` on an input, and returns the
document.

Two nodes rather than a backend dropdown on one, because the settings do not
overlap: a text encoder is picked from disk and a server is an address, a model
name and a variable holding a key, and a single node carrying both would show
every user four widgets that mean nothing to them. It also keeps the address out
of the refiner node, so a workflow shared with somebody else carries their
server's settings on a node they can see and replace.

**Why the key is not a widget.** A widget's value is saved into the workflow
`.json`, and a workflow is a thing people hand around, paste into issues and
publish beside a render. So `H3RefinerServer` takes the *name* of an environment
variable, never a key, and `remote.endpoint` is what reads it — see there for
what binds a key to the address it may travel to.
"""

from . import assets, harness, local, prompting, remote, request, skills

# What the two backend nodes hand the refiner. A dict, not a class, because
# every field in it is a setting somebody typed and nothing in it is state.
REFINER = "H3_REFINER"

CATEGORY = "H3 Prompt Refiner"


def _pil(image, index=0):
    """One frame of a ComfyUI `IMAGE` batch -> a PIL image.

    ComfyUI passes pictures around as `[B, H, W, C]` floats in 0..1. The
    backends want a PIL image because that is what both of them narrow from —
    `local.to_tensor` downscales and hands back a tensor, `remote.to_data_url`
    downscales and encodes — and doing the conversion once here means the two
    paths cannot disagree about what the model was shown.
    """
    import numpy
    from PIL import Image

    frame = image[index].detach().cpu().numpy()
    return Image.fromarray((numpy.clip(frame, 0.0, 1.0) * 255.0).round().astype("uint8"))


def _frames(image):
    """Every frame of an `IMAGE` batch, as PIL images. `None` -> no frames."""
    if image is None:
        return []
    return [_pil(image, index) for index in range(image.shape[0])]


def _skill_names():
    """The instruction files installed, for the widget. Always with "none" first."""
    return ["none"] + list(skills.list_skills())


class H3RefinerLocalModel:
    """A Qwen3-VL text encoder, loaded in this process, as a refiner backend.

    The default for the reason `local.py` states at length: a second runtime
    with its own copy of a model is VRAM ComfyUI can neither see nor reclaim,
    and on a machine already streaming H3's own encoder off system RAM that is
    the difference between a rewrite that takes twenty seconds and one that
    takes ten minutes. The weights are released the moment the generation ends,
    so the sampler downstream gets the space back.

    It is **not** H3's own encoder — that checkpoint is truncated and has no
    language head. Load a separate Qwen3-VL 4B or 8B; `local._check` refuses the
    rest by name rather than returning tokens nobody should read.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_encoder": (local.list_models(),)}}

    RETURN_TYPES = (REFINER,)
    RETURN_NAMES = ("refiner",)
    FUNCTION = "backend"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The refiner's model, loaded here as a ComfyUI text encoder. "
                   "A separate Qwen3-VL 4B or 8B — not H3's own encoder, which is "
                   "truncated and cannot generate text.")

    def backend(self, text_encoder):
        return ({"where": "local", "model": text_encoder},)


class H3RefinerServer:
    """An OpenAI-compatible server, as a refiner backend.

    For the machine where the in-process load is the redundant copy: an LM
    Studio or Ollama already resident for other work, a box on the LAN with the
    big model, a hosted API. One client covers all of them — LM Studio, Ollama's
    `/v1`, llama.cpp's server, vLLM, KoboldCpp, OpenRouter and OpenAI speak the
    same chat-completions dialect, and Anthropic and Gemini publish
    compatibility endpoints for it.

    `api_key_env` is the *name* of an environment variable, never a key: see the
    module docstring. Leave it as it is for a local server that wants no key —
    an unset variable is simply no key, not an error.

    `eject` asks the server to drop the model again once the rewrite is in hand,
    which on one machine is the whole difference between a refine and a slower
    render. Asked only of a server on this machine or this LAN, and failing to
    free memory never fails a rewrite that already succeeded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "", "tooltip":
                                     "The model name as the server lists it."}),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1",
                                        "tooltip": "The server's OpenAI-compatible "
                                                   "base URL. Empty falls back to "
                                                   f"${remote.URL_ENV}."}),
                "api_key_env": ("STRING", {"default": remote.KEY_ENV, "tooltip":
                                           "The name of an environment variable "
                                           "holding the API key. Never the key "
                                           "itself — widgets are saved into the "
                                           "workflow."}),
                "eject": ("BOOLEAN", {"default": False, "tooltip":
                                      "Ask the server to unload the model once the "
                                      "rewrite is done. Local and LAN servers only."}),
            }
        }

    RETURN_TYPES = (REFINER,)
    RETURN_NAMES = ("refiner",)
    FUNCTION = "backend"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The refiner over an OpenAI-compatible server you already run "
                   "— LM Studio, Ollama, llama.cpp, vLLM, or a hosted API.")

    def backend(self, model, base_url, api_key_env, eject):
        return ({"where": "remote", "model": model, "url": base_url,
                 "key_env": api_key_env, "eject": eject},)


def _backend(refiner):
    """An `H3_REFINER` -> `(chat, look)`.

    `look` turns a PIL image into whatever that backend's `chat` attaches — a
    tensor in-process, a `data:` URL on the wire. Bound together because the two
    always travel together and a mismatch would be a picture the model cannot
    read rather than an error.
    """
    import functools

    if not isinstance(refiner, dict) or "where" not in refiner:
        raise harness.RefineError(
            "no refiner backend is connected — add an H3 Refiner Model or H3 "
            "Refiner Server node and wire it into `refiner`")
    if refiner["where"] == "remote":
        url, key = remote.endpoint(refiner.get("url", ""),
                                   refiner.get("key_env", remote.KEY_ENV))
        chat = functools.partial(remote.chat, url=url, key=key,
                                 eject=bool(refiner.get("eject")))
        return chat, remote.to_data_url
    return local.chat, local.to_tensor


class H3PromptRefiner:
    """A sentence in, the description MiniMax H3 was trained to read out.

    H3 is two models. The hosted half rewrites what you typed into a labelled,
    sectioned intermediate representation, and the open weights were only ever
    trained on that output — which is why a plain sentence gets so much less out
    of them than the samples suggest. This node is the local stand-in for that
    hosted half: a small vision LLM expands the request, and the field names,
    the instruction line, the `[Shot N]` markers and the cut times are assembled
    around its prose.

    Attach the same pictures you are about to give the sampler. What is plugged
    in decides the mode — nothing is T2VA, a start frame is I2VA, an end frame
    L2VA, both FL2VA, and anything on `references` is REF2VA, which is the
    six-section reference form. `template` pins one of those over the derived
    answer; the pin is honoured and a note says what crossing the reference
    boundary costs.

    The references are labelled in the order they are presented — every
    reference takes the `<Picture N>` it would have had on its own, and the
    start and end frames take the ordinals after them — so feed the sampler its
    pictures in that same order or the prompt will point at the wrong one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner": (REFINER,),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What you want, in a sentence. It is "
                                                 "the specification: everything it "
                                                 "names survives into the rewrite."}),
                "seconds": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 120.0,
                                      "step": 0.1, "tooltip":
                                      "How long the finished video runs. Written into "
                                      "the alignment line, and what decides how many "
                                      "shots the rewrite may hold."}),
                "template": (list(prompting.TEMPLATES), {"default": "auto",
                             "tooltip": "Which form the rewrite is written in. "
                                        "`auto` follows what is attached."}),
                "cut_shots": ("BOOLEAN", {"default": True, "tooltip":
                              "Let the model divide the request into several shots "
                              "and pick where the cuts land."}),
                "language": ("STRING", {"default": "English", "tooltip":
                             "The language the prose and any dialogue are written in. "
                             "The structural syntax stays English either way."}),
                "temperature": ("FLOAT", {"default": request.TEMPERATURE, "min": 0.0,
                                          "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "reply_tokens": ("INT", {"default": harness.NUM_PREDICT,
                                         "min": harness.MIN_PREDICT,
                                         "max": harness.MAX_PREDICT, "step": 256,
                                         "tooltip": "How long the reply may run. Not a "
                                                    "context size — the prompt is never "
                                                    "truncated."}),
            },
            "optional": {
                "start_frame": ("IMAGE", {"tooltip": "The video's first frame."}),
                "end_frame": ("IMAGE", {"tooltip": "The video's final frame."}),
                "references": ("IMAGE", {"tooltip": "Reference pictures — a batch is "
                                                    "several of them, labelled in "
                                                    "batch order."}),
                "reference_takes": (list(assets.TAKES), {"default": "full",
                                    "tooltip": "What of the references is the "
                                               "reference. `full` is the whole "
                                               "picture."}),
                "instructions": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Your own writing instructions, added to "
                                             "the built-in prompting. They outrank the "
                                             "craft notes and never the reply format."}),
                "skill": (_skill_names(), {"default": "none", "tooltip":
                          "An instruction file from the node's skills/ folder."}),
                "skill_mode": (list(skills.MODES), {"default": skills.ADD,
                               "tooltip": "`add` keeps the built-in prompting and joins "
                                          "the file onto it; `replace` hands the file "
                                          "over as the whole instruction."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "description", "soundscape", "music", "notes")
    OUTPUT_TOOLTIPS = (
        "The finished Context-IR document. This is what goes to the text encoder.",
        "The rewritten prose alone, without the sections around it.",
        "What the video sounds like.",
        "The score, or empty where the request asked for none.",
        "What the model saw in your pictures, and anything wrong with the rewrite.",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Expands a sentence into the sectioned description MiniMax H3 "
                   "was trained to read, using a small vision LLM.")

    def run(self, refiner, prompt, seconds, template, cut_shots, language,
            temperature, seed, reply_tokens, start_frame=None, end_frame=None,
            references=None, reference_takes="full", instructions="", skill="none",
            skill_mode=skills.ADD):
        chat, look = _backend(refiner)

        # The presentation order, which is what `assets.plan` labels against:
        # references first, then the frames. Built here because this is where the
        # sockets are, and handed over as ordinary assets from there on.
        pictures, refs = {}, []
        for index, picture in enumerate(_frames(references), start=1):
            asset = assets.image("reference", index, f"reference {index}",
                                 takes=reference_takes)
            refs.append(asset)
            pictures[asset.handle] = picture

        first = last = None
        ordinal = len(refs)
        for role, image, name in (("first_frame", start_frame, "start frame"),
                                  ("last_frame", end_frame, "end frame")):
            frames = _frames(image)
            if not frames:
                continue
            ordinal += 1
            asset = assets.image(role, ordinal, name)
            pictures[asset.handle] = frames[0]
            if role == "first_frame":
                first = asset
            else:
                last = asset

        loaded = None
        if skill and skill != "none":
            loaded = skills.load(skill)

        result = request.refine(
            prompt, chat, look,
            model=refiner.get("model", ""),
            first_frame=first, last_frame=last, references=refs, pictures=pictures,
            seconds=seconds, template=template, language=language,
            temperature=temperature, seed=seed, max_tokens=reply_tokens,
            # A skill set to `replace` takes the whole prompt over; one set to
            # `add` is text joined onto the built-in prompting, alongside whatever
            # was typed into `instructions`. Exactly one of the two is ever set.
            extra="\n\n".join(t for t in (
                instructions,
                skills.instructions(loaded) if loaded and skill_mode == skills.ADD else "",
            ) if (t or "").strip()),
            skill=loaded if loaded and skill_mode == skills.REPLACE else None,
            cut_shots=cut_shots,
        )
        return (result["prompt"], result["body"], result["soundscape"],
                result["music"], _notes(result))


def _notes(result):
    """What the model saw and what is wrong with the rewrite, as one readout.

    One socket rather than two, because they are read together and separately
    they are both usually empty. What the model saw comes first — "did it
    actually look at my images" is the question that field exists to answer —
    and the problems follow as a list, worded as they were raised.
    """
    lines = []
    if result.get("seen"):
        lines += ["What the model saw:", result["seen"]]
    if result.get("problems"):
        if lines:
            lines.append("")
        lines.append("Check the rewrite:")
        lines += [f"- {problem}" for problem in result["problems"]]
    if not lines:
        template = result["mode"]
        skill = result.get("skill")
        lines.append(f"Written under {template}"
                     + (f" by the {skill} skill." if skill else "."))
    return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "H3PromptRefiner": H3PromptRefiner,
    "H3RefinerLocalModel": H3RefinerLocalModel,
    "H3RefinerServer": H3RefinerServer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptRefiner": "H3 Prompt Refiner",
    "H3RefinerLocalModel": "H3 Refiner Model",
    "H3RefinerServer": "H3 Refiner Server",
}
