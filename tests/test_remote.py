"""Where the API key comes from, and where it refuses to go.

    python3 tests/test_remote.py

No network: `_request` is never reached. What is checked is everything decided
before a socket is opened, which is where every way of leaking a credential
lives.

The threat this is written against is the one ComfyUI's own architecture
creates: the server has no authentication, so anyone who can queue a prompt can
also edit the base URL on a node. Left alone, that is enough to point the next
refine at a host of their choosing and have it deliver the stored key. Upstream
answered it by binding a saved key to the URL it was saved with; here there is
also an environment key, so it is bound the same way — to `URL_ENV`, when one is
set — and `_headers` still refuses to put any key on plain http to anywhere but
loopback.
"""

import os

import layout

pkg = layout.load("remote", "harness")
remote, refine = pkg.remote, pkg.harness

from harness import FAILURES, check, expect, passed  # noqa: E402


def env(**values):
    """Set these variables for one call, and put the environment back after."""
    def wrap(fn):
        def run(*args):
            before = {name: os.environ.get(name) for name in values}
            os.environ.update({k: v for k, v in values.items() if v is not None})
            for name, value in values.items():
                if value is None:
                    os.environ.pop(name, None)
            try:
                return fn(*args)
            finally:
                for name, value in before.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        return run
    return wrap


def no_file(fn):
    """Run `fn` with no stored credentials file — the ordinary install."""
    def run(*args):
        original = remote._read
        remote._read = lambda: {}
        try:
            return fn(*args)
        finally:
            remote._read = original
    return run


def stored(url="", key=""):
    def wrap(fn):
        def run(*args):
            original = remote._read
            remote._read = lambda: {"url": url, "key": key}
            try:
                return fn(*args)
            finally:
                remote._read = original
        return run
    return wrap


# ---- the URL ----------------------------------------------------------------


def test_normalize_url_forgives_the_obvious_paste():
    check("a trailing slash goes",
          remote.normalize_url("http://localhost:1234/v1/") == "http://localhost:1234/v1")
    check("a copied chat endpoint goes",
          remote.normalize_url("https://api.example.com/v1/chat/completions")
          == "https://api.example.com/v1")


def test_normalize_url_refuses_a_scheme_that_is_not_http():
    expect("file: is a disk read dressed as a setting",
           lambda: remote.normalize_url("file:///etc/passwd"),
           "must start with http:// or https://")
    expect("and a bare host is not a URL",
           lambda: remote.normalize_url("localhost:1234"),
           "must start with http:// or https://")


# ---- where the key comes from -----------------------------------------------


@no_file
@env(H3_REFINER_API_KEY=None, H3_REFINER_BASE_URL=None)
def test_the_node_url_wins_and_carries_no_key():
    url, key = remote.endpoint("http://localhost:1234/v1")
    check("the widget is the endpoint", url == "http://localhost:1234/v1", url)
    check("and a local server with no key set gets none", key == "", "a key appeared")


@no_file
@env(H3_REFINER_API_KEY="sk-secret", H3_REFINER_BASE_URL=None)
def test_an_environment_key_is_used_where_nothing_pins_it():
    _, key = remote.endpoint("https://api.example.com/v1")
    check("an unpinned key follows the node", key == "sk-secret", "the key was dropped")


@no_file
@env(H3_REFINER_API_KEY="sk-secret", H3_REFINER_BASE_URL="https://api.example.com/v1")
def test_a_pinned_key_refuses_to_follow_the_node_somewhere_else():
    _, key = remote.endpoint("https://api.example.com/v1")
    check("it travels to the address it is pinned to", key == "sk-secret")
    expect("and refuses to travel anywhere else",
           lambda: remote.endpoint("http://attacker.example/v1"),
           "is pinned to")


@no_file
@env(H3_REFINER_API_KEY=None, H3_REFINER_BASE_URL="https://api.example.com/v1")
def test_the_environment_url_is_the_fallback_for_an_empty_widget():
    url, _ = remote.endpoint("")
    check("an empty widget falls back to the variable",
          url == "https://api.example.com/v1", url)


@stored(url="https://saved.example.com/v1", key="sk-saved")
@env(H3_REFINER_API_KEY=None, H3_REFINER_BASE_URL=None)
def test_a_saved_key_is_still_bound_to_the_url_it_was_saved_with():
    _, key = remote.endpoint("https://saved.example.com/v1")
    check("the endpoint it was earned for gets it", key == "sk-saved")
    _, moved = remote.endpoint("http://attacker.example/v1")
    check("and moving the endpoint drops it", moved == "", "the key followed the move")


@stored(url="https://saved.example.com/v1", key="sk-saved")
@env(H3_REFINER_API_KEY=None, H3_REFINER_BASE_URL=None)
def test_status_says_whether_not_what():
    reported = remote.status()
    check("the URL may be said", reported["url"] == "https://saved.example.com/v1")
    check("the key may only be acknowledged", reported["key_set"] is True)
    check("and never appears", "sk-saved" not in str(reported), str(reported))


# ---- where the key refuses to go --------------------------------------------


def test_a_key_never_rides_plain_http_off_this_machine():
    expect("http to a remote host with a key attached is refused",
           lambda: remote._headers("http://box.example.com/v1", "sk-secret"),
           "will not send your API key over plain http")
    check("https is fine",
          remote._headers("https://api.example.com/v1", "sk-secret")["Authorization"]
          == "Bearer sk-secret")
    check("and so is loopback, which never leaves the machine",
          remote._headers("http://localhost:1234/v1", "sk-secret")["Authorization"]
          == "Bearer sk-secret")


def test_no_key_means_no_authorization_header_at_all():
    check("an open local server is asked plainly",
          "Authorization" not in remote._headers("http://localhost:1234/v1", ""))


def test_a_key_is_blotted_out_of_anything_that_might_carry_it():
    check("the key does not survive into a message",
          remote._scrub("failed with sk-secret", "sk-secret") == "failed with •••")


# ---- ejecting ----------------------------------------------------------------


def test_only_a_machine_that_could_hold_our_model_is_asked_to_free_it():
    check("loopback is ours", remote._private("localhost"))
    check("a LAN address is ours", remote._private("192.168.1.40"))
    check("a bare hostname is a LAN name", remote._private("lmstudio-box"))
    check("a hosted API is not", not remote._private("api.example.com"))


@no_file
@env(H3_REFINER_API_KEY=None, H3_REFINER_BASE_URL=None)
def test_a_hosted_api_is_never_sent_an_unload():
    check("there is nothing of ours there to free",
          remote.unload("gpt-x", "https://api.example.com/v1", "sk-secret") == "")


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_"):
            value()
    passed("test_remote")
