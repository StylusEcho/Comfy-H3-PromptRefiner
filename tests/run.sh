#!/bin/sh
# Every suite, in one go. No pytest, no torch, no ComfyUI — see tests/harness.py.
#
#     sh tests/run.sh
#
# Exits non-zero on the first suite that fails, so it is usable as a hook.
set -e
cd "$(dirname "$0")/.."
for suite in tests/test_*.py; do
    python3 "$suite"
done
