#!/usr/bin/env bash
# Rebuild the submission from a fresh clone and compare it byte for byte with a given CSV.
#
#   scripts/clean_clone_check.sh YT.csv [repo-url-or-path] [ref]
#
# MTAH_DATA_DIR must point at the unpacked data bundle (the folder that holds tactical/).
# The clone parses the raw tracking itself (--no-cache), so a stale cache cannot hide a change.
set -euo pipefail

submitted="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
repo="${2:-$(git rev-parse --show-toplevel)}"
ref="${3:-HEAD}"
: "${MTAH_DATA_DIR:?set MTAH_DATA_DIR to the folder that holds tactical/}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -d "$repo" ]; then
    ref="$(git -C "$repo" rev-parse "$ref")"
fi
git clone --quiet "$repo" "$tmp/repo"
git -C "$tmp/repo" checkout --quiet "$ref"

echo "rebuilding from $(git -C "$tmp/repo" rev-parse --short HEAD) with data in $MTAH_DATA_DIR"
(cd "$tmp/repo" && python run_pipeline.py --no-cache --out "$tmp/rebuilt.csv" >/dev/null)

a="$(sha256sum "$submitted" | cut -d' ' -f1)"
b="$(sha256sum "$tmp/rebuilt.csv" | cut -d' ' -f1)"
echo "submitted $a  $submitted"
echo "rebuilt   $b"
if [ "$a" = "$b" ]; then
    echo "OK: identical"
else
    echo "MISMATCH" >&2
    exit 1
fi
