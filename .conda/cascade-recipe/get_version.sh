#!/usr/bin/env bash
# Derives the Cascade fork's package version.
#
# On an exact "v<basever>-cascade.<n>" tag: prints "<basever>+cascade.<n>.<shortsha>".
# Otherwise: prints "<basever>+cascade.dev.<shortsha>", where <basever> comes from
# CMakeLists.txt's VERSION_MAJOR/API/ABI/PATCH.
#
# --strict: require the exact cascade tag; fail instead of falling back.
set -euo pipefail

cascade_re='^v([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)-cascade\.([0-9]+)$'

strict=false
[[ "${1:-}" == "--strict" ]] && strict=true

shortsha=$(git rev-parse --short HEAD)

for t in $(git tag --points-at HEAD); do
    if [[ "$t" =~ $cascade_re ]]; then
        echo "${BASH_REMATCH[1]}+cascade.${BASH_REMATCH[2]}.${shortsha}"
        exit 0
    fi
done

if $strict; then
    echo "error: HEAD has no exact v<basever>-cascade.<n> tag" >&2
    exit 1
fi

version_var() {
    sed -n "s/^[Ss][Ee][Tt](${1} *\([A-Za-z0-9]*\)).*/\1/p" CMakeLists.txt | head -1
}
major=$(version_var VERSION_MAJOR)
api=$(version_var VERSION_API)
abi=$(version_var VERSION_ABI)
patch=$(version_var VERSION_PATCH)
[[ "$patch" =~ ^[0-9]+$ ]] || patch=0

echo "${major}.${api}.${abi}.${patch}+cascade.dev.${shortsha}"
