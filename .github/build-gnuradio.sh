#!/usr/bin/env bash
#
# Build and install GNU Radio from source into the active conda prefix.
#
# In its own file so the cache key can hash it: what CI caches is gnuradio as
# configured here, so the recipe has to be part of the key. See docs/CI.md.
#
# Expects an activated conda environment (CONDA_PREFIX) and GNURADIO_REF.

set -euo pipefail

: "${CONDA_PREFIX:?must be run with the conda environment activated}"
: "${GNURADIO_REF:?must be set to a gnuradio commit sha}"

src="${RUNNER_TEMP:-/tmp}/gnuradio-src"
rm -rf "$src"

# One commit, not a clone: the pin is an exact sha.
git init -q "$src"
git -C "$src" remote add origin https://github.com/gnuradio/gnuradio.git
git -C "$src" fetch -q --depth 1 origin "$GNURADIO_REF"
git -C "$src" checkout -q FETCH_HEAD

# gr-cuda links only gnuradio-runtime and imports only gr, blocks, fft and
# gr_unittest, so the GUI/hardware/docs components come out. QTGUI also fails
# headless.
#
# ENABLE_TESTING must stay ON despite being the obvious saving: gnuradio
# templates it into the installed GnuradioConfig.cmake, and with it off every
# downstream OOT calling GR_ADD_CPP_TEST fails to configure on a missing
# Boost::unit_test_framework. gnuradio's tests are built here, never run.
cmake -S "$src" -B "$src/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DENABLE_DOXYGEN=OFF \
    -DENABLE_GR_QTGUI=OFF \
    -DENABLE_GR_UHD=OFF \
    -DENABLE_GR_AUDIO=OFF \
    -DENABLE_GR_VIDEO_SDL=OFF \
    -DENABLE_GR_SOAPY=OFF \
    -DENABLE_GR_IIO=OFF

cmake --build "$src/build" -j "$(nproc)"
cmake --install "$src/build"

rm -rf "$src"
