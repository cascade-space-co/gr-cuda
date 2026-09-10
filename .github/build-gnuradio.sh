#!/usr/bin/env bash
#
# Build and install GNU Radio from source into the active conda prefix.
#
# This lives in its own file rather than inline in gpu-qa.yml so that the cache
# key can hash it. What the workflow caches is gnuradio *as configured here*, so
# the recipe has to be part of the key or a flag change silently reuses an
# environment built with the old flags. Hashing the whole workflow file did that
# correctly but far too broadly: editing a comment or a trigger forced a
# 21-minute rebuild and stored another 2 GB against a 10 GB cache cap. This file
# changes when the build changes, and not otherwise.
#
# Expects an activated conda environment (CONDA_PREFIX) and GNURADIO_REF.

set -euo pipefail

: "${CONDA_PREFIX:?must be run with the conda environment activated}"
: "${GNURADIO_REF:?must be set to a gnuradio commit sha}"

src="${RUNNER_TEMP:-/tmp}/gnuradio-src"
rm -rf "$src"

# Fetching one commit rather than cloning: the pin is an exact sha, and the
# history is not wanted.
git init -q "$src"
git -C "$src" remote add origin https://github.com/gnuradio/gnuradio.git
git -C "$src" fetch -q --depth 1 origin "$GNURADIO_REF"
git -C "$src" checkout -q FETCH_HEAD

# gr-cuda links exactly one gnuradio target (gnuradio-runtime, see
# lib/CMakeLists.txt) and its QA suite imports only gr, blocks, fft and
# gr_unittest. Everything switched off below is a GUI, hardware or docs
# component that gr-cuda neither links nor imports. QTGUI matters twice over:
# it is the largest component, and it is the one that fails on a headless
# runner.
#
# ENABLE_TESTING is deliberately NOT switched off, though it would be faster.
# gnuradio templates its build-time value into the installed
# GnuradioConfig.cmake, which only requests Boost's unit_test_framework
# component when testing is on. With it off, every downstream OOT calling
# GR_ADD_CPP_TEST fails to configure:
#
#   Target "cuda_qa_seq.cc" links to: Boost::unit_test_framework
#   but the target was not found.
#
# gnuradio's own tests are built here but never run: this job tests gr-cuda,
# gnuradio's suite is not ours to police, and it does not pass headless.
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
