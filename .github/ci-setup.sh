#!/usr/bin/env bash
#
# Build the conda environment gpu-qa.yml tests in: dependencies, then GNU Radio
# from source.
#
# This is a CI artifact, not a developer environment. It exists as one file
# because the cache key hashes it -- what gets cached is the environment this
# script produces, so any change to it must invalidate the cache.
#
# GNU Radio is built from source because gr-cuda needs APIs that no released
# version has. conda-forge's newest is 3.10.12.0, installed here only to pull in
# the build and runtime dependency closure, then removed before the source build
# replaces it.
#
# Needs ENV_PATH and GNURADIO_REF.

set -euo pipefail

: "${ENV_PATH:?}"
: "${GNURADIO_REF:?}"

source /opt/miniforge/etc/profile.d/conda.sh

mamba create -y -p "$ENV_PATH" -c conda-forge --override-channels \
    'python>=3.12' \
    'gnuradio-build-deps=3.10.12' \
    'pygccxml>=3.0.2,<4.0' \
    'cuda-nvcc<13.1' \
    'cuda-cudart-dev<13.1' \
    'cuda-driver-dev<13.1' \
    libcufft-dev \
    cupy \
    pytest \
    pytest-timeout

# --no-prune-deps keeps the dependency closure that was the point of installing
# gnuradio in the first place.
pkgs="$(mamba list -p "$ENV_PATH" | awk '$1 ~ /^gnuradio/ {print $1}')"
if [ -n "$pkgs" ]; then
    # shellcheck disable=SC2086
    mamba remove -p "$ENV_PATH" -y --no-prune-deps $pkgs
fi

# conda's activation scripts reference unset variables, so -u has to come off
# around them (cuda-nvcc's touches NVCC_PREPEND_FLAGS).
set +u
conda activate "$ENV_PATH"
set -u

src="${RUNNER_TEMP:-/tmp}/gnuradio-src"
rm -rf "$src"
git init -q "$src"
git -C "$src" remote add origin https://github.com/gnuradio/gnuradio.git
git -C "$src" fetch -q --depth 1 origin "$GNURADIO_REF"
git -C "$src" checkout -q FETCH_HEAD

# gr-cuda links only gnuradio-runtime and imports only gr, blocks, fft and
# gr_unittest, so the GUI, hardware and docs components come out. ENABLE_TESTING
# must stay on: GnuradioConfig.cmake only requests Boost's unit_test_framework
# when it is, and without that every OOT calling GR_ADD_CPP_TEST fails to
# configure. GNU Radio's own tests are built here, never run.
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
