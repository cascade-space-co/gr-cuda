#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import os

# The cuda_buffer settings the tests run with under pytest, which imports every
# test module into one process; cuda_buffer reads them once, at its first
# allocation. Keep in sync with GR_TEST_ENVIRONS in the top-level CMakeLists.txt.
os.environ.setdefault("GR_CONF_CUDA_BUFFER_MIN_BUFFER_BYTES", str(2 << 20))
os.environ.setdefault("GR_CONF_CUDA_BUFFER_GPU_BUSY_WAIT", "True")
