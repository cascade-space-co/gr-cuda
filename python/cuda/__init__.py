#
# Copyright 2008,2009 Free Software Foundation, Inc.
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# The presence of this file turns this directory into a Python package

"""
This is the GNU Radio CUDA module. Place your Python package
description here (python/__init__.py).
"""

# import pybind11 generated symbols into the cuda namespace
try:
    # this might fail if the module is python-only
    from .cuda_python import *
except ModuleNotFoundError:
    pass

# import python helpers
from .add_cupy import add_cupy
from .cuda_block import basic_block, decim_block, interp_block, sync_block
from .fft_cupy import fft_cupy

# import python blocks
from .multiply_const_cupy import multiply_const_cupy
from .utils import as_cupy, io_signature_make
