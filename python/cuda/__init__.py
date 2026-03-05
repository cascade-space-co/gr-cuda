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
from .utils import as_cupy, io_signature_make
from .cuda_block import basic_block, decim_block, interp_block, sync_block

# import python blocks (these depend on cuda_block via cuda.sync_block)
from .add_cupy import add_cupy
from .fft_cupy import fft_cupy
from .complex_to_mag_cupy import complex_to_mag_cupy
from .complex_to_mag_squared_cupy import complex_to_mag_squared_cupy
from .multiply_const_cupy import multiply_const_cupy
