#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from typing import List, Tuple, Union

import numpy as np
import cupy as cp
from gnuradio import gr
from gnuradio.gr.gateway import py_io_signature as BasePyIOSignature

from .cuda_python import cuda_buffer

# Accepted dtype arguments:
# a single numpy dtype/type, or a list/tuple of them.
DtypeSpec = Union[np.dtype, type, List, Tuple]


def as_cupy(numpy_array: np.ndarray) -> cp.ndarray:
    """
    Wrap a numpy array backed by CUDA memory into a CuPy array (zero-copy).

    The numpy array's data pointer must be a device pointer (e.g. from a
    cuda_buffer). This is validated at runtime.

    Parameters
    ----------
    numpy_array : np.ndarray
        A numpy array whose underlying data lives in CUDA device memory.

    Returns
    -------
    cp.ndarray
        A CuPy array sharing the same device memory.

    Raises
    ------
    ValueError
        If the pointer is not a valid CUDA device or managed pointer.
    """
    ptr, _ = numpy_array.__array_interface__['data']

    try:
        attrs = cp.cuda.runtime.pointerGetAttributes(ptr)
        # attrs.type: 0=Unregistered Host, 1=Host, 2=Device, 3=Managed
        if attrs.type != 2 and attrs.type != 3:
            raise ValueError(f"Pointer is not a device pointer (type={attrs.type})")
    except cp.cuda.runtime.CUDARuntimeError:
        raise ValueError("Pointer is not a valid CUDA pointer")

    mem = cp.cuda.UnownedMemory(ptr, numpy_array.nbytes, owner=numpy_array)
    mptr = cp.cuda.MemoryPointer(mem, 0)
    
    # Create a cupy array with the same shape/dtype
    return cp.ndarray(
        numpy_array.shape,
        dtype=numpy_array.dtype,
        memptr=mptr,
    )


def io_signature_make(
    min_ports: int,
    max_ports: int,
    dtype: DtypeSpec,
) -> BasePyIOSignature:
    """
    Create a CUDA IO signature for use in Python blocks.

    Parameters
    ----------
    min_ports : int
        Minimum number of connected ports.
    max_ports : int
        Maximum number of connected ports.
    dtype : numpy.dtype, type, list, or tuple
        Data type(s) for the ports. A single type applies to all ports;
        a list of types corresponds to each port individually.

    Returns
    -------
    BasePyIOSignature
        An IO signature compatible with ``gr.gateway`` that ensures
        buffers are allocated as CUDA buffers.
    """
    if isinstance(dtype, (list, tuple)):
        type_list = dtype
    else:
        type_list = [dtype]
        
    # Create the base object which gateway.py expects
    sig = BasePyIOSignature(min_ports, max_ports, type_list)
    
    # Calculate item sizes
    sizes = [np.dtype(t).itemsize for t in type_list]
    if not sizes:
        sizes = [0]

    buftypes = [cuda_buffer.type] * len(sizes)

    # Monkey-patch gr_io_signature on this instance so that gr.gateway
    # constructs the C++ io_signature with CUDA buffer types. There is
    # currently no public API in GR to achieve this without patching.
    def custom_gr_io_signature():
        return gr.io_signature(min_ports, max_ports, sizes, buftypes)

    sig.gr_io_signature = custom_gr_io_signature

    return sig
