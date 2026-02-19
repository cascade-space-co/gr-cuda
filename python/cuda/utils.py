#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np

import cupy as cp

from .cuda_python import cuda_buffer

def as_cupy(numpy_array):
    """
    Wrap a numpy array backed by CUDA memory into a CuPy array (zero-copy).
    
    This function assumes the numpy array's data pointer is actually a device pointer
    (e.g. from a cuda_buffer).
    """
    
    # Get the data pointer and size from the numpy array
    # __array_interface__['data'] returns (ptr, read_only)
    ptr, _ = numpy_array.__array_interface__['data']
    
    # Check if pointer is actually a device pointer
    try:
        attrs = cp.cuda.runtime.pointerGetAttributes(ptr)
        # attrs.type: 0=Unregistered Host, 1=Host, 2=Device, 3=Managed
        if attrs.type != 2 and attrs.type != 3:
             raise RuntimeError(f"Pointer is not a device pointer (type={attrs.type})")
    except cp.cuda.runtime.CUDARuntimeError:
         raise RuntimeError("Pointer is not a valid CUDA pointer")
        
    size = numpy_array.nbytes
    
    # Create an UnownedMemory object
    # We pass numpy_array as owner to keep the underlying buffer alive if needed
    mem = cp.cuda.UnownedMemory(ptr, size, owner=numpy_array)
    
    # Create a MemoryPointer
    mptr = cp.cuda.MemoryPointer(mem, 0)
    
    # Create a cupy array with the same shape/dtype
    return cp.ndarray(
        numpy_array.shape, 
        dtype=numpy_array.dtype, 
        memptr=mptr
    )

def io_signature_make(min_ports, max_ports, dtype):
    """
    Create a CUDA IO signature for use in Python blocks.
    
    Args:
        min_ports (int): Minimum number of connected ports
        max_ports (int): Maximum number of connected ports
        dtype (numpy.dtype, list, or tuple): Data type(s) for the ports.
            Can be a single type (e.g. np.complex64) which applies to all ports,
            or a list of types corresponding to each port.
            
    Returns:
        A special IO signature object compatible with gr.gateway that ensures
        buffers are allocated as CUDA buffers.
    """
    from gnuradio import gr
    from gnuradio.gr.gateway import py_io_signature as BasePyIOSignature
    
    # Handle single type vs list of types
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
        
    # Define the closure that returns the C++ io_signature with CUDA buffer type
    def custom_gr_io_signature():
        # We need to construct the C++ io_signature with the correct buffer type
        # The constructor taking (min, max, sizes_list, buftypes_list) requires
        # the lists to match or be compatible.
        
        # Note: cuda_buffer.type is available in this namespace (imported from .cuda_python)
        # We need a list of buffer types matching the sizes
        buftypes = [cuda_buffer.type] * len(sizes)
        
        return gr.io_signature(min_ports, max_ports, sizes, buftypes)
        
    # Monkey-patch the method on this instance
    sig.gr_io_signature = custom_gr_io_signature
    
    return sig

