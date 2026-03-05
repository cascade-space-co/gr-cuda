# Using gr-cuda in your own OOT

## Root `CMakeLists.txt`

Enable CUDA in your project:

```cmake
project(gr-myoot CXX C CUDA)
```

## `lib/CMakeLists.txt`

```cmake
# Find gr-cuda
find_package(gnuradio-cuda REQUIRED)

# Compile your CUDA kernels
add_library(gnuradio-myoot-cu STATIC my_kernel.cu)
set_target_properties(gnuradio-myoot-cu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Link your OOT to gr-cuda and your kernels
target_link_libraries(gnuradio-myoot PUBLIC gnuradio::gnuradio-runtime gnuradio::gnuradio-cuda)
target_link_libraries(gnuradio-myoot PRIVATE gnuradio-myoot-cu)
```
