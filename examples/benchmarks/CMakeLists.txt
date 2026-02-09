install(
    PROGRAMS
        benchmark_gr_cuda_transfer.py
        benchmark_gr_cuda_fft.py
    DESTINATION ${GR_PKG_DATA_DIR}/examples/benchmarks
)
