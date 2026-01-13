#include <stdio.h>
#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Use char* to handle arbitrary item sizes
// N = number of vectors to process
// itemsize = bytes per scalar item
// vlen = number of streams (or vector length)

__global__ void kernel_interleave(const void** inputs, 
                                  char* out, 
                                  int num_streams, 
                                  int itemsize, 
                                  int N)
{
    // N = number of output vectors
    // One thread per output vector? Or one thread per scalar?
    // One thread per scalar is better parallelism.
    // Total scalars = N * num_streams
    
    int total_scalars = N * num_streams;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_scalars) {
        // Output layout: [S0_0, S1_0, S2_0], [S0_1, S1_1, S2_1], ...
        // idx maps to specific output scalar slot.
        
        // Which vector index (time step)?
        int vec_idx = idx / num_streams;
        // Which stream index?
        int stream_idx = idx % num_streams;
        
        // Input layout: 
        // Stream 0: [S0_0, S0_1, ...]
        // Stream 1: [S1_0, S1_1, ...]
        
        const char* in_ptr = (const char*)inputs[stream_idx];
        
        // Copy itemsize bytes
        // Doing byte-wise copy inside a thread for 'itemsize' might be slow if itemsize is large.
        // But usually it's 4 or 8 bytes.
        // For arbitrary itemsize, we loop.
        
        int in_offset = vec_idx * itemsize;
        int out_offset = idx * itemsize;
        
        for (int b = 0; b < itemsize; b++) {
            out[out_offset + b] = in_ptr[in_offset + b];
        }
    }
}

__global__ void kernel_deinterleave(const char* in, 
                                    void** outputs, 
                                    int num_streams, 
                                    int itemsize, 
                                    int N)
{
    // N = number of input vectors
    int total_scalars = N * num_streams;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_scalars) {
        // idx corresponds to scalar index in the INPUT buffer (interleaved)
        // [S0_0, S1_0], [S0_1, S1_1] ...
        
        int vec_idx = idx / num_streams;
        int stream_idx = idx % num_streams;
        
        char* out_ptr = (char*)outputs[stream_idx];
        
        int in_offset = idx * itemsize;
        int out_offset = vec_idx * itemsize;
        
        for (int b = 0; b < itemsize; b++) {
            out_ptr[out_offset + b] = in[in_offset + b];
        }
    }
}

void exec_interleave(const void** inputs,
                     void* out,
                     int num_streams,
                     int itemsize,
                     int N, // number of vectors
                     int grid_size,
                     int block_size,
                     cudaStream_t stream)
{
    kernel_interleave<<<grid_size, block_size, 0, stream>>>(inputs, (char*)out, num_streams, itemsize, N);
    check_cuda_errors(cudaGetLastError());
}

void exec_deinterleave(const void* in,
                       void** outputs,
                       int num_streams,
                       int itemsize,
                       int N, // number of vectors
                       int grid_size,
                       int block_size,
                       cudaStream_t stream)
{
    kernel_deinterleave<<<grid_size, block_size, 0, stream>>>((const char*)in, outputs, num_streams, itemsize, N);
    check_cuda_errors(cudaGetLastError());
}

// Helpers to calculate grid size based on total SCALARS, not vectors
void get_interleave_block_and_grid(int* minGrid, int* minBlock)
{
    // We use a generic kernel that doesn't depend on T, just char*
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_interleave, 0, 0));
}

void get_deinterleave_block_and_grid(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_deinterleave, 0, 0));
}

