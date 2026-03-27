/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_IBV_COMMON_H
#define INCLUDED_GR_CUDA_IBV_COMMON_H

#include <infiniband/verbs.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace gr {
namespace cuda {

/*!
 * \brief RAII wrapper for an IB transport: device context, PD, CQ, and QP.
 *
 * Opens the named IB device, allocates a protection domain, creates a
 * completion queue, creates a raw-Ethernet queue pair, and transitions
 * the QP through RESET -> INIT -> RTR (-> RTS if \c cfg.rts is set).
 * All resources are released in the destructor in reverse order.
 */
class ibv_transport
{
public:
    struct qp_config {
        int max_send_wr = 1;
        int max_send_sge = 1;
        int max_recv_wr = 1;
        int max_recv_sge = 1;
        int cq_size = 1024;
        bool rts = false;
    };

    ibv_transport(const std::string& device_name, const qp_config& cfg);
    ~ibv_transport();

    ibv_transport(const ibv_transport&) = delete;
    ibv_transport& operator=(const ibv_transport&) = delete;

    struct ibv_context* ctx() const { return d_ctx; }
    struct ibv_pd* pd() const { return d_pd; }
    struct ibv_cq* cq() const { return d_cq; }
    struct ibv_qp* qp() const { return d_qp; }

private:
    struct ibv_context* d_ctx = nullptr;
    struct ibv_pd* d_pd = nullptr;
    struct ibv_cq* d_cq = nullptr;
    struct ibv_qp* d_qp = nullptr;
};

/*!
 * \brief RAII wrapper for a GPU buffer registered as an IB memory region.
 *
 * Allocates GPU memory via cudaMalloc (dmabuf / GPUDirect RDMA path) or
 * cudaHostAlloc (unified-addressing fallback), then registers the buffer
 * with the given protection domain.  The destructor deregisters the MR
 * and frees the memory.
 *
 * \note The caller must call cudaSetDevice() before constructing this object.
 */
class ibv_gpu_buffer
{
public:
    ibv_gpu_buffer(int gpu_id, size_t size, struct ibv_pd* pd);
    ~ibv_gpu_buffer();

    ibv_gpu_buffer(const ibv_gpu_buffer&) = delete;
    ibv_gpu_buffer& operator=(const ibv_gpu_buffer&) = delete;

    uint8_t* data() const { return d_buf; }
    size_t size() const { return d_size; }
    struct ibv_mr* mr() const { return d_mr; }
    bool is_dmabuf() const { return d_mode == DMABUF; }

private:
    uint8_t* d_buf = nullptr;
    size_t d_size = 0;
    int d_dmabuf_fd = -1;
    struct ibv_mr* d_mr = nullptr;
    enum { DMABUF, UNIFIED } d_mode = DMABUF;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_GR_CUDA_IBV_COMMON_H */
