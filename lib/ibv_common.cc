/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ibv_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace cuda {

// ibv_transport

ibv_transport::ibv_transport(const std::string& device_name, const qp_config& cfg)
{
    /* ── Open IB device ── */
    int num_devices = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0)
        throw std::runtime_error("ibv_transport: no IB devices found");

    struct ibv_device* ib_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        if (device_name == ibv_get_device_name(dev_list[i])) {
            ib_dev = dev_list[i];
            break;
        }
    }
    if (!ib_dev) {
        ibv_free_device_list(dev_list);
        throw std::runtime_error("ibv_transport: device '" + device_name +
                                 "' not found");
    }

    d_ctx = ibv_open_device(ib_dev);
    ibv_free_device_list(dev_list);
    if (!d_ctx)
        throw std::runtime_error("ibv_transport: ibv_open_device: " +
                                 std::string(strerror(errno)));

    /* ── Protection domain ── */
    d_pd = ibv_alloc_pd(d_ctx);
    if (!d_pd)
        throw std::runtime_error("ibv_transport: ibv_alloc_pd: " +
                                 std::string(strerror(errno)));

    /* ── Completion queue ── */
    d_cq = ibv_create_cq(d_ctx, cfg.cq_size, nullptr, nullptr, 0);
    if (!d_cq)
        throw std::runtime_error("ibv_transport: ibv_create_cq: " +
                                 std::string(strerror(errno)));

    /* ── Queue pair (raw Ethernet) ── */
    struct ibv_qp_init_attr qp_init;
    memset(&qp_init, 0, sizeof(qp_init));
    qp_init.qp_type = IBV_QPT_RAW_PACKET;
    qp_init.send_cq = d_cq;
    qp_init.recv_cq = d_cq;
    qp_init.cap.max_send_wr = cfg.max_send_wr;
    qp_init.cap.max_send_sge = cfg.max_send_sge;
    qp_init.cap.max_recv_wr = cfg.max_recv_wr;
    qp_init.cap.max_recv_sge = cfg.max_recv_sge;

    d_qp = ibv_create_qp(d_pd, &qp_init);
    if (!d_qp)
        throw std::runtime_error("ibv_transport: ibv_create_qp: " +
                                 std::string(strerror(errno)));

    /* ── QP state transitions: RESET → INIT → RTR [→ RTS] ── */
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE | IBV_QP_PORT))
        throw std::runtime_error("ibv_transport: RESET→INIT: " +
                                 std::string(strerror(errno)));

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE))
        throw std::runtime_error("ibv_transport: INIT→RTR: " +
                                 std::string(strerror(errno)));

    if (cfg.rts) {
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE))
            throw std::runtime_error("ibv_transport: RTR→RTS: " +
                                     std::string(strerror(errno)));
    }
}

ibv_transport::~ibv_transport()
{
    if (d_qp)
        ibv_destroy_qp(d_qp);
    if (d_cq)
        ibv_destroy_cq(d_cq);
    if (d_pd)
        ibv_dealloc_pd(d_pd);
    if (d_ctx)
        ibv_close_device(d_ctx);
}

// ─────────────────────────────────────────────────────────────────────
// ibv_gpu_buffer
// ─────────────────────────────────────────────────────────────────────

ibv_gpu_buffer::ibv_gpu_buffer(int gpu_id, size_t size, struct ibv_pd* pd)
    : d_size(size)
{
    CUdevice cu_dev;
    cuDeviceGet(&cu_dev, gpu_id);

    int dmabuf_supported = 0;
    cuDeviceGetAttribute(
        &dmabuf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cu_dev);
    int unified_addressing = 0;
    cuDeviceGetAttribute(
        &unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cu_dev);

    if (dmabuf_supported) {
        d_mode = DMABUF;
        if (cudaMalloc(reinterpret_cast<void**>(&d_buf), size) != cudaSuccess)
            throw std::runtime_error("ibv_gpu_buffer: cudaMalloc failed");
        cudaMemset(d_buf, 0, size);

        CUresult err =
            cuMemGetHandleForAddressRange(reinterpret_cast<void*>(&d_dmabuf_fd),
                                          reinterpret_cast<CUdeviceptr>(d_buf),
                                          size,
                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                          0);
        if (err != CUDA_SUCCESS)
            throw std::runtime_error(
                "ibv_gpu_buffer: cuMemGetHandleForAddressRange failed");

        d_mr = ibv_reg_dmabuf_mr(pd,
                                 0,
                                 size,
                                 reinterpret_cast<uint64_t>(d_buf),
                                 d_dmabuf_fd,
                                 IBV_ACCESS_LOCAL_WRITE);
    } else if (unified_addressing) {
        d_mode = UNIFIED;
        if (cudaHostAlloc(reinterpret_cast<void**>(&d_buf),
                          size,
                          cudaHostAllocDefault) != cudaSuccess)
            throw std::runtime_error("ibv_gpu_buffer: cudaHostAlloc failed");
        memset(d_buf, 0, size);

        d_mr = ibv_reg_mr(pd, d_buf, size, IBV_ACCESS_LOCAL_WRITE);
    } else {
        throw std::runtime_error(
            "ibv_gpu_buffer: GPU supports neither dmabuf nor unified addressing");
    }

    if (!d_mr)
        throw std::runtime_error("ibv_gpu_buffer: MR registration failed: " +
                                 std::string(strerror(errno)));
}

ibv_gpu_buffer::~ibv_gpu_buffer()
{
    if (d_mr)
        ibv_dereg_mr(d_mr);
    if (d_dmabuf_fd >= 0)
        close(d_dmabuf_fd);
    if (d_buf) {
        if (d_mode == DMABUF)
            cudaFree(d_buf);
        else
            cudaFreeHost(d_buf);
    }
}

} // namespace cuda
} // namespace gr
