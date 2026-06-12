/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ibv_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <dirent.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gr {
namespace cuda {

// ibv_transport

ibv_transport::ibv_transport(const std::string& device_name, const qp_config& cfg)
{
    // Open IB device
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

    // Protection domain
    d_pd = ibv_alloc_pd(d_ctx);
    if (!d_pd)
        throw std::runtime_error("ibv_transport: ibv_alloc_pd: " +
                                 std::string(strerror(errno)));

    // Optional completion channel: lets the CQ deliver interrupt-driven
    // events on a file descriptor, so a consumer can block instead of
    // busy-polling.  Created before the CQ because the CQ binds to it.
    if (cfg.use_comp_channel) {
        d_comp_channel = ibv_create_comp_channel(d_ctx);
        if (!d_comp_channel)
            throw std::runtime_error("ibv_transport: ibv_create_comp_channel: " +
                                     std::string(strerror(errno)));
    }

    // Completion queue (bound to the completion channel when present)
    d_cq = ibv_create_cq(d_ctx, cfg.cq_size, nullptr, d_comp_channel, 0);
    if (!d_cq)
        throw std::runtime_error("ibv_transport: ibv_create_cq: " +
                                 std::string(strerror(errno)));

    // Queue pair (raw Ethernet)
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

    // QP state transitions: RESET -> INIT -> RTR [-> RTS]
    struct ibv_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE | IBV_QP_PORT))
        throw std::runtime_error("ibv_transport: RESET->INIT: " +
                                 std::string(strerror(errno)));

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE))
        throw std::runtime_error("ibv_transport: INIT->RTR: " +
                                 std::string(strerror(errno)));

    if (cfg.rts) {
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        if (ibv_modify_qp(d_qp, &attr, IBV_QP_STATE))
            throw std::runtime_error("ibv_transport: RTR->RTS: " +
                                     std::string(strerror(errno)));
    }
}

std::string ibv_transport::netdev_name() const
{
    std::string net_dir = std::string(d_ctx->device->ibdev_path) + "/device/net";
    DIR* dir = opendir(net_dir.c_str());
    if (!dir)
        throw std::runtime_error("ibv_transport::netdev_name: cannot open " + net_dir);

    // Collect all netdevs bound to this IB device's PCI function.  There can
    // be more than one, and readdir() order is unspecified, so we must not
    // just take the first entry, which would non-deterministically pick a
    // possibly-wrong netdev.
    std::vector<std::string> netdevs;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.')
            continue;
        netdevs.emplace_back(entry->d_name);
    }
    closedir(dir);

    const std::string dev_name(d_ctx->device->name);
    if (netdevs.empty())
        throw std::runtime_error("ibv_transport::netdev_name: no netdev found for " +
                                 dev_name);
    if (netdevs.size() > 1) {
        // Sort for a stable message, then refuse to guess: the caller should
        // pass an explicit netdev name (ibv_sink/ibv_source expose a param).
        std::sort(netdevs.begin(), netdevs.end());
        std::string list;
        for (size_t i = 0; i < netdevs.size(); i++)
            list += (i ? ", " : "") + netdevs[i];
        throw std::runtime_error(
            "ibv_transport::netdev_name: " + dev_name + " has multiple netdevs (" +
            list + "); specify one explicitly via the block's netdev parameter");
    }
    return netdevs.front();
}

ibv_transport::~ibv_transport()
{
    if (d_qp)
        ibv_destroy_qp(d_qp);
    if (d_cq)
        ibv_destroy_cq(d_cq);
    if (d_comp_channel)
        ibv_destroy_comp_channel(d_comp_channel);
    if (d_pd)
        ibv_dealloc_pd(d_pd);
    if (d_ctx)
        ibv_close_device(d_ctx);
}

// ibv_gpu_buffer

ibv_gpu_buffer::ibv_gpu_buffer(size_t size, struct ibv_pd* pd) : d_size(size)
{
    int device;
    cudaGetDevice(&device);
    CUdevice cu_dev;
    cuDeviceGet(&cu_dev, device);

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
