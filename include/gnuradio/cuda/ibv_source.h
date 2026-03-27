/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_IBV_SOURCE_H
#define INCLUDED_CUDA_IBV_SOURCE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <string>

namespace gr {
namespace cuda {

/*!
 * \brief Zero-copy NIC-to-GPU source using libibverbs raw Ethernet
 * \ingroup cuda
 *
 * Receives UDP packets directly into GPU memory using a raw Ethernet QP
 * with hardware flow steering.  The NIC DMAs packets into a GPU landing
 * buffer (registered via dmabuf or unified-memory MR); a CUDA kernel
 * strips the 42-byte Ethernet/IP/UDP header and copies payloads
 * contiguously into the output cuda_buffer.
 *
 * Supports unicast and multicast (IGMP join handled automatically).
 *
 * \section prereqs Prerequisites
 *   - ConnectX-7 (or later) in Ethernet mode, link up
 *   - CUDA 11.7+ with dmabuf support, or Grace Blackwell unified memory
 *   - rdma-core / MLNX_OFED
 *   - PCIe ACS disabled on switches between NIC and GPU
 *   - CAP_NET_RAW (run flowgraph as root or with appropriate capability)
 */
class CUDA_API ibv_source : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<ibv_source> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::ibv_source.
     *
     * \param ibv_device   IB device name (e.g. "rocep119s0f0")
     * \param interface    Network interface (e.g. "ens10f0np0") for IGMP join
     * \param udp_port     UDP destination port to match in flow steering
     * \param payload_size Expected UDP payload size in bytes
     * \param mcast_group  Multicast group IP (e.g. "239.1.2.3"), empty for unicast
     * \param gpu_id       CUDA GPU device ID (default: 0)
     */
    static sptr make(const std::string& ibv_device,
                     const std::string& interface,
                     int udp_port,
                     int payload_size,
                     const std::string& mcast_group = "",
                     int gpu_id = 0);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SOURCE_H */
