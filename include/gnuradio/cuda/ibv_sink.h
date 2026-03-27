/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_IBV_SINK_H
#define INCLUDED_CUDA_IBV_SINK_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <string>

namespace gr {
namespace cuda {

/*!
 * \brief Zero-copy GPU-to-NIC sink using libibverbs raw Ethernet
 * \ingroup cuda
 *
 * Sends UDP packets directly from GPU memory using a raw Ethernet QP.
 * A CUDA kernel prepends 42-byte Ethernet/IP/UDP headers to payload
 * chunks from the input cuda_buffer, building complete frames in a GPU
 * landing buffer (registered via dmabuf or unified-memory MR).  The NIC
 * DMAs them out with zero CPU-side data copies.
 *
 * Supports unicast and multicast (destination MAC auto-derived from
 * the multicast group IP).
 *
 * Optimized for line-rate: batched kernel launch, chained WR posting
 * (64 per ibv_post_send), 4096-deep send pipeline.
 *
 * \section prereqs Prerequisites
 *   - ConnectX-7 (or later) in Ethernet mode, link up
 *   - CUDA 11.7+ with dmabuf support, or Grace Blackwell unified memory
 *   - rdma-core / MLNX_OFED
 *   - PCIe ACS disabled on switches between NIC and GPU
 *   - CAP_NET_RAW (run flowgraph as root or with appropriate capability)
 */
class CUDA_API ibv_sink : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<ibv_sink> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::ibv_sink.
     *
     * \param ibv_device   IB device name (e.g. "rocep119s0f0")
     * \param interface    Network interface (e.g. "ens10f0np0") for src MAC/IP
     * \param dst_ip       Destination IP address (ignored if mcast_group is set)
     * \param dst_port     Destination UDP port
     * \param payload_size Payload size per packet in bytes
     * \param dst_mac      Destination MAC "xx:xx:xx:xx:xx:xx" (ignored if mcast_group
     * is set) \param mcast_group  Multicast group IP (e.g. "239.1.2.3"), empty for
     * unicast \param gpu_id       CUDA GPU device ID (default: 0)
     */
    static sptr make(const std::string& ibv_device,
                     const std::string& interface,
                     const std::string& dst_ip,
                     int dst_port,
                     int payload_size,
                     const std::string& dst_mac = "",
                     const std::string& mcast_group = "",
                     int gpu_id = 0);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SINK_H */
