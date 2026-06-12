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
 * (512 per ibv_post_send), 4096-deep send pipeline.
 *
 * \note If GPUDirect RDMA throughput is lower than expected, PCIe ACS
 * (Access Control Services) may be enabled on a bridge between the NIC
 * and the GPU.  Disable it with:
 * \code
 *   sudo setpci -s <bridge_bdf> ECAP_ACS+6.w=0000
 * \endcode
 * Run `sudo lspci -vvv | grep -i ACSCtl` to check.
 */
class CUDA_API ibv_sink : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<ibv_sink> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::ibv_sink.
     *
     * \param ibv_device   IB device name as shown by `ibv_devices`
     *                     (e.g. "rocep119s0f0"); the associated Linux
     *                     netdev is derived automatically via sysfs for
     *                     source MAC/IP.
     * \param dst_ip       Destination IP address (ignored if mcast_group is set)
     * \param dst_port     Destination UDP port
     * \param payload_size Payload size per packet in bytes
     * \param dst_mac      Destination MAC "xx:xx:xx:xx:xx:xx" (ignored if
     *                     mcast_group is set)
     * \param mcast_group  Multicast group IP (e.g. "239.1.2.3"), empty for
     *                     unicast
     * \param src_port     Source UDP port written into each frame's UDP
     *                     header.  This is metadata only (raw Ethernet QP,
     *                     no socket/bind); useful for receiver-side flow
     *                     filtering or NIC RSS hashing.
     */
    static sptr make(const std::string& ibv_device,
                     const std::string& dst_ip,
                     int dst_port,
                     int payload_size,
                     const std::string& dst_mac = "",
                     const std::string& mcast_group = "",
                     int src_port = 12345);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SINK_H */
