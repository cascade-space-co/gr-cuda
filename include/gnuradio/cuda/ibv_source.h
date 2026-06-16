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
 * \note If GPUDirect RDMA throughput is lower than expected, PCIe ACS
 * (Access Control Services) may be enabled on a bridge between the NIC
 * and the GPU.  Disable it with:
 * \code
 *   sudo setpci -s <bridge_bdf> ECAP_ACS+6.w=0000
 * \endcode
 * Run `sudo lspci -vvv | grep -i ACSCtl` to check.
 */
class CUDA_API ibv_source : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<ibv_source> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of cuda::ibv_source.
     *
     * \param ibv_device   IB device name as shown by `ibv_devices`
     *                     (e.g. "rocep119s0f0"); the associated Linux
     *                     netdev is derived automatically via sysfs for
     *                     IGMP multicast joins.
     * \param udp_port     UDP destination port to match in flow steering
     * \param payload_size Expected UDP payload size in bytes
     * \param mcast_group  Multicast group IP (e.g. "239.1.2.3"), empty for unicast
     * \param netdev       Linux netdev name (e.g. "ens10f1np1") to use for the
     *                     IGMP join.  Leave empty to auto-detect from the IB
     *                     device; required if that device exposes more than
     *                     one netdev (SR-IOV/switchdev), where auto-detect
     *                     cannot safely choose.
     */
    static sptr make(const std::string& ibv_device,
                     int udp_port,
                     int payload_size,
                     const std::string& mcast_group = "",
                     const std::string& netdev = "");
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SOURCE_H */
