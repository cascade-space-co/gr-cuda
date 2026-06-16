/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_IBV_SINK_IMPL_H
#define INCLUDED_CUDA_IBV_SINK_IMPL_H

#include "network/ibv_common.h"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/ibv_sink.h>

#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace cuda {

class ibv_sink_impl : public ibv_sink, public cuda_block
{
private:
    // Maximum L2 (Ethernet) frame, in bytes, that one NIC slot must hold.
    // 9216 (= 9 * 1024) is the de-facto max "9K jumbo" frame size supported
    // by typical datacenter NICs (e.g. ConnectX); it holds a 9000-byte jumbo
    // IP MTU plus the 42-byte Eth/IP/UDP headers.  Jumbo frames are not part
    // of IEEE 802.3, so this is a common vendor ceiling, not a formal
    // standard.  Hard limit on static buffer geometry (not a tuning knob),
    // so it stays compile-time.
    static constexpr int MAX_SLOT_SIZE = 9216;

    // Compile-time defaults; the runtime values (d_*) are populated in
    // the constructor from gr::prefs and may override these.  Override
    // in the GR user prefs (path: `gnuradio-config-info --userprefsdir`).
    // See docs/OPTIMIZATIONS.md for what each knob does and how to tune it.
    //
    //   [ibv_sink]
    //   num_wr          = 4096
    //   signal_batch    = 512
    //   cq_size         = 8192     ; default is 2 * num_wr
    //   cq_poll_batch   = 64
    //   gpu_buf_bytes   = 67108864 ; 64 MiB
    static constexpr int DEFAULT_NUM_WR = 4096;
    static constexpr int DEFAULT_SIGNAL_BATCH = 512;
    static constexpr int DEFAULT_CQ_SIZE = DEFAULT_NUM_WR * 2;
    static constexpr int DEFAULT_CQ_POLL_BATCH = 64;
    static constexpr size_t DEFAULT_GPU_BUF_BYTES = 64UL * 1024 * 1024;

    // Runtime-resolved tuning knobs (read from gr::prefs in the ctor).
    int d_num_wr;
    int d_signal_batch;
    int d_cq_size;
    int d_cq_poll_batch;
    size_t d_gpu_buf_size;

    int d_payload_size;
    std::string d_dst_ip;
    int d_dst_port;
    std::string d_dst_mac;
    std::string d_mcast_group;
    int d_src_port;
    // Resolved Linux netdev (user-supplied, or auto-detected from the IB
    // device) whose MAC/IP populate the frame header's source fields.
    std::string d_netdev;

    int d_frame_size;
    int d_slot_size;

    // IBV transport (ctx, pd, cq, qp) and GPU landing buffer (MR)
    std::unique_ptr<ibv_transport> d_xport;
    std::unique_ptr<ibv_gpu_buffer> d_gpu_buf;

    // Pre-allocated WR/SGE pool (sized to d_num_wr)
    struct ibv_sge* d_sges = nullptr;
    struct ibv_send_wr* d_wrs = nullptr;

    // Reusable scratch buffer for ibv_poll_cq() (sized to d_cq_poll_batch)
    std::vector<struct ibv_wc> d_wc_pool;

    // 42-byte Eth/IP/UDP header template on the GPU
    uint8_t* d_header_template = nullptr;

    // Pipeline tracking
    int d_outstanding = 0;
    uint64_t d_wr_counter = 0;
    uint32_t d_num_slots;
    uint32_t d_next_slot = 0;

    void build_header();
    void drain_cq();

public:
    ibv_sink_impl(const std::string& ibv_device,
                  const std::string& dst_ip,
                  int dst_port,
                  int payload_size,
                  const std::string& dst_mac,
                  const std::string& mcast_group,
                  int src_port,
                  const std::string& netdev);
    ~ibv_sink_impl() override;

    bool start() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SINK_IMPL_H */
