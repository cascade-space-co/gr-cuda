/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_IBV_SOURCE_IMPL_H
#define INCLUDED_CUDA_IBV_SOURCE_IMPL_H

#include "network/ibv_common.h"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/ibv_source.h>

#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace cuda {

class ibv_source_impl : public ibv_source, public cuda_block
{
private:
    // One slot holds a full raw Ethernet frame (jumbo).  This is a hard
    // frame-size limit (used in static buffer geometry), not a tuning
    // knob, so it stays compile-time.
    static constexpr int SLOT_SIZE = 9216;

    // Compile-time defaults; the runtime values (d_*) are populated in
    // the constructor from gr::prefs and may override these.  Override
    // in the GR user prefs (path: `gnuradio-config-info --userprefsdir`).
    // See docs/OPTIMIZATIONS.md for what each knob does and how to tune it.
    //
    //   [ibv_source]
    //   num_wr          = 4096
    //   cq_size         = 8192     ; default is 2 * num_wr
    //   cq_poll_batch   = 512
    //   gpu_buf_bytes   = 67108864 ; 64 MiB
    static constexpr int DEFAULT_NUM_WR = 4096;
    static constexpr int DEFAULT_CQ_SIZE = DEFAULT_NUM_WR * 2;
    static constexpr int DEFAULT_CQ_POLL_BATCH = 512;
    static constexpr size_t DEFAULT_GPU_BUF_BYTES = 64UL * 1024 * 1024;

    // Runtime-resolved tuning knobs (read from gr::prefs in the ctor).
    int d_num_wr;
    int d_cq_size;
    int d_cq_poll_batch;
    size_t d_gpu_buf_size;

    int d_payload_size;
    std::string d_mcast_group;
    int d_udp_port;
    // Resolved Linux netdev (user-supplied, or auto-detected from the IB
    // device) used for the IGMP multicast join.
    std::string d_netdev;

    // IBV transport (ctx, pd, cq, qp) and GPU landing buffer (MR)
    std::unique_ptr<ibv_transport> d_xport;
    std::unique_ptr<ibv_gpu_buffer> d_gpu_buf;
    struct ibv_flow* d_flow = nullptr;

    // WR/SGE pool -- reused for every batch post (sized to d_num_wr)
    struct ibv_sge* d_sges = nullptr;
    struct ibv_recv_wr* d_wrs = nullptr;

    // Reusable scratch buffer for ibv_poll_cq() (sized to d_cq_poll_batch)
    std::vector<struct ibv_wc> d_wc_pool;

    uint32_t d_num_slots;
    uint32_t d_next_slot = 0;

    // Sequential completion tracking: slots complete in posting order.
    uint32_t d_ready_slot = 0;
    int d_ready_count = 0;

    int d_igmp_sock = -1;

    // Completion-channel fd (from d_xport->comp_channel()), cached so the
    // receive path can block on it instead of busy-polling when idle.
    int d_comp_channel_fd = -1;

    // Latches once we've logged an ibv_poll_cq() failure, so a persistently
    // broken CQ doesn't spam the log on every work() call.
    bool d_poll_cq_err_logged = false;

    void setup_multicast();
    void setup_flow_steering();
    void post_recv_batch(int count);

    // Drain ready completions from the CQ into d_ready_count.  Returns the
    // number harvested (>= 0), or -1 on poll error.
    int poll_cq();

    // Block (up to timeout_ms) on the completion channel until the CQ has a
    // new completion, then drain it.  Used only when the link is idle so we
    // give the CPU back instead of spinning.  Returns true if any completion
    // was harvested.
    bool wait_for_completion(int timeout_ms);

public:
    ibv_source_impl(const std::string& ibv_device,
                    int udp_port,
                    int payload_size,
                    const std::string& mcast_group,
                    const std::string& netdev);
    ~ibv_source_impl() override;

    bool start() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SOURCE_IMPL_H */
