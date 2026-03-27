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

#include "ibv_common.h"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/ibv_source.h>

#include <infiniband/verbs.h>
#include <memory>
#include <string>

namespace gr {
namespace cuda {

class ibv_source_impl : public ibv_source, public cuda_block
{
private:
    // One slot holds a full raw Ethernet frame (jumbo)
    static constexpr int SLOT_SIZE = 9216;
    static constexpr int NUM_WR = 2048;
    static constexpr int CQ_SIZE = NUM_WR * 2;
    static constexpr int CQ_POLL_BATCH = 64;
    static constexpr size_t GPU_BUF_SIZE = 64UL * 1024 * 1024;

    int d_payload_size;
    std::string d_interface;
    std::string d_mcast_group;
    int d_udp_port;
    int d_gpu_id;

    // IBV transport (ctx, pd, cq, qp) and GPU landing buffer (MR)
    std::unique_ptr<ibv_transport> d_xport;
    std::unique_ptr<ibv_gpu_buffer> d_gpu_buf;
    struct ibv_flow* d_flow = nullptr;

    // WR/SGE pool -- reused for every batch post
    struct ibv_sge* d_sges = nullptr;
    struct ibv_recv_wr* d_wrs = nullptr;

    // Slot ring: d_next_slot cycles through all GPU_BUF_SIZE/SLOT_SIZE slots
    uint32_t d_num_slots;
    uint32_t d_next_slot = 0;

    /*
     * Ready ring: completed-but-not-yet-processed slot indices.
     * Fixed-size ring buffer (no heap allocations in work()).
     */
    uint32_t d_ready_ring[NUM_WR];
    int d_ready_head = 0;
    int d_ready_count = 0;

    // Pinned-host + device arrays for passing slot indices to the kernel
    uint32_t* d_slot_indices_host = nullptr;
    uint32_t* d_slot_indices_dev = nullptr;

    int d_igmp_sock = -1;

    void setup_multicast();
    void setup_flow_steering();
    void post_recv_batch(int count);

public:
    ibv_source_impl(const std::string& ibv_device,
                    const std::string& interface,
                    int udp_port,
                    int payload_size,
                    const std::string& mcast_group,
                    int gpu_id);
    ~ibv_source_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SOURCE_IMPL_H */
