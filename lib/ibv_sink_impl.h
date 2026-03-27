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

#include "ibv_common.h"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/ibv_sink.h>

#include <infiniband/verbs.h>
#include <memory>
#include <string>

namespace gr {
namespace cuda {

class ibv_sink_impl : public ibv_sink, public cuda_block
{
private:
    static constexpr int MAX_SLOT_SIZE = 9216;
    static constexpr int NUM_WR = 4096;
    static constexpr int SIGNAL_BATCH = 64;
    static constexpr int CQ_SIZE = NUM_WR * 2;
    static constexpr int CQ_POLL_BATCH = 64;
    static constexpr size_t GPU_BUF_SIZE = 64UL * 1024 * 1024;

    int d_payload_size;
    std::string d_interface;
    std::string d_dst_ip;
    int d_dst_port;
    std::string d_dst_mac;
    std::string d_mcast_group;
    int d_gpu_id;

    int d_frame_size;
    int d_slot_size;

    // IBV transport (ctx, pd, cq, qp) and GPU landing buffer (MR)
    std::unique_ptr<ibv_transport> d_xport;
    std::unique_ptr<ibv_gpu_buffer> d_gpu_buf;

    // Pre-allocated WR/SGE pool
    struct ibv_sge* d_sges = nullptr;
    struct ibv_send_wr* d_wrs = nullptr;

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
                  const std::string& interface,
                  const std::string& dst_ip,
                  int dst_port,
                  int payload_size,
                  const std::string& dst_mac,
                  const std::string& mcast_group,
                  int gpu_id);
    ~ibv_sink_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_IBV_SINK_IMPL_H */
