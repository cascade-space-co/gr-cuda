/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "ibv_source.cuh"
#include "ibv_source_impl.h"
#include "net_headers.h"
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>

#include <arpa/inet.h>
#include <endian.h>
#include <net/if.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>

namespace gr {
namespace cuda {

ibv_source::sptr ibv_source::make(const std::string& ibv_device,
                                  const std::string& interface,
                                  int udp_port,
                                  int payload_size,
                                  const std::string& mcast_group,
                                  int gpu_id)
{
    return gnuradio::make_block_sptr<ibv_source_impl>(
        ibv_device, interface, udp_port, payload_size, mcast_group, gpu_id);
}

ibv_source_impl::ibv_source_impl(const std::string& ibv_device,
                                 const std::string& interface,
                                 int udp_port,
                                 int payload_size,
                                 const std::string& mcast_group,
                                 int gpu_id)
    : sync_block("ibv_source",
                 io_signature::make(0, 0, 0),
                 io_signature::make(1, 1, sizeof(char), cuda_buffer::type)),
      d_payload_size(payload_size),
      d_interface(interface),
      d_mcast_group(mcast_group),
      d_udp_port(udp_port),
      d_gpu_id(gpu_id)
{
    set_output_multiple(d_payload_size);
    check_cuda_errors(cudaSetDevice(d_gpu_id), "ibv_source: cudaSetDevice", d_logger);

    ibv_transport::qp_config cfg;
    cfg.max_recv_wr = NUM_WR;
    cfg.max_recv_sge = 1;
    cfg.cq_size = CQ_SIZE;
    d_xport = std::make_unique<ibv_transport>(ibv_device, cfg);

    d_gpu_buf = std::make_unique<ibv_gpu_buffer>(d_gpu_id, GPU_BUF_SIZE, d_xport->pd());
    d_num_slots = GPU_BUF_SIZE / SLOT_SIZE;
    if (d_num_slots < static_cast<uint32_t>(NUM_WR))
        throw std::runtime_error("ibv_source: GPU_BUF_SIZE too small for NUM_WR slots");

    check_cuda_errors(cudaHostAlloc(reinterpret_cast<void**>(&d_slot_indices_host),
                                    NUM_WR * sizeof(uint32_t),
                                    cudaHostAllocDefault),
                      "ibv_source: cudaHostAlloc slot_indices",
                      d_logger);
    check_cuda_errors(cudaMalloc(reinterpret_cast<void**>(&d_slot_indices_dev),
                                 NUM_WR * sizeof(uint32_t)),
                      "ibv_source: cudaMalloc slot_indices_dev",
                      d_logger);

    d_sges = static_cast<struct ibv_sge*>(calloc(NUM_WR, sizeof(struct ibv_sge)));
    d_wrs =
        static_cast<struct ibv_recv_wr*>(calloc(NUM_WR, sizeof(struct ibv_recv_wr)));

    setup_multicast();
    setup_flow_steering();
    post_recv_batch(NUM_WR);

    get_strip_headers_block_and_grid(&d_min_grid_size, &d_block_size);
}

ibv_source_impl::~ibv_source_impl()
{
    if (d_igmp_sock >= 0)
        close(d_igmp_sock);
    if (d_flow)
        ibv_destroy_flow(d_flow);
    /* d_gpu_buf and d_xport are destroyed by unique_ptr in reverse
       declaration order (MR deregistered before PD is freed). */
    if (d_slot_indices_host)
        cudaFreeHost(d_slot_indices_host);
    if (d_slot_indices_dev)
        cudaFree(d_slot_indices_dev);
    free(d_sges);
    free(d_wrs);
}

// ─────────────────────────────────────────────────────────────────────
// Setup helpers (called once from the constructor)
// ─────────────────────────────────────────────────────────────────────

void ibv_source_impl::setup_multicast()
{
    if (d_mcast_group.empty())
        return;

    d_igmp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_igmp_sock < 0)
        throw std::runtime_error("ibv_source: socket(IGMP): " +
                                 std::string(strerror(errno)));

    struct ip_mreqn mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = inet_addr(d_mcast_group.c_str());
    mreq.imr_ifindex = if_nametoindex(d_interface.c_str());
    if (setsockopt(d_igmp_sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) <
        0) {
        close(d_igmp_sock);
        d_igmp_sock = -1;
        throw std::runtime_error("ibv_source: IP_ADD_MEMBERSHIP: " +
                                 std::string(strerror(errno)));
    }
}

void ibv_source_impl::setup_flow_steering()
{
    struct {
        struct ibv_flow_attr attr;
        struct ibv_flow_spec_eth eth;
        struct ibv_flow_spec_ipv4 ipv4;
        struct ibv_flow_spec_tcp_udp udp;
    } rule;
    memset(&rule, 0, sizeof(rule));

    rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    rule.attr.size = sizeof(rule);
    rule.attr.num_of_specs = 3;
    rule.attr.port = 1;

    rule.eth.type = IBV_FLOW_SPEC_ETH;
    rule.eth.size = sizeof(rule.eth);

    if (!d_mcast_group.empty()) {
        uint32_t mcast_ip = inet_addr(d_mcast_group.c_str());
        mcast_ip_to_mac(mcast_ip, rule.eth.val.dst_mac);
        memset(rule.eth.mask.dst_mac, 0xff, 6);

        rule.ipv4.val.dst_ip = mcast_ip;
        rule.ipv4.mask.dst_ip = 0xffffffff;
    }

    rule.ipv4.type = IBV_FLOW_SPEC_IPV4;
    rule.ipv4.size = sizeof(rule.ipv4);

    rule.udp.type = IBV_FLOW_SPEC_UDP;
    rule.udp.size = sizeof(rule.udp);
    rule.udp.val.dst_port = htobe16(static_cast<uint16_t>(d_udp_port));
    rule.udp.mask.dst_port = 0xffff;

    d_flow = ibv_create_flow(d_xport->qp(), &rule.attr);
    if (!d_flow)
        throw std::runtime_error("ibv_source: ibv_create_flow: " +
                                 std::string(strerror(errno)));
}

// ─────────────────────────────────────────────────────────────────────
// Receive WR management
// ─────────────────────────────────────────────────────────────────────

void ibv_source_impl::post_recv_batch(int count)
{
    for (int i = 0; i < count; i++) {
        d_sges[i].addr = reinterpret_cast<uint64_t>(
            d_gpu_buf->data() + static_cast<uint64_t>(d_next_slot) * SLOT_SIZE);
        d_sges[i].length = SLOT_SIZE;
        d_sges[i].lkey = d_gpu_buf->mr()->lkey;

        d_wrs[i].wr_id = d_next_slot;
        d_wrs[i].sg_list = &d_sges[i];
        d_wrs[i].num_sge = 1;
        d_wrs[i].next = (i < count - 1) ? &d_wrs[i + 1] : nullptr;

        d_next_slot = (d_next_slot + 1) % d_num_slots;
    }

    struct ibv_recv_wr* bad_wr = nullptr;
    if (ibv_post_recv(d_xport->qp(), &d_wrs[0], &bad_wr))
        throw std::runtime_error("ibv_source: ibv_post_recv: " +
                                 std::string(strerror(errno)));
}

// ─────────────────────────────────────────────────────────────────────
// work()
// ─────────────────────────────────────────────────────────────────────

int ibv_source_impl::work(int noutput_items,
                          gr_vector_const_void_star& input_items,
                          gr_vector_void_star& output_items)
{
    gr::cuda::wait_for_work(detail(), d_stream);
    auto out = static_cast<uint8_t*>(output_items[0]);

    // 1. Poll CQ
    struct ibv_wc wc[CQ_POLL_BATCH];
    int n = ibv_poll_cq(d_xport->cq(), CQ_POLL_BATCH, wc);
    if (n < 0) {
        gr::cuda::mark_work_done(detail(), d_stream);
        return 0;
    }

    // 2. Validate completions
    const uint32_t expected_len = L2L3L4_HDR_LEN + d_payload_size;
    int bad_count = 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS || wc[i].byte_len != expected_len) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                GR_LOG_WARN(d_logger,
                            "ibv_source: WC error status=" +
                                std::to_string(wc[i].status));
            } else {
                GR_LOG_WARN(d_logger,
                            "ibv_source: unexpected length " +
                                std::to_string(wc[i].byte_len));
            }
            bad_count++;
            continue;
        }
        int idx = (d_ready_head + d_ready_count) % NUM_WR;
        d_ready_ring[idx] = static_cast<uint32_t>(wc[i].wr_id);
        d_ready_count++;
    }

    // Repost bad slots immediately (no kernel dependency)
    if (bad_count > 0)
        post_recv_batch(bad_count);

    // 3. Determine batch size
    int max_pkts = noutput_items / d_payload_size;
    int num_pkts = std::min(max_pkts, d_ready_count);
    if (num_pkts <= 0) {
        gr::cuda::mark_work_done(detail(), d_stream);
        return 0;
    }

    // 4. Copy slot indices to GPU
    for (int i = 0; i < num_pkts; i++)
        d_slot_indices_host[i] = d_ready_ring[(d_ready_head + i) % NUM_WR];

    check_cuda_errors(cudaMemcpyAsync(d_slot_indices_dev,
                                      d_slot_indices_host,
                                      num_pkts * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice,
                                      d_stream),
                      "ibv_source: cudaMemcpyAsync slot_indices",
                      d_logger);

    // 5. Launch strip-headers kernel
    exec_strip_headers_kernel(d_gpu_buf->data(),
                              d_slot_indices_dev,
                              out,
                              L2L3L4_HDR_LEN,
                              SLOT_SIZE,
                              d_payload_size,
                              num_pkts,
                              d_min_grid_size,
                              d_block_size,
                              d_stream);

    /*
     * Sync: the kernel reads from landing-buffer slots; we must not repost
     * them (allowing the NIC to overwrite) until the kernel is done.
     */
    check_cuda_errors(
        cudaStreamSynchronize(d_stream), "ibv_source: cudaStreamSynchronize", d_logger);

    // 6. Advance ready ring and batch-repost consumed slots
    d_ready_head = (d_ready_head + num_pkts) % NUM_WR;
    d_ready_count -= num_pkts;

    post_recv_batch(num_pkts);

    gr::cuda::mark_work_done(detail(), d_stream);
    return num_pkts * d_payload_size;
}

} /* namespace cuda */
} /* namespace gr */
