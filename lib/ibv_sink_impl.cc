/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "ibv_sink.cuh"
#include "ibv_sink_impl.h"
#include "network/net_headers.h"
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>

#include <arpa/inet.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>

namespace gr {
namespace cuda {

ibv_sink::sptr ibv_sink::make(const std::string& ibv_device,
                              const std::string& dst_ip,
                              int dst_port,
                              int payload_size,
                              const std::string& dst_mac,
                              const std::string& mcast_group)
{
    return gnuradio::make_block_sptr<ibv_sink_impl>(
        ibv_device, dst_ip, dst_port, payload_size, dst_mac, mcast_group);
}

ibv_sink_impl::ibv_sink_impl(const std::string& ibv_device,
                             const std::string& dst_ip,
                             int dst_port,
                             int payload_size,
                             const std::string& dst_mac,
                             const std::string& mcast_group)
    : sync_block("ibv_sink",
                 io_signature::make(1, 1, payload_size, cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_payload_size(payload_size),
      d_dst_ip(dst_ip),
      d_dst_port(dst_port),
      d_dst_mac(dst_mac),
      d_mcast_group(mcast_group)
{
    if (d_payload_size <= 0)
        throw std::runtime_error("ibv_sink: payload_size must be > 0");

    // Each NIC slot holds one complete raw Ethernet frame: 42-byte
    // L2/L3/L4 header + payload.
    d_frame_size = L2L3L4_HDR_LEN + d_payload_size;
    d_slot_size = d_frame_size;
    if (d_slot_size > MAX_SLOT_SIZE)
        throw std::runtime_error(
            "ibv_sink: frame size (" + std::to_string(d_slot_size) +
            ") exceeds max slot size (" + std::to_string(MAX_SLOT_SIZE) + ")");

    // Ensure the scheduler always hands us at least SIGNAL_BATCH items
    // so we can amortize CQ signalling over a batch of sends.
    set_output_multiple(SIGNAL_BATCH);

    // Open the IB device and create the QP in Ready-To-Send state
    // (raw Ethernet QP; no connection handshake needed).
    ibv_transport::qp_config cfg;
    cfg.max_send_wr = NUM_WR;
    cfg.max_send_sge = 1;
    cfg.cq_size = CQ_SIZE;
    cfg.rts = true;
    d_xport = std::make_unique<ibv_transport>(ibv_device, cfg);

    // Allocate a GPU-resident landing buffer registered as an IB MR.
    // The NIC reads frame data directly from this GPU memory via DMA.
    d_gpu_buf = std::make_unique<ibv_gpu_buffer>(GPU_BUF_SIZE, d_xport->pd());
    d_num_slots = GPU_BUF_SIZE / d_slot_size;

    // Build the 42-byte Eth/IP/UDP header template on the GPU; the
    // CUDA kernel will replicate it into every slot before each send.
    build_header();

    // Pre-allocate the WR/SGE pool.  Fields that never change (length,
    // lkey, opcode) are set once here; the per-send addr and signal
    // flag are patched in work().
    d_sges = static_cast<struct ibv_sge*>(calloc(NUM_WR, sizeof(struct ibv_sge)));
    d_wrs =
        static_cast<struct ibv_send_wr*>(calloc(NUM_WR, sizeof(struct ibv_send_wr)));
    for (int i = 0; i < NUM_WR; i++) {
        d_sges[i].length = d_frame_size;
        d_sges[i].lkey = d_gpu_buf->mr()->lkey;
        d_wrs[i].sg_list = &d_sges[i];
        d_wrs[i].num_sge = 1;
        d_wrs[i].opcode = IBV_WR_SEND;
    }

    get_build_frames_block_and_grid(&d_min_grid_size, &d_block_size);
}

ibv_sink_impl::~ibv_sink_impl()
{
    /* Drain outstanding sends before tearing down the QP. */
    if (d_xport && d_outstanding > 0) {
        struct ibv_wc wc[CQ_POLL_BATCH];
        for (int retry = 0; retry < 200 && d_outstanding > 0; retry++) {
            int n = ibv_poll_cq(d_xport->cq(), CQ_POLL_BATCH, wc);
            for (int i = 0; i < n; i++)
                d_outstanding -= SIGNAL_BATCH;
            usleep(500);
        }
    }
    if (d_header_template)
        cudaFree(d_header_template);
    /* d_gpu_buf and d_xport destroyed by unique_ptr in reverse
       declaration order (MR deregistered before PD is freed). */
    free(d_sges);
    free(d_wrs);
}

// start() - check scheduler buffer sizing
bool ibv_sink_impl::start()
{
    size_t buf_items = detail()->input(0)->buffer()->bufsize();
    size_t buf_bytes = buf_items * d_payload_size;

    GR_LOG_INFO(d_logger,
                "ibv_sink: input buffer " + std::to_string(buf_items) + " packets (" +
                    std::to_string(buf_bytes / (1024 * 1024)) +
                    " MB), SIGNAL_BATCH=" + std::to_string(SIGNAL_BATCH));

    if (buf_items < static_cast<size_t>(SIGNAL_BATCH) * 4) {
        GR_LOG_WARN(d_logger,
                    "ibv_sink: input buffer holds only " + std::to_string(buf_items) +
                        " packets — recommend >= " + std::to_string(SIGNAL_BATCH * 4) +
                        " (4x SIGNAL_BATCH) to avoid scheduler stalls");
    }
    return sync_block::start();
}


// Build a 42-byte Eth/IP/UDP header template on the GPU.
//
// The template is built once on the CPU (with proper checksums, byte
// order, MACs, etc.) and then copied to GPU memory.  At runtime the
// CUDA kernel memcpy's this template into every outgoing slot, so we
// avoid any per-packet header construction on the hot path.
//
// For multicast the destination MAC is derived from the group IP
// (IANA mapping 01:00:5e:xx:xx:xx) and TTL is set to 1.  For
// unicast the caller-supplied MAC/IP are used with TTL 64.
void ibv_sink_impl::build_header()
{
    // Resolve our own MAC and IP from the Linux netdev associated
    // with the IB device (discovered via sysfs).
    std::string netdev = d_xport->netdev_name();

    uint8_t src_mac[6], dst_mac_bytes[6];
    if (get_mac_address(netdev.c_str(), src_mac))
        throw std::runtime_error("ibv_sink: cannot get MAC for " + netdev);

    uint32_t src_ip;
    if (get_ipv4_address(netdev.c_str(), &src_ip))
        throw std::runtime_error("ibv_sink: cannot get IP for " + netdev);

    uint32_t dst_ip_addr;
    uint8_t ttl;

    if (!d_mcast_group.empty()) {
        uint32_t mcast_ip = inet_addr(d_mcast_group.c_str());
        mcast_ip_to_mac(mcast_ip, dst_mac_bytes);
        dst_ip_addr = mcast_ip;
        ttl = 1;
    } else {
        if (parse_mac(d_dst_mac.c_str(), dst_mac_bytes))
            throw std::runtime_error("ibv_sink: invalid dst MAC: " + d_dst_mac);
        dst_ip_addr = inet_addr(d_dst_ip.c_str());
        ttl = 64;
    }

    // Assemble the 42-byte header on the stack: [Eth 14B][IP 20B][UDP 8B]
    uint8_t hdr[L2L3L4_HDR_LEN];
    memset(hdr, 0, sizeof(hdr));

    auto* eth = reinterpret_cast<eth_hdr*>(hdr);
    memcpy(eth->dst_mac, dst_mac_bytes, 6);
    memcpy(eth->src_mac, src_mac, 6);
    eth->ethertype = htons(0x0800);

    auto* ip = reinterpret_cast<ip_hdr*>(hdr + ETH_HDR_LEN);
    ip->ver_ihl = 0x45;
    ip->total_len = htons(IP_HDR_LEN + UDP_HDR_LEN + d_payload_size);
    ip->flags_frag = htons(0x4000); // Don't Fragment
    ip->ttl = ttl;
    ip->protocol = 17; // UDP
    ip->src_ip = src_ip;
    ip->dst_ip = dst_ip_addr;
    ip->checksum = ip_checksum(ip, IP_HDR_LEN);

    auto* udp = reinterpret_cast<udp_hdr*>(hdr + ETH_HDR_LEN + IP_HDR_LEN);
    udp->src_port = htons(12345);
    udp->dst_port = htons(static_cast<uint16_t>(d_dst_port));
    udp->length = htons(UDP_HDR_LEN + d_payload_size);
    // UDP checksum left as 0 (optional for IPv4).

    // Upload the template to GPU; the CUDA kernel copies it into
    // each slot's first 42 bytes before every send batch.
    check_cuda_errors(
        cudaMalloc(reinterpret_cast<void**>(&d_header_template), L2L3L4_HDR_LEN),
        "ibv_sink: cudaMalloc header_template",
        d_logger);
    check_cuda_errors(
        cudaMemcpy(d_header_template, hdr, L2L3L4_HDR_LEN, cudaMemcpyHostToDevice),
        "ibv_sink: cudaMemcpy header_template",
        d_logger);
}

// CQ drain
void ibv_sink_impl::drain_cq()
{
    struct ibv_wc wc[CQ_POLL_BATCH];
    int n = ibv_poll_cq(d_xport->cq(), CQ_POLL_BATCH, wc);
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            GR_LOG_ERROR(d_logger,
                         "ibv_sink: send WC error status=" +
                             std::to_string(wc[i].status));
        }
        d_outstanding -= SIGNAL_BATCH;
    }
    if (d_outstanding < 0)
        d_outstanding = 0;
}

// work()
int ibv_sink_impl::work(int noutput_items,
                        gr_vector_const_void_star& input_items,
                        gr_vector_void_star& output_items)
{
    // noutput_items is in packet units and always a multiple of SIGNAL_BATCH
    // (guaranteed by set_output_multiple).
    int num_pkts = std::min(noutput_items, static_cast<int>(d_num_slots));
    num_pkts = (num_pkts / SIGNAL_BATCH) * SIGNAL_BATCH;

    // Bounded spin-wait for NIC send slots (pure IBV, no CUDA overhead).
    // After the spin, do a partial send with however many slots freed up.
    constexpr int SPIN_LIMIT = 4000;
    for (int spin = 0; spin < SPIN_LIMIT; spin++) {
        drain_cq();
        if (NUM_WR - d_outstanding >= num_pkts)
            break;
    }
    int available = NUM_WR - d_outstanding;
    num_pkts = std::min(num_pkts, (available / SIGNAL_BATCH) * SIGNAL_BATCH);
    if (num_pkts <= 0)
        return 0;

    // We have packets to send and slots to post into — sync GPU.
    gr::cuda::wait_for_work(detail(), d_stream);
    auto in = static_cast<const uint8_t*>(input_items[0]);

    const uint32_t first_slot = d_next_slot;
    exec_build_frames_kernel(d_gpu_buf->data(),
                             d_slot_size,
                             first_slot,
                             d_num_slots,
                             d_header_template,
                             L2L3L4_HDR_LEN,
                             in,
                             d_payload_size,
                             num_pkts,
                             d_min_grid_size,
                             d_block_size,
                             d_stream);

    // Frames must be in GPU memory before the NIC DMAs them.
    check_cuda_errors(
        cudaStreamSynchronize(d_stream), "ibv_sink: cudaStreamSynchronize", d_logger);

    // Post send WRs in chained batches of SIGNAL_BATCH.
    int posted_pkts = 0;
    for (int base = 0; base < num_pkts; base += SIGNAL_BATCH) {
        int batch_end = base + SIGNAL_BATCH;
        const uint64_t batch_wr_base = d_wr_counter;

        for (int j = base; j < batch_end; j++) {
            uint64_t wr_id = batch_wr_base + static_cast<uint64_t>(j - base);
            int idx = static_cast<int>(wr_id % NUM_WR);
            uint32_t slot = (first_slot + static_cast<uint32_t>(j)) % d_num_slots;

            d_sges[idx].addr = reinterpret_cast<uint64_t>(
                d_gpu_buf->data() + static_cast<uint64_t>(slot) * d_slot_size);
            d_wrs[idx].wr_id = wr_id;
            d_wrs[idx].send_flags = (j == batch_end - 1) ? IBV_SEND_SIGNALED : 0;
            d_wrs[idx].next = (j < batch_end - 1)
                                  ? &d_wrs[static_cast<int>((wr_id + 1) % NUM_WR)]
                                  : nullptr;
        }

        int first_idx = static_cast<int>(batch_wr_base % NUM_WR);
        struct ibv_send_wr* bad_wr = nullptr;
        if (ibv_post_send(d_xport->qp(), &d_wrs[first_idx], &bad_wr)) {
            GR_LOG_ERROR(d_logger, "ibv_sink: ibv_post_send failed");
            break;
        }
        d_wr_counter += SIGNAL_BATCH;
        d_outstanding += SIGNAL_BATCH;
        d_next_slot = (d_next_slot + SIGNAL_BATCH) % d_num_slots;
        posted_pkts += SIGNAL_BATCH;
    }

    gr::cuda::mark_work_done(detail(), d_stream);
    return posted_pkts;
}

} /* namespace cuda */
} /* namespace gr */
