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
#include "network/net_headers.h"
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/prefs.h>

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
                                  int udp_port,
                                  int payload_size,
                                  const std::string& mcast_group)
{
    return gnuradio::make_block_sptr<ibv_source_impl>(
        ibv_device, udp_port, payload_size, mcast_group);
}

ibv_source_impl::ibv_source_impl(const std::string& ibv_device,
                                 int udp_port,
                                 int payload_size,
                                 const std::string& mcast_group)
    : sync_block("ibv_source",
                 io_signature::make(0, 0, 0),
                 io_signature::make(1, 1, payload_size, cuda_buffer::type)),
      d_payload_size(payload_size),
      d_mcast_group(mcast_group),
      d_udp_port(udp_port)
{
    if (d_payload_size <= 0)
        throw std::runtime_error("ibv_source: payload_size must be > 0");
    // Validate the UDP port before it is truncated to uint16_t for the
    // flow-steering rule; otherwise out-of-range values wrap silently.
    if (d_udp_port < 1 || d_udp_port > 65535)
        throw std::runtime_error("ibv_source: udp_port must be in [1, 65535], got " +
                                 std::to_string(d_udp_port));
    // Each NIC slot holds one complete raw Ethernet frame: 42-byte
    // L2/L3/L4 header + payload.
    if (L2L3L4_HDR_LEN + d_payload_size > SLOT_SIZE)
        throw std::runtime_error("ibv_source: frame size (" +
                                 std::to_string(L2L3L4_HDR_LEN + d_payload_size) +
                                 ") exceeds slot size (" + std::to_string(SLOT_SIZE) +
                                 ")");

    // Resolve runtime tuning knobs from gr::prefs (with the compile-time
    // DEFAULT_* values as fallbacks).
    auto* prefs = gr::prefs::singleton();
    d_num_wr =
        static_cast<int>(prefs->get_long("ibv_source", "num_wr", DEFAULT_NUM_WR));
    d_cq_size =
        static_cast<int>(prefs->get_long("ibv_source", "cq_size", d_num_wr * 2));
    d_cq_poll_batch = static_cast<int>(
        prefs->get_long("ibv_source", "cq_poll_batch", DEFAULT_CQ_POLL_BATCH));
    d_gpu_buf_size = static_cast<size_t>(prefs->get_long(
        "ibv_source", "gpu_buf_bytes", static_cast<long>(DEFAULT_GPU_BUF_BYTES)));

    if (d_num_wr <= 0 || d_cq_poll_batch <= 0)
        throw std::invalid_argument("ibv_source: num_wr and cq_poll_batch must be > 0");
    if (d_cq_size < d_num_wr)
        throw std::invalid_argument("ibv_source: cq_size must be >= num_wr "
                                    "(every recv WR produces a completion)");
    if (d_gpu_buf_size / SLOT_SIZE < static_cast<size_t>(d_num_wr))
        throw std::invalid_argument("ibv_source: gpu_buf_bytes must hold at least "
                                    "num_wr slots of SLOT_SIZE bytes");

    // Ensure the scheduler always provides at least cq_poll_batch items
    // of output space so we can deliver a full CQ poll in one kernel launch.
    set_output_multiple(d_cq_poll_batch);

    // Open the IB device and create the QP (raw Ethernet, receive-only;
    // no RTS transition needed for recv).
    ibv_transport::qp_config cfg;
    cfg.max_recv_wr = d_num_wr;
    cfg.max_recv_sge = 1;
    cfg.cq_size = d_cq_size;
    d_xport = std::make_unique<ibv_transport>(ibv_device, cfg);

    // Allocate a GPU-resident landing buffer registered as an IB MR.
    // The NIC DMAs raw Ethernet frames directly into this GPU memory.
    d_gpu_buf = std::make_unique<ibv_gpu_buffer>(d_gpu_buf_size, d_xport->pd());
    d_num_slots = d_gpu_buf_size / SLOT_SIZE;

    // Pre-allocate the WR/SGE pool; per-post fields (addr, wr_id, next)
    // are patched in post_recv_batch().  Sized to d_num_wr.
    d_sges = static_cast<struct ibv_sge*>(calloc(d_num_wr, sizeof(struct ibv_sge)));
    d_wrs =
        static_cast<struct ibv_recv_wr*>(calloc(d_num_wr, sizeof(struct ibv_recv_wr)));
    d_wc_pool.resize(d_cq_poll_batch);

    // Join multicast group (IGMP), install NIC flow-steering rule, and
    // seed the receive queue with d_num_wr buffers so the NIC can start
    // receiving immediately.
    setup_multicast();
    setup_flow_steering();
    post_recv_batch(d_num_wr);

    get_strip_headers_block_and_grid(&d_min_grid_size, &d_block_size);
}

ibv_source_impl::~ibv_source_impl()
{
    if (d_igmp_sock >= 0)
        close(d_igmp_sock);
    if (d_flow)
        ibv_destroy_flow(d_flow);
    // d_gpu_buf and d_xport are destroyed by unique_ptr in reverse
    // declaration order (MR deregistered before PD is freed)
    free(d_sges);
    free(d_wrs);
}

// start() - check scheduler buffer sizing
bool ibv_source_impl::start()
{
    size_t buf_items = detail()->output(0)->bufsize();
    size_t buf_bytes = buf_items * d_payload_size;

    GR_LOG_INFO(d_logger,
                "ibv_source: output buffer " + std::to_string(buf_items) +
                    " packets (" + std::to_string(buf_bytes / (1024 * 1024)) +
                    " MB), num_wr=" + std::to_string(d_num_wr) +
                    ", cq_poll_batch=" + std::to_string(d_cq_poll_batch));

    if (buf_items < static_cast<size_t>(d_cq_poll_batch) * 4) {
        GR_LOG_WARN(
            d_logger,
            "ibv_source: output buffer holds only " + std::to_string(buf_items) +
                " packets — recommend >= " + std::to_string(d_cq_poll_batch * 4) +
                " (4x cq_poll_batch) to avoid scheduler stalls");
    }
    return sync_block::start();
}

// Setup helpers (called once from the constructor)
// Join an IGMP multicast group on the NIC so the switch forwards
// traffic to us.  We open a throwaway UDP socket and issue
// IP_ADD_MEMBERSHIP with the interface index derived from the IB
// device (via sysfs).  The socket is kept open for the lifetime of
// the block so the kernel maintains the IGMP membership; it is
// closed in the destructor.
void ibv_source_impl::setup_multicast()
{
    if (d_mcast_group.empty())
        return;

    d_igmp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (d_igmp_sock < 0)
        throw std::runtime_error("ibv_source: socket(IGMP): " +
                                 std::string(strerror(errno)));

    std::string netdev = d_xport->netdev_name();
    struct ip_mreqn mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = inet_addr(d_mcast_group.c_str());
    mreq.imr_ifindex = if_nametoindex(netdev.c_str());
    if (setsockopt(d_igmp_sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) <
        0) {
        close(d_igmp_sock);
        d_igmp_sock = -1;
        throw std::runtime_error("ibv_source: IP_ADD_MEMBERSHIP: " +
                                 std::string(strerror(errno)));
    }
}

// Install a hardware flow-steering rule on the NIC so that only
// matching packets land in our QP.  The rule is a three-layer filter:
//   L2 (Eth) -- for multicast: match the IANA-mapped mcast MAC
//   L3 (IPv4) -- for multicast: match the mcast group IP
//   L4 (UDP)  -- always: match the destination port
//
// For unicast the L2/L3 specs are left zeroed (wildcard), so the NIC
// steers solely on the UDP dst port.  The rule is destroyed in the
// destructor via ibv_destroy_flow().
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
    rule.attr.port = 1; // physical port 1 (ConnectX numbering)

    rule.eth.type = IBV_FLOW_SPEC_ETH;
    rule.eth.size = sizeof(rule.eth);

    if (!d_mcast_group.empty()) {
        // Map the multicast IP to its IEEE 802.3 MAC (01:00:5e:xx:xx:xx)
        // and require an exact match on both L2 dst MAC and L3 dst IP.
        uint32_t mcast_ip = inet_addr(d_mcast_group.c_str());
        mcast_ip_to_mac(mcast_ip, rule.eth.val.dst_mac);
        memset(rule.eth.mask.dst_mac, 0xff, 6);

        rule.ipv4.val.dst_ip = mcast_ip;
        rule.ipv4.mask.dst_ip = 0xffffffff;
    }

    // IPv4 spec -- for unicast the val/mask fields stay zeroed (wildcard).
    rule.ipv4.type = IBV_FLOW_SPEC_IPV4;
    rule.ipv4.size = sizeof(rule.ipv4);

    // UDP spec -- always match on dst port (network byte order).
    rule.udp.type = IBV_FLOW_SPEC_UDP;
    rule.udp.size = sizeof(rule.udp);
    rule.udp.val.dst_port = htobe16(static_cast<uint16_t>(d_udp_port));
    rule.udp.mask.dst_port = 0xffff;

    // Attach the rule to our QP; the NIC will now deliver only matching
    // frames into this QP's receive queue.
    d_flow = ibv_create_flow(d_xport->qp(), &rule.attr);
    if (!d_flow)
        throw std::runtime_error("ibv_source: ibv_create_flow: " +
                                 std::string(strerror(errno)));
}

// Receive WR management
void ibv_source_impl::post_recv_batch(int count)
{
    // Build a linked list of `count` receive WRs, each pointing at the
    // next free slot in the GPU landing buffer.  The NIC will DMA one
    // raw Ethernet frame into each slot.  We chain them so a single
    // ibv_post_recv() call posts the entire batch atomically.
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

// work()
int ibv_source_impl::work(int noutput_items,
                          gr_vector_const_void_star& input_items,
                          gr_vector_void_star& output_items)
{
    // Spin-poll CQ to accumulate a full batch before launching a kernel.
    // Reset the idle counter on progress so we keep spinning as long as
    // packets are flowing, and only give up after a burst of empty polls.
    const uint32_t expected_len = L2L3L4_HDR_LEN + d_payload_size;
    constexpr int IDLE_LIMIT = 4000;
    int idle = 0;
    struct ibv_wc* wc = d_wc_pool.data();

    while (idle < IDLE_LIMIT && d_ready_count < d_cq_poll_batch) {
        int n = ibv_poll_cq(d_xport->cq(), d_cq_poll_batch, wc);
        if (n < 0)
            return 0;
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    GR_LOG_WARN(d_logger,
                                "ibv_source: WC error status=" +
                                    std::to_string(wc[i].status));
                } else if (wc[i].byte_len != expected_len) {
                    GR_LOG_WARN(d_logger,
                                "ibv_source: unexpected length " +
                                    std::to_string(wc[i].byte_len));
                }
            }
            d_ready_count += n;
            idle = 0;
        } else {
            idle++;
        }
    }

    int num_pkts = std::min(noutput_items, d_ready_count);
    if (num_pkts <= 0)
        return 0;

    auto out = static_cast<uint8_t*>(output_items[0]);

    const uint32_t first_slot = d_ready_slot;
    exec_strip_headers_kernel(d_gpu_buf->data(),
                              SLOT_SIZE,
                              first_slot,
                              d_num_slots,
                              out,
                              L2L3L4_HDR_LEN,
                              d_payload_size,
                              num_pkts,
                              d_block_size,
                              d_stream);

    // Sync: the kernel reads from landing-buffer slots; we must not repost
    // them (allowing the NIC to overwrite) until the kernel is done.
    check_cuda_errors(
        cudaStreamSynchronize(d_stream), "ibv_source: cudaStreamSynchronize", d_logger);

    d_ready_slot = (d_ready_slot + num_pkts) % d_num_slots;
    d_ready_count -= num_pkts;
    post_recv_batch(num_pkts);

    return num_pkts;
}

} /* namespace cuda */
} /* namespace gr */
