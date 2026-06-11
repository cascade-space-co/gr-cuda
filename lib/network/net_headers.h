/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Packed network header structs and helpers used by the IBV sink/source
 * blocks to build or parse raw Ethernet/IPv4/UDP frames on the GPU.
 *
 * All multi-byte header fields are stored in network byte order unless
 * otherwise noted.  The structs are __attribute__((packed)) so they can
 * be overlaid directly onto a frame buffer without padding surprises.
 */

#ifndef INCLUDED_GR_CUDA_NET_HEADERS_H
#define INCLUDED_GR_CUDA_NET_HEADERS_H

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gr {
namespace cuda {

// Standard header lengths (bytes).  The combined 42 bytes precede
// every UDP payload in the raw Ethernet frames we send/receive.
static constexpr int ETH_HDR_LEN = 14; // dst(6) + src(6) + ethertype(2)
static constexpr int IP_HDR_LEN = 20;  // IPv4 without options
static constexpr int UDP_HDR_LEN = 8;  // src_port + dst_port + len + csum
static constexpr int L2L3L4_HDR_LEN = ETH_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN; // 42

// ---- Layer 2: Ethernet II ------------------------------------------------

struct __attribute__((packed)) eth_hdr {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype; // 0x0800 = IPv4
};

// ---- Layer 3: IPv4 -------------------------------------------------------

struct __attribute__((packed)) ip_hdr {
    uint8_t ver_ihl; // version (4 bits) | IHL (4 bits); 0x45 = IPv4, 20B
    uint8_t dscp_ecn;
    uint16_t total_len; // IP header + UDP header + payload
    uint16_t id;
    uint16_t flags_frag; // 0x4000 = Don't Fragment
    uint8_t ttl;
    uint8_t protocol; // 17 = UDP
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
};

// ---- Layer 4: UDP --------------------------------------------------------

struct __attribute__((packed)) udp_hdr {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;   // UDP header + payload
    uint16_t checksum; // optional for IPv4 (may be left 0)
};

// ---- Utilities -----------------------------------------------------------

// One's-complement checksum over `len` bytes (must be even).
// Used to compute the IPv4 header checksum.
//
// Byte-addressed to avoid any strict-aliasing pitfalls with the
// `uint16_t*` cast, and to not depend on `data` being 2-byte aligned.
inline uint16_t ip_checksum(const void* data, int len)
{
    auto bytes = static_cast<const uint8_t*>(data);
    uint32_t sum = 0;
    for (int i = 0; i + 1 < len; i += 2)
        sum += (static_cast<uint32_t>(bytes[i]) << 8) | bytes[i + 1];
    if (len & 1)
        sum += static_cast<uint32_t>(bytes[len - 1]) << 8;
    while (sum >> 16)
        sum = (sum & 0xffff) + (sum >> 16);
    return htons(static_cast<uint16_t>(~sum));
}

// Parameters for build_frame_header().  MACs are raw 6-byte arrays; IPs are
// in network byte order (as returned by inet_addr / SIOCGIFADDR); ports and
// payload_size are in host byte order.
struct frame_header_params {
    uint8_t src_mac[6];
    uint8_t dst_mac[6];
    uint32_t src_ip;   // network byte order
    uint32_t dst_ip;   // network byte order
    uint16_t src_port; // host byte order
    uint16_t dst_port; // host byte order
    uint8_t ttl;
    uint16_t payload_size;
};

// Assemble a 42-byte Ethernet/IPv4/UDP header into `out`.
//
// Pure function: the output depends only on `p` (no netdev queries, no GPU).
// The IPv4 header checksum is computed and filled in; the UDP checksum is
// left 0 (optional for IPv4).  This is the unit-testable core of the IBV
// sink's header construction, kept here so it can be exercised without any
// ibverbs/hardware dependency.
inline void build_frame_header(uint8_t out[L2L3L4_HDR_LEN],
                               const frame_header_params& p)
{
    memset(out, 0, L2L3L4_HDR_LEN);

    auto* eth = reinterpret_cast<eth_hdr*>(out);
    memcpy(eth->dst_mac, p.dst_mac, 6);
    memcpy(eth->src_mac, p.src_mac, 6);
    eth->ethertype = htons(0x0800);

    auto* ip = reinterpret_cast<ip_hdr*>(out + ETH_HDR_LEN);
    ip->ver_ihl = 0x45;
    ip->total_len =
        htons(static_cast<uint16_t>(IP_HDR_LEN + UDP_HDR_LEN + p.payload_size));
    ip->flags_frag = htons(0x4000); // Don't Fragment
    ip->ttl = p.ttl;
    ip->protocol = 17; // UDP
    ip->src_ip = p.src_ip;
    ip->dst_ip = p.dst_ip;
    ip->checksum = ip_checksum(ip, IP_HDR_LEN);

    auto* udp = reinterpret_cast<udp_hdr*>(out + ETH_HDR_LEN + IP_HDR_LEN);
    udp->src_port = htons(p.src_port);
    udp->dst_port = htons(p.dst_port);
    udp->length = htons(static_cast<uint16_t>(UDP_HDR_LEN + p.payload_size));
    // UDP checksum left as 0 (optional for IPv4).
}

// Map IPv4 multicast IP (network byte order) to its IEEE 802.3 MAC
// address per RFC 7042 / IANA OUI 01:00:5e.  The low 23 bits of the
// IP are placed into the low 23 bits of the MAC.
inline void mcast_ip_to_mac(uint32_t mcast_ip_net, uint8_t* mac)
{
    uint32_t mip = ntohl(mcast_ip_net);
    mac[0] = 0x01;
    mac[1] = 0x00;
    mac[2] = 0x5e;
    mac[3] = (mip >> 16) & 0x7f;
    mac[4] = (mip >> 8) & 0xff;
    mac[5] = mip & 0xff;
}

// Query the 6-byte hardware (MAC) address of a Linux network interface
// via ioctl(SIOCGIFHWADDR).  Returns 0 on success, -1 on failure.
inline int get_mac_address(const char* ifname, uint8_t* mac)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
        return -1;
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    int ret = ioctl(fd, SIOCGIFHWADDR, &ifr);
    close(fd);
    if (ret < 0)
        return -1;
    memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
    return 0;
}

// Query the IPv4 address (network byte order) of a Linux network
// interface via ioctl(SIOCGIFADDR).  Returns 0 on success, -1 on failure.
inline int get_ipv4_address(const char* ifname, uint32_t* ip)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
        return -1;
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    int ret = ioctl(fd, SIOCGIFADDR, &ifr);
    close(fd);
    if (ret < 0)
        return -1;
    *ip = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_addr)->sin_addr.s_addr;
    return 0;
}

// Parse a colon-separated MAC string "xx:xx:xx:xx:xx:xx" into 6 bytes.
// Returns 0 on success, -1 on failure.
inline int parse_mac(const char* str, uint8_t* mac)
{
    return (sscanf(str,
                   "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
                   &mac[0],
                   &mac[1],
                   &mac[2],
                   &mac[3],
                   &mac[4],
                   &mac[5]) == 6)
               ? 0
               : -1;
}

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_GR_CUDA_NET_HEADERS_H */
