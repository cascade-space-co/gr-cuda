/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

static constexpr int ETH_HDR_LEN = 14;
static constexpr int IP_HDR_LEN = 20;
static constexpr int UDP_HDR_LEN = 8;
static constexpr int L2L3L4_HDR_LEN = ETH_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN;

struct __attribute__((packed)) eth_hdr {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
};

struct __attribute__((packed)) ip_hdr {
    uint8_t ver_ihl;
    uint8_t dscp_ecn;
    uint16_t total_len;
    uint16_t id;
    uint16_t flags_frag;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
};

struct __attribute__((packed)) udp_hdr {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
};

inline uint16_t ip_checksum(const void* data, int len)
{
    auto words = static_cast<const uint16_t*>(data);
    uint32_t sum = 0;
    for (int i = 0; i < len / 2; i++)
        sum += ntohs(words[i]);
    while (sum >> 16)
        sum = (sum & 0xffff) + (sum >> 16);
    return htons(static_cast<uint16_t>(~sum));
}

/* RFC 7042 / IANA OUI 01:00:5e -- derive multicast MAC from IP (network order). */
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
