/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Host-only unit tests for the pure helpers in network/net_headers.h:
 * the IPv4 checksum, multicast IP->MAC mapping, MAC string parsing, and
 * the Ethernet/IPv4/UDP header builder.  None of these touch a netdev,
 * ibverbs, or the GPU, so the file builds and runs without ENABLE_IBV.
 */

#include "network/net_headers.h"

#include <arpa/inet.h>
#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <cstring>

using namespace gr::cuda;

// Worked IPv4-header checksum example (the classic Wikipedia one):
//   45 00 00 73 00 00 40 00 40 11 00 00 c0 a8 00 01 c0 a8 00 c7
// with the checksum field zeroed yields checksum bytes 0xb8 0x61.
BOOST_AUTO_TEST_CASE(ip_checksum_known_vector)
{
    const uint8_t hdr[IP_HDR_LEN] = { 0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40,
                                      0x00, 0x40, 0x11, 0x00, 0x00, 0xc0, 0xa8,
                                      0x00, 0x01, 0xc0, 0xa8, 0x00, 0xc7 };
    uint16_t csum = ip_checksum(hdr, IP_HDR_LEN);

    // ip_checksum returns the value in network byte order; check the bytes
    // directly so the test is endianness-independent.
    const auto* b = reinterpret_cast<const uint8_t*>(&csum);
    BOOST_CHECK_EQUAL(b[0], 0xb8);
    BOOST_CHECK_EQUAL(b[1], 0x61);
}

// A header carrying its own correct checksum must checksum to zero.
BOOST_AUTO_TEST_CASE(ip_checksum_self_validates)
{
    uint8_t hdr[IP_HDR_LEN] = { 0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40,
                                0x00, 0x40, 0x11, 0x00, 0x00, 0xc0, 0xa8,
                                0x00, 0x01, 0xc0, 0xa8, 0x00, 0xc7 };
    uint16_t csum = ip_checksum(hdr, IP_HDR_LEN);
    std::memcpy(&hdr[10], &csum, sizeof(csum)); // checksum field at offset 10
    BOOST_CHECK_EQUAL(ip_checksum(hdr, IP_HDR_LEN), 0);
}

BOOST_AUTO_TEST_CASE(mcast_ip_to_mac_basic)
{
    uint8_t mac[6];
    mcast_ip_to_mac(inet_addr("239.1.2.3"), mac);
    const uint8_t expect[6] = { 0x01, 0x00, 0x5e, 0x01, 0x02, 0x03 };
    BOOST_CHECK_EQUAL(std::memcmp(mac, expect, 6), 0);

    mcast_ip_to_mac(inet_addr("224.0.0.1"), mac);
    const uint8_t expect2[6] = { 0x01, 0x00, 0x5e, 0x00, 0x00, 0x01 };
    BOOST_CHECK_EQUAL(std::memcmp(mac, expect2, 6), 0);
}

// Only the low 23 bits of the IP map into the MAC, so the high bit of the
// second octet is masked off: 239.129.2.3 collides with 239.1.2.3.
BOOST_AUTO_TEST_CASE(mcast_ip_to_mac_23bit_mask)
{
    uint8_t mac_a[6], mac_b[6];
    mcast_ip_to_mac(inet_addr("239.1.2.3"), mac_a);
    mcast_ip_to_mac(inet_addr("239.129.2.3"), mac_b);
    BOOST_CHECK_EQUAL(std::memcmp(mac_a, mac_b, 6), 0);
    BOOST_CHECK_EQUAL(mac_b[3], 0x01); // 129 & 0x7f == 1
}

BOOST_AUTO_TEST_CASE(parse_mac_valid)
{
    uint8_t mac[6];
    BOOST_CHECK_EQUAL(parse_mac("01:23:45:67:89:ab", mac), 0);
    const uint8_t expect[6] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab };
    BOOST_CHECK_EQUAL(std::memcmp(mac, expect, 6), 0);
}

BOOST_AUTO_TEST_CASE(parse_mac_invalid)
{
    uint8_t mac[6];
    BOOST_CHECK_EQUAL(parse_mac("not-a-mac", mac), -1);
    BOOST_CHECK_EQUAL(parse_mac("01:23:45", mac), -1); // too few octets
}

BOOST_AUTO_TEST_CASE(build_frame_header_fields)
{
    frame_header_params p{};
    const uint8_t src_mac[6] = { 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };
    const uint8_t dst_mac[6] = { 0x21, 0x22, 0x23, 0x24, 0x25, 0x26 };
    std::memcpy(p.src_mac, src_mac, 6);
    std::memcpy(p.dst_mac, dst_mac, 6);
    p.src_ip = inet_addr("192.168.0.1");
    p.dst_ip = inet_addr("192.168.0.199");
    p.src_port = 12345;
    p.dst_port = 5000;
    p.ttl = 64;
    p.payload_size = 1024;

    uint8_t hdr[L2L3L4_HDR_LEN];
    build_frame_header(hdr, p);

    const auto* eth = reinterpret_cast<const eth_hdr*>(hdr);
    BOOST_CHECK_EQUAL(std::memcmp(eth->dst_mac, dst_mac, 6), 0);
    BOOST_CHECK_EQUAL(std::memcmp(eth->src_mac, src_mac, 6), 0);
    BOOST_CHECK_EQUAL(eth->ethertype, htons(0x0800));

    const auto* ip = reinterpret_cast<const ip_hdr*>(hdr + ETH_HDR_LEN);
    BOOST_CHECK_EQUAL(ip->ver_ihl, 0x45);
    BOOST_CHECK_EQUAL(ip->total_len, htons(IP_HDR_LEN + UDP_HDR_LEN + 1024));
    BOOST_CHECK_EQUAL(ip->flags_frag, htons(0x4000));
    BOOST_CHECK_EQUAL(ip->ttl, 64);
    BOOST_CHECK_EQUAL(ip->protocol, 17);
    BOOST_CHECK_EQUAL(ip->src_ip, p.src_ip);
    BOOST_CHECK_EQUAL(ip->dst_ip, p.dst_ip);
    // The filled-in checksum must self-validate.
    BOOST_CHECK_EQUAL(ip_checksum(ip, IP_HDR_LEN), 0);

    const auto* udp = reinterpret_cast<const udp_hdr*>(hdr + ETH_HDR_LEN + IP_HDR_LEN);
    BOOST_CHECK_EQUAL(udp->src_port, htons(12345));
    BOOST_CHECK_EQUAL(udp->dst_port, htons(5000));
    BOOST_CHECK_EQUAL(udp->length, htons(UDP_HDR_LEN + 1024));
    BOOST_CHECK_EQUAL(udp->checksum, 0); // left unset for IPv4
}
