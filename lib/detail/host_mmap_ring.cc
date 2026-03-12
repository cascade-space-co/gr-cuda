/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "host_mmap_ring.h"

#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/logger.h>

#include <cassert>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace gr {
namespace detail {

host_mmap_ring::~host_mmap_ring() { this->reset(); }

#ifdef _WIN32
// Windows: VirtualAlloc2 + MapViewOfFile3 placeholder mechanism (Win 10 1803+).

std::unique_ptr<host_mmap_ring>
host_mmap_ring::create(size_t requested_bytes,
                       const std::shared_ptr<gr::logger>& logger)
{
    auto ring = std::unique_ptr<host_mmap_ring>(new host_mmap_ring());
    ring->d_logger = logger;

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    assert(requested_bytes % si.dwPageSize == 0);

    ring->d_bytes = requested_bytes;
    logger->debug("host_mmap_ring: requesting {} bytes (page_size={})",
                  requested_bytes,
                  si.dwPageSize);

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE,
                                        nullptr,
                                        PAGE_READWRITE,
                                        (DWORD)(ring->d_bytes >> 32),
                                        (DWORD)(ring->d_bytes & 0xFFFFFFFF),
                                        nullptr);
    if (!section)
        throw std::runtime_error("host_mmap_ring: CreateFileMapping failed");

    // Reserve 2N contiguous VA as a placeholder.
    void* region = VirtualAlloc2(nullptr,
                                 nullptr,
                                 2 * ring->d_bytes,
                                 MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                 PAGE_NOACCESS,
                                 nullptr,
                                 0);
    if (!region) {
        CloseHandle(section);
        throw std::runtime_error("host_mmap_ring: VirtualAlloc2 reserve failed");
    }

    // Split the 2N placeholder into two N-sized placeholders.
    if (!VirtualFree(region, ring->d_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
        VirtualFree(region, 0, MEM_RELEASE);
        CloseHandle(section);
        throw std::runtime_error("host_mmap_ring: placeholder split failed");
    }

    // Map section into first placeholder [base, base+N).
    void* p1 = MapViewOfFile3(section,
                              nullptr,
                              region,
                              0,
                              ring->d_bytes,
                              MEM_REPLACE_PLACEHOLDER,
                              PAGE_READWRITE,
                              nullptr,
                              0);
    if (!p1) {
        VirtualFree(region, 0, MEM_RELEASE);
        VirtualFree(static_cast<char*>(region) + ring->d_bytes, 0, MEM_RELEASE);
        CloseHandle(section);
        throw std::runtime_error("host_mmap_ring: MapViewOfFile3 first half failed");
    }

    // Map same section into second placeholder [base+N, base+2N).
    void* p2 = MapViewOfFile3(section,
                              nullptr,
                              static_cast<char*>(region) + ring->d_bytes,
                              0,
                              ring->d_bytes,
                              MEM_REPLACE_PLACEHOLDER,
                              PAGE_READWRITE,
                              nullptr,
                              0);
    if (!p2) {
        UnmapViewOfFile(p1);
        VirtualFree(static_cast<char*>(region) + ring->d_bytes, 0, MEM_RELEASE);
        CloseHandle(section);
        throw std::runtime_error("host_mmap_ring: MapViewOfFile3 second half failed");
    }

    CloseHandle(section);
    ring->d_base = region;
    logger->debug(
        "host_mmap_ring: mapped 2x{} bytes at {}", ring->d_bytes, ring->d_base);
    return ring;
}

#else // POSIX

std::unique_ptr<host_mmap_ring>
host_mmap_ring::create(size_t requested_bytes,
                       const std::shared_ptr<gr::logger>& logger)
{
    auto ring = std::unique_ptr<host_mmap_ring>(new host_mmap_ring());
    ring->d_logger = logger;

    // Caller (cuda_buffer::allocate_buffer) already rounds up to VMM
    // granularity, which is always a multiple of the system page size.
    long page_size = sysconf(_SC_PAGESIZE);
    assert(requested_bytes % (size_t)page_size == 0);

    ring->d_bytes = requested_bytes;
    logger->debug("host_mmap_ring: requesting {} bytes (page_size={})",
                  requested_bytes,
                  page_size);

    // Anonymous file backed by RAM
    int fd = static_cast<int>(syscall(SYS_memfd_create, "gr_cuda_buf", 0));
    if (fd < 0)
        throw std::runtime_error("host_circ_create: memfd_create failed");

    if (ftruncate(fd, (off_t)ring->d_bytes) != 0) {
        close(fd);
        throw std::runtime_error("host_circ_create: ftruncate failed");
    }

    // Reserve 2N contiguous VA with no access rights (placeholder).
    void* region =
        mmap(nullptr, 2 * ring->d_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("host_circ_create: VA reservation mmap failed");
    }

    // Map the fd into the first half [base, base+N), replacing placeholder.
    void* p1 = mmap(
        region, ring->d_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (p1 == MAP_FAILED) {
        munmap(region, 2 * ring->d_bytes);
        close(fd);
        throw std::runtime_error("host_circ_create: first-half mmap failed");
    }

    // Map the same fd into the second half [base+N, base+2N).
    void* p2 = mmap(static_cast<char*>(region) + ring->d_bytes,
                    ring->d_bytes,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    fd,
                    0);
    if (p2 == MAP_FAILED) {
        munmap(region, 2 * ring->d_bytes);
        close(fd);
        throw std::runtime_error("host_circ_create: second-half mmap failed");
    }

    // fd can be closed immediately; mappings hold a reference.
    close(fd);
    ring->d_base = region;
    logger->debug(
        "host_mmap_ring: mapped 2x{} bytes at {}", ring->d_bytes, ring->d_base);
    return ring;
}

#endif // _WIN32

void host_mmap_ring::register_pinned()
{
    // Register only one half. post_work() splits wrap-crossing DMA
    // into [0, N) segments.
    cudaError_t rc = cudaHostRegister(d_base, d_bytes, cudaHostRegisterDefault);
    check_cuda_errors(rc, "host_circ_create: cudaHostRegister failed", d_logger);
    d_registered = true;
    d_logger->debug("host_mmap_ring: pinned {} bytes at {}", d_bytes, d_base);
}

char* host_mmap_ring::base_ptr() { return static_cast<char*>(d_base); }

void host_mmap_ring::reset()
{
    if (d_registered && d_base) {
        cudaHostUnregister(d_base);
        d_registered = false;
    }
    if (d_base) {
#ifdef _WIN32
        UnmapViewOfFile(d_base);
        UnmapViewOfFile(static_cast<char*>(d_base) + d_bytes);
#else
        munmap(d_base, 2 * d_bytes);
#endif
        d_base = nullptr;
    }
    d_bytes = 0;
}

} // namespace detail
} // namespace gr
