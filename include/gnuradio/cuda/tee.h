/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_TEE_H
#define INCLUDED_CUDA_TEE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Split one input into two identical outputs.
 * \ingroup cuda
 *
 * GNU Radio does not support mixed fan-out: a single output port
 * cannot feed both GPU and CPU downstream blocks because all readers
 * share one buffer with one transfer type.  This block works around
 * that by explicitly copying the input to two separate output ports,
 * each with its own buffer.  Downstream blocks connect to whichever
 * port matches their domain, and GNU Radio's buffer negotiation
 * handles any H2D / D2H transfers at the boundaries automatically.
 *
 * When \p gpu is true, all ports use cuda_buffer and the copy is a
 * device-to-device cudaMemcpyAsync.  When false, all ports use the
 * default host buffer and the copy is a plain memcpy.  The \p gpu
 * flag must match the upstream block's buffer type because
 * io_signature is fixed at construction time.
 */
class CUDA_API tee : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<tee> sptr;

    /*!
     * \param itemsize Size of a stream item in bytes.
     * \param gpu      If true, use cuda_buffer (D2D copy); otherwise use
     *                 default host buffers (memcpy).
     */
    static sptr make(size_t itemsize, bool gpu = true);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TEE_H */
