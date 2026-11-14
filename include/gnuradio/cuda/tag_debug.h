/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_TAG_DEBUG_H
#define INCLUDED_CUDA_TAG_DEBUG_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/tags.h>

namespace gr {
namespace cuda {

/*!
 * \brief Print or store the stream tags on GPU streams.
 * \ingroup cuda
 *
 * The GPU counterpart of blocks.tag_debug, with the same interface.  Tags are
 * kept on the host, so the block reads them without touching the samples:
 * unlike blocks.tag_debug on a GPU stream, no device-to-host copy is made.
 */
class CUDA_API tag_debug : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<tag_debug> sptr;

    /*!
     * \brief Build a GPU tag debug block.
     *
     * \param sizeof_stream_item Size of a stream item in bytes.
     * \param name Name to identify which debug sink generated the info.
     * \param key_filter Only report tags with this key (all tags if empty).
     * \param num_inputs Number of input ports.
     */
    static sptr make(size_t sizeof_stream_item,
                     const std::string& name,
                     const std::string& key_filter = "",
                     int num_inputs = 1);

    /*!
     * \brief Tags from the last work() call, or all tags received if
     * set_save_all(true).
     */
    virtual std::vector<tag_t> current_tags() = 0;

    /*!
     * \brief Number of tags still held by input 0's buffer.
     */
    virtual int num_tags() = 0;

    /*!
     * \brief Turn printing of tags to stdout on or off.
     */
    virtual void set_display(bool d) = 0;

    /*!
     * \brief Store all tags ever received (true) or only those from the last
     * work() call (false).
     */
    virtual void set_save_all(bool s) = 0;

    /*!
     * \brief Set the key to filter with (all tags if empty).
     */
    virtual void set_key_filter(const std::string& key_filter) = 0;

    /*!
     * \brief The current filter key.
     */
    virtual std::string key_filter() const = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TAG_DEBUG_H */
