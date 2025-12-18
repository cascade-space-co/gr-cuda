#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#

from gnuradio import blocks
from gnuradio import gr
from gnuradio import cuda
import sys
import signal
from argparse import ArgumentParser
import time

class benchmark_multiply(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Benchmark Multiply", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        veclen = args.veclen
        self.actual_samples = actual_samples = int(nsamples /  veclen)
        num_blocks = args.nblocks
        self.k = 2.0

        ##################################################
        # Blocks
        ##################################################
        mult_blocks = []
        for i in range(num_blocks):
            mult_blocks.append(
                cuda.multiply_const_ff(self.k, veclen)
            )

        # Source
        self.blocks_null_source_0 = blocks.null_source(
            gr.sizeof_float*veclen)
            
        # Convert to cuda buffer type (load block handles this usually, but for pure multiply test we need to bridge)
        # However, multiply_const expects cuda_buffer input/output. 
        # We can use the 'load' block as the source/bridge to get data into cuda domain.
        self.loader = cuda.load(10, gr.sizeof_float*veclen, True)

        self.blocks_null_sink_0 = blocks.null_sink(
            gr.sizeof_float*veclen)
        self.blocks_head_0 = blocks.head(
            gr.sizeof_float*veclen, actual_samples)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_null_source_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.loader, 0))
        
        last_block = self.loader
        for i in range(num_blocks):
            self.connect((last_block, 0), (mult_blocks[i], 0))
            last_block = mult_blocks[i]

        self.connect((last_block, 0),
                     (self.blocks_null_sink_0, 0))


def main(top_block_cls=benchmark_multiply, options=None):

    parser = ArgumentParser(description='Run a flowgraph iterating over parameters for benchmarking multiply_const')
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=1e8)
    parser.add_argument('--veclen', type=int, default=1024)
    parser.add_argument('--nblocks', type=int, default=10)

    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    tb = top_block_cls(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("starting ...")
    startt = time.time()
    tb.start()

    tb.wait()
    endt = time.time()

    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')

if __name__ == '__main__':
    main()

