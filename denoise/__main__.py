#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
denoise: Noise2Inverse CT denoising library.

Command-line interface for training the Noise2Inverse model and denoising
CT reconstructions using the trained model.

Usage
-----
Train the model (requires torchrun for distributed training)::

    (n2i) $ torchrun --nproc_per_node=2 -m denoise train \\
                --config baseline_config.yaml --gpus 0,1

Denoise a single CT slice::

    (n2i) $ denoise slice --config baseline_config.yaml --slice-number 500

Denoise the full CT volume::

    (n2i) $ denoise volume --config baseline_config.yaml

Denoise a sub-volume (slices 200 to 400)::

    (n2i) $ denoise volume --config baseline_config.yaml \\
                --start-slice 200 --end-slice 400
"""

import os
import sys
import datetime
import argparse
import logging

from denoise import log


def train(args):
    """
    Train the Noise2Inverse model using distributed data parallel (DDP).

    Must be launched via torchrun::

        torchrun --nproc_per_node=<N_GPUS> -m denoise train \\
            --config <yaml> [--gpus <gpu_ids>]

    Parameters
    ----------
    args.config : str
        Path to the YAML configuration file.
    args.gpus : str
        Comma-separated list of GPU IDs (e.g. ``0,1``).
    """
    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    logging.getLogger('matplotlib.font_manager').disabled = True

    log.info("Starting training with config: %s" % args.config)

    from denoise import train as train_mod
    train_mod.run(args)


def denoise_slice(args):
    """
    Denoise a single CT slice using the trained Noise2Inverse model.

    Parameters
    ----------
    args.config : str
        Path to the YAML configuration file.
    args.slice_number : int
        Index of the slice to denoise.
    args.gpus : str
        Comma-separated list of GPU IDs (e.g. ``0``).
    """
    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logging.getLogger('matplotlib.font_manager').disabled = True

    log.info("Denoising slice %d with config: %s" % (args.slice_number, args.config))

    from denoise import slice as slice_mod
    slice_mod.run(args)


def denoise_volume(args):
    """
    Denoise the entire CT volume (or a sub-volume) using the trained model.

    Parameters
    ----------
    args.config : str
        Path to the YAML configuration file.
    args.start_slice : str
        First slice index to process (empty string for the first slice).
    args.end_slice : str
        Last slice index to process (None for the last slice).
    args.gpus : str
        Comma-separated list of GPU IDs (e.g. ``0``).
    """
    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logging.getLogger('matplotlib.font_manager').disabled = True

    log.info("Denoising volume with config: %s" % args.config)
    if len(args.start_slice) > 0:
        log.info("Processing slices %s to %s" % (args.start_slice, args.end_slice))

    from denoise import volume as volume_mod
    volume_mod.run(args)


def main():
    home = os.path.expanduser("~")
    logs_home = os.path.join(home, 'logs')

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(
        logs_home,
        'denoise_' + datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H:%M:%S") + '.log'
    )
    log.setup_custom_logger(lfname)

    parser = argparse.ArgumentParser(
        prog='denoise',
        description='Noise2Inverse CT denoising library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # (command name, function, positional help text)
    cmd_parsers = [
        ('train',  train,          "Train the Noise2Inverse model (launch via torchrun for DDP)"),
        ('slice',  denoise_slice,  "Denoise a single CT slice"),
        ('volume', denoise_volume, "Denoise the entire CT volume"),
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, text in cmd_parsers:
        cmd_parser = subparsers.add_parser(
            cmd,
            help=text,
            description=text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        cmd_parser.add_argument(
            '--config',
            type=str,
            required=True,
            metavar='FILE',
            help='Path to the YAML configuration file',
        )
        cmd_parser.add_argument(
            '--gpus',
            type=str,
            default='0',
            metavar='IDS',
            help='Comma-separated list of visible GPU IDs',
        )

        if cmd == 'train':
            # no extra args: DDP parameters (LOCAL_RANK, RANK, WORLD_SIZE) are
            # injected automatically by torchrun as environment variables
            pass

        elif cmd == 'slice':
            cmd_parser.add_argument(
                '--slice-number',
                type=int,
                required=True,
                metavar='N',
                help='Index of the CT slice to denoise',
            )

        elif cmd == 'volume':
            cmd_parser.add_argument(
                '--start-slice',
                type=str,
                default='',
                metavar='N',
                help='Start slice index (default: first slice)',
            )
            cmd_parser.add_argument(
                '--end-slice',
                type=str,
                default=None,
                metavar='N',
                help='End slice index (default: last slice)',
            )

        cmd_parser.set_defaults(_func=func)

    args = parser.parse_args()

    if not hasattr(args, '_func'):
        parser.print_help()
        sys.exit(0)

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
