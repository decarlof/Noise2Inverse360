#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
denoise: Noise2Inverse CT denoising library.

Command-line interface for training the Noise2Inverse model and denoising
CT reconstructions using the trained model.

Usage
-----
Train the model::

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1

Denoise a single CT slice::

    (denoise) $ denoise slice --config /data/sample_rec_config.yaml --slice-number 500

Denoise the full CT volume::

    (denoise) $ denoise volume --config /data/sample_rec_config.yaml

Denoise a sub-volume (slices 200 to 400)::

    (denoise) $ denoise volume --config /data/sample_rec_config.yaml \\
                --start-slice 200 --end-slice 400
"""

import os
import sys
import datetime
import argparse
import logging

from denoise import log



def _print_registry_matches(matches):
    """Print a formatted table of registry search results."""
    log.warning("Registry search found %d matching model(s):" % len(matches))
    for i, m in enumerate(matches):
        meta = m['config'].get('metadata', {})
        log.warning(
            "  [%d] %s  (%d/%d criteria match — %.0f%%)" % (
                i + 1, m['dir'].name, m['matched'], m['total'],
                100 * m['score'],
            )
        )
        for key in ('beamline', 'mode', 'energy', 'type', 'serial_number',
                    'exposure_time', 'binning_x', 'binning_y', 'temperature'):
            if key in meta:
                log.warning("       %-20s %s" % (key + ':', meta[key]))
        log.warning("       %-20s %s" % ('registry path:', m['dir']))


def train(args):
    """
    Train the Noise2Inverse model using distributed data parallel (DDP).

    When called directly (``denoise train``), this function automatically
    re-launches itself via ``torchrun`` with ``PYTHONNOUSERSITE=1``, so no
    manual ``torchrun`` invocation is required.

    Parameters
    ----------
    args.config : str
        Path to the YAML configuration file.
    args.gpus : str
        Comma-separated list of GPU IDs (e.g. ``0,1``).
    args.no_search : bool
        Skip registry search before training.
    """
    if 'LOCAL_RANK' not in os.environ:
        # Not inside a torchrun context yet — optionally search registry,
        # then re-launch via torchrun.
        if not getattr(args, 'no_search', False):
            from denoise import registry as reg
            matches = reg.search(args.config)
            if matches:
                _print_registry_matches(matches)
                log.warning(
                    "A compatible model may already exist. "
                    "Copy the registry path above as --model-dir for slice/volume inference."
                )
                answer = input(
                    "\nTrain a new model anyway? [y/N] "
                ).strip().lower()
                if answer not in ('y', 'yes'):
                    log.info("Training cancelled. Use an existing model from the registry.")
                    sys.exit(0)
            else:
                all_entries = reg.list_registry()
                if all_entries:
                    log.info("Registry search: no matching models found. Existing models:")
                    for entry_name, _ in all_entries:
                        log.info("  - %s" % entry_name)
                else:
                    log.info("Registry search: registry is empty.")
                answer = input(
                    "\nProceed with new training? [Y/n] "
                ).strip().lower()
                if answer in ('n', 'no'):
                    log.info("Training cancelled.")
                    sys.exit(0)

        import subprocess
        n_gpus = len(args.gpus.split(',')) if args.gpus else 1
        env = {**os.environ, 'PYTHONNOUSERSITE': '1'}
        if args.gpus:
            env['CUDA_VISIBLE_DEVICES'] = args.gpus
        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--nproc_per_node', str(n_gpus),
            '-m', 'denoise', 'train',
            '--config', args.config,
            '--gpus', args.gpus,
        ]
        if getattr(args, 'resume', False):
            cmd.append('--resume')
        log.info("Launching training via torchrun (%d GPU(s)) ..." % n_gpus)
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)

    # Inside torchrun context — proceed with training.
    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    logging.getLogger('matplotlib.font_manager').disabled = True

    log.info("Starting training with config: %s" % args.config)

    from denoise import train as train_mod
    train_mod.run(args)


def register_model(args):
    """Register a trained model in the local registry."""
    import pathlib
    from denoise import registry as reg

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s" % cfg_path)
        sys.exit(1)

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.is_dir():
        log.error("Model directory not found: %s" % model_dir)
        sys.exit(1)

    entry_dir, copied = reg.register(cfg_path, model_dir, name=args.name)
    if not copied:
        log.error("No checkpoint files found in %s" % model_dir)
        sys.exit(1)

    log.info("Registered model at: %s" % entry_dir)
    log.info("Checkpoints copied: %s" % ', '.join(copied))
    log.info("Registry: %s" % reg.REGISTRY_DIR)


def search_registry(args):
    """Search the registry for models matching the config's noise fingerprint."""
    import pathlib
    from denoise import registry as reg

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s" % cfg_path)
        sys.exit(1)

    matches = reg.search(cfg_path)
    if not matches:
        log.info("No matching models found in registry (%s)." % reg.REGISTRY_DIR)
        return

    _print_registry_matches(matches)


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
    # Prevent user-local packages in ~/.local/ from shadowing the conda
    # environment (e.g. old typing_extensions breaking pydantic/albumentations).
    if not os.environ.get('PYTHONNOUSERSITE'):
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

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
        ('train',  train,          "Train the Noise2Inverse model"),
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
            cmd_parser.add_argument(
                '--resume',
                action='store_true',
                default=False,
                help='Resume training from the last completed epoch (requires resume.pth in TrainOutput/)',
            )
            cmd_parser.add_argument(
                '--no-search',
                action='store_true',
                default=False,
                help='Skip registry search before training',
            )

        elif cmd == 'slice':
            cmd_parser.add_argument(
                '--slice-number',
                type=int,
                required=True,
                metavar='N',
                help='Index of the CT slice to denoise',
            )
            cmd_parser.add_argument(
                '--checkpoint',
                type=str,
                default='lcl',
                choices=['val', 'lcl', 'edge'],
                help='Checkpoint to use: val=lowest val loss, lcl=lowest LCL loss, edge=highest edge score',
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
            cmd_parser.add_argument(
                '--checkpoint',
                type=str,
                default='lcl',
                choices=['val', 'lcl', 'edge'],
                help='Checkpoint to use: val=lowest val loss, lcl=lowest LCL loss, edge=highest edge score',
            )

        cmd_parser.set_defaults(_func=func)

    # --- register ---
    reg_parser = subparsers.add_parser(
        'register',
        help='Register a trained model in the local registry (~/.denoise/registry/)',
        description='Register a trained model in the local registry for later reuse.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reg_parser.add_argument(
        '--config',
        type=str,
        required=True,
        metavar='FILE',
        help='Path to the YAML configuration file used for training',
    )
    reg_parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        metavar='DIR',
        help='Directory containing the trained checkpoints (TrainOutput/)',
    )
    reg_parser.add_argument(
        '--name',
        type=str,
        default=None,
        metavar='NAME',
        help='Registry entry name (auto-generated from metadata if omitted)',
    )
    reg_parser.set_defaults(_func=register_model)

    # --- search ---
    srch_parser = subparsers.add_parser(
        'search',
        help='Search the registry for models matching a config noise fingerprint',
        description='Search the registry for models trained under compatible instrument conditions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    srch_parser.add_argument(
        '--config',
        type=str,
        required=True,
        metavar='FILE',
        help='Path to the YAML configuration file to match against',
    )
    srch_parser.set_defaults(_func=search_registry)

    args, unknown = parser.parse_known_args()

    if not hasattr(args, '_func'):
        parser.print_help()
        sys.exit(0)

    if unknown:
        parser.error('unrecognized arguments: %s' % ' '.join(unknown))

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
