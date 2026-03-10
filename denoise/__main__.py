#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
denoise: Noise2Inverse CT denoising library.

Command-line interface for training the Noise2Inverse model and denoising
CT reconstructions using the trained model.

Usage
-----
Create sub-reconstructions and config (run in the tomocupy environment)::

    (tomocupy) $ denoise prepare --out-path-name /data/exp_rec \\
                     --file-name raw.h5 --energy 50 --pixel-size 1.17

Train the model::

    (denoise) $ denoise train --config /data/exp_rec_config.yaml --gpus 0,1

Denoise a single CT slice::

    (denoise) $ denoise slice --config baseline_config.yaml --slice-number 500

Denoise the full CT volume::

    (denoise) $ denoise volume --config baseline_config.yaml

Denoise a sub-volume (slices 200 to 400)::

    (denoise) $ denoise volume --config baseline_config.yaml \\
                --start-slice 200 --end-slice 400
"""

import os
import sys
import datetime
import argparse
import logging

from denoise import log


def prepare(args):
    """
    Create the two Noise2Inverse sub-reconstructions using tomocupy and write
    a ready-to-use configuration file.

    Calls ``tomocupy recon`` twice — once with even-indexed projections
    (``--start-proj 0 --proj-step 2``) and once with odd-indexed projections
    (``--start-proj 1 --proj-step 2``) — then writes a ``denoise`` config file
    next to the reconstruction directories.

    Parameters
    ----------
    args.out_path_name : str
        Base output path for the full reconstruction (the tomocupy
        ``--out-path-name`` value).  The sub-reconstructions are written to
        ``<out_path_name>_0`` and ``<out_path_name>_1``.
    args.tomocupy_args : list of str
        All remaining tomocupy recon arguments passed through verbatim
        (e.g. ``--file-name``, ``--energy``, ``--pixel-size`` …).
        Do **not** include ``--start-proj``, ``--proj-step``, or
        ``--out-path-name`` — they are set automatically.

    Example
    -------
    ::

        (tomocupy) $ denoise prepare --out-path-name /data/exp_rec \\
                         --file-name raw.h5 --energy 50 --pixel-size 1.17
    """
    import subprocess
    import pathlib
    import yaml

    out_path   = pathlib.Path(args.out_path_name)
    parent_dir = out_path.parent
    rec_name   = out_path.name
    extra      = args.tomocupy_args  # passthrough list

    # Reject forbidden flags
    forbidden = {'--start-proj', '--proj-step', '--out-path-name'}
    for flag in extra:
        if flag in forbidden:
            log.error("Do not include %s in extra args — it is set automatically." % flag)
            raise RuntimeError("Forbidden flag: %s" % flag)

    for idx, label in ((0, 'even'), (1, 'odd')):
        log.info("Sub-reconstruction %d (%s projections) ..." % (idx, label))
        cmd = ['tomocupy', 'recon'] + extra + [
            '--start-proj', str(idx),
            '--proj-step',  '2',
            '--out-path-name', str(out_path) + '_%d' % idx,
        ]
        subprocess.run(cmd, check=True)

    # --- instrument metadata from the raw HDF5 file ---
    hdf_keys = [
        '/process/acquisition/start_date',
        '/measurement/sample/experimenter/name',
        '/measurement/instrument/source/beamline',
        '/measurement/instrument/source/current',
        '/measurement/instrument/monochromator/energy',
        '/measurement/instrument/monochromator/mode',
        '/measurement/instrument/detection_system/scintillator/type',
        '/measurement/instrument/detection_system/scintillator/active_thickness',
        '/measurement/instrument/detection_system/objective/magnification',
        '/measurement/instrument/detection_system/objective/resolution',
        '/measurement/instrument/detector/manufacturer',
        '/measurement/instrument/detector/model',
        '/measurement/instrument/detector/serial_number',
        '/measurement/instrument/detector/exposure_time',
        '/measurement/instrument/detector/temperature',
        '/measurement/instrument/detector/binning_x',
        '/measurement/instrument/detector/binning_y',
        '/measurement/instrument/detector_motor_stack/setup/z',
    ]
    metadata = {}
    # Extract --file-name from the passthrough args
    h5_file = None
    for i, tok in enumerate(extra):
        if tok == '--file-name' and i + 1 < len(extra):
            h5_file = extra[i + 1]
            break
    if h5_file is not None:
        try:
            import meta as meta_lib
            mp = meta_lib.read_meta.Hdf5MetadataReader(h5_file)
            meta_dict = mp.readMetadata()
            mp.close()
            for hdf_path in hdf_keys:
                if hdf_path not in meta_dict:
                    continue
                val   = meta_dict[hdf_path][0]
                units = meta_dict[hdf_path][1]
                key   = hdf_path.split('/')[-1]
                if hdf_path == '/measurement/instrument/detector_motor_stack/setup/z':
                    key = 'propagation_distance'
                if key == 'mode':
                    metadata[key] = {0: 'mono', 1: 'pink', 2: 'white'}.get(int(val), str(val))
                elif units is None or isinstance(val, str):
                    metadata[key] = val
                else:
                    metadata[key] = '%s %s' % (val, units)
            log.info("Instrument metadata read from: %s" % h5_file)
        except ImportError:
            log.warning("'meta' library not installed — skipping metadata block.")
        except Exception as exc:
            log.warning("Could not read metadata from %s: %s" % (h5_file, exc))
    else:
        log.warning("--file-name not found in tomocupy args — skipping metadata block.")

    config = {
        'dataset': {
            'directory_to_reconstructions': str(parent_dir),
            'sub_recon_name0': '%s_0' % rec_name,
            'sub_recon_name1': '%s_1' % rec_name,
            'full_recon_name': rec_name,
        },
        'train':  {'psz': 256, 'n_slices': 5, 'mbsz': 32, 'lr': 0.001,
                   'warmup': 2000, 'maxep': 2000},
        'infer':  {'overlap': 0.5, 'window': 'cosine'},
    }
    if metadata:
        config['metadata'] = metadata
    config_path = parent_dir / ('%s_config.yaml' % rec_name)
    with open(config_path, 'w') as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    log.info("Config written to: %s" % config_path)
    log.info(
        "Next step:\n"
        "  conda activate denoise\n"
        "  denoise train --config %s --gpus 0,1" % config_path
    )


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

    # --- prepare (does not share --config / --gpus with the other commands) ---
    prep_parser = subparsers.add_parser(
        'prepare',
        help='Create Noise2Inverse (N2I) sub-reconstructions with tomocupy and write a config file',
        description='Create Noise2Inverse (N2I) sub-reconstructions with tomocupy and write a config file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prep_parser.add_argument(
        '--out-path-name',
        type=str,
        required=True,
        metavar='PATH',
        help='Base output path of the full reconstruction (tomocupy --out-path-name)',
    )
    prep_parser.add_argument(
        'tomocupy_args',
        nargs=argparse.REMAINDER,
        help='All other tomocupy recon arguments passed through verbatim',
    )
    prep_parser.set_defaults(_func=prepare)

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
