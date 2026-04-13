#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Denoise the entire CT volume using a trained Noise2Inverse model.
"""

import os
import time
import shutil
import yaml
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from denoise.model import unet_ns_gn
from denoise.model3d import unet3d
from denoise.data import TomoDatasetInfer
from denoise.data3d import TomoDataset3DInfer
from denoise.data_utils import InferenceBatchSizeOptimizer
from denoise import tiffs as tiffs_mod
from denoise import log


def run(args):

    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # Determine mode: CLI flag > YAML config > default 2.5d
    mode = getattr(args, 'mode', None) or params['train'].get('mode', '2.5d')

    # setup output directory
    full_recon_name = params['dataset']['full_recon_name']
    base_name = full_recon_name[:-4] if full_recon_name.endswith('_rec') else full_recon_name
    output_dir = params['dataset']['directory_to_reconstructions'] + '/' + base_name + '_denoised_volume_' + mode
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    log.info("Inference mode: %s" % mode)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    _ckpt_map = {'val': 'best_val_model.pth', 'lcl': 'best_lcl_model.pth', 'edge': 'best_edge_model.pth'}
    ckpt_name = _ckpt_map[getattr(args, 'checkpoint', 'lcl')]
    model_dir = getattr(args, 'model_dir', None) or \
        params['dataset']['directory_to_reconstructions'] + '/TrainOutput'
    path_to_mdl = os.path.join(model_dir, ckpt_name)
    log.info("Using checkpoint: %s" % ckpt_name)
    ckpt = torch.load(path_to_mdl, map_location='cpu', weights_only=False)

    if mode == '3d':
        n_blocks   = int(params['train'].get('n_blocks_3d', 3))
        start_filts = int(params['train'].get('start_filts_3d', 32))
        model = unet3d(in_channels=1, out_channels=1, n_blocks=n_blocks, start_filts=start_filts)
    else:
        n_slices = params['train']['n_slices']
        model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev).eval()

    log.info("Loading data into CPU memory, it will take a while ...")

    if mode == '3d':
        ds_test = TomoDataset3DInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
        psz_3d = ds_test.psz
        patch_shape = (psz_3d, psz_3d, psz_3d)

        optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=patch_shape, device=dev,
                                                         max_batch_size=64, precision='fp32', n_channels=1)
        stats = optimal_batch_size.profile()
        mbsz = stats['optimal_batch_size']

        dl_test = DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,
                             num_workers=2, drop_last=False, prefetch_factor=2, pin_memory=True)

        preds = np.zeros((len(ds_test), psz_3d, psz_3d, psz_3d), dtype=np.float32)
        insert_cnt = 0

        log.info("Processing %d 3D patches ..." % len(ds_test))
        with torch.no_grad():
            for X in tqdm(dl_test):
                out = model(X.to(dev)).cpu().squeeze(1).numpy()  # [B, psz, psz, psz]
                preds[insert_cnt:insert_cnt + X.shape[0]] = out
                insert_cnt += X.shape[0]

        log.info("Stitching 3D denoised volume ...")
        preds = ds_test.stitch_predictions(preds, window=params['infer'].get('window', 'hann'))

    else:
        ds_test = TomoDatasetInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
        log.info("Loaded %d slices of size %dx%d" % (ds_test.vol.shape[0], ds_test.vol.shape[1], ds_test.vol.shape[2]))

        patch_shape = (params['train']['psz'], params['train']['psz'])
        optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=patch_shape, device=dev,
                                                         max_batch_size=512, precision='fp32')
        stats = optimal_batch_size.profile()
        mbsz = stats['optimal_batch_size']

        dl_test = DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,
                             num_workers=4, drop_last=False, prefetch_factor=6, pin_memory=True)

        preds = np.zeros((dl_test.dataset.total_patches, params['train']['psz'], params['train']['psz']))
        log.info("Patch volume size: %dx%dx%d" % (preds.shape[0], preds.shape[1], preds.shape[2]))
        insert_cnt = 0

        log.info("Processing data ...")
        with torch.no_grad():
            for X, _ in tqdm(dl_test):
                output = model(X.to(dev)).cpu().squeeze(dim=1).numpy()
                preds[insert_cnt:(insert_cnt + X.shape[0])] = output
                insert_cnt += X.shape[0]

        log.info("Stitching denoised data ...")
        preds = ds_test.stitch_predictions(preds, window=params['infer']['window'], keep_k_dim=False)

    # Rescale back to original intensity range
    preds = preds * params['dataset']['std4norm'] + params['dataset']['mean4norm']

    # Save volume
    log.info("Saving data to %s ..." % output_dir)
    if len(args.start_slice) == 0:
        tiffs_mod.save_stack(output_dir, preds)
    else:
        tiffs_mod.save_stack(output_dir, preds, offset=int(args.start_slice))

    log.info("Done.")
