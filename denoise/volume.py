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
from denoise.data import TomoDatasetInfer
from denoise.data_utils import InferenceBatchSizeOptimizer
from denoise import tiffs as tiffs_mod
from denoise import log


def run(args):

    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # setup output directory
    full_recon_name = params['dataset']['full_recon_name']
    base_name = full_recon_name[:-4] if full_recon_name.endswith('_rec') else full_recon_name
    output_dir = params['dataset']['directory_to_reconstructions'] + '/' + base_name + '_denoised_volume'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # setup cuda device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in model
    n_slices = params['train']['n_slices']
    _ckpt_map = {'val': 'best_val_model.pth', 'lcl': 'best_lcl_model.pth', 'edge': 'best_edge_model.pth'}
    ckpt_name = _ckpt_map[getattr(args, 'checkpoint', 'lcl')]
    path_to_mdl = params['dataset']['directory_to_reconstructions'] + '/' + 'TrainOutput' + '/' + ckpt_name
    log.info("Using checkpoint: %s" % ckpt_name)
    checkpoint = torch.load(path_to_mdl, map_location=torch.device('cpu'), weights_only=False)
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev).eval()

    log.info("Loading data into CPU memory, it will take a while ...")

    # load in data
    ds_test = TomoDatasetInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
    log.info("Loaded %d slices of size %dx%d" % (ds_test.vol.shape[0], ds_test.vol.shape[1], ds_test.vol.shape[2]))

    # determine maximum batch size given GPU system memory, model size, and patch size
    # NOTE: use patch size (psz, psz), not full slice size, so the optimizer profiles
    # the actual inputs the model receives during inference.
    patch_shape = (params['train']['psz'], params['train']['psz'])
    optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=patch_shape, device=dev,
                                                     max_batch_size=512, precision='fp32')
    stats = optimal_batch_size.profile()
    mbsz = stats['optimal_batch_size']

    dl_test = DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,
                         num_workers=4, drop_last=False, prefetch_factor=6, pin_memory=True)

    # initialize empty array for denoised volume
    preds = np.zeros((dl_test.dataset.total_patches, params['train']['psz'], params['train']['psz']))
    log.info("Patch volume size: %dx%dx%d" % (preds.shape[0], preds.shape[1], preds.shape[2]))
    insert_cnt = 0

    # denoise volume
    log.info("Processing data ...")
    with torch.no_grad():
        for X, _ in tqdm(dl_test):
            output = model(X.to(dev)).cpu().squeeze(dim=1).numpy()
            preds[insert_cnt:(insert_cnt + X.shape[0])] = output
            insert_cnt += X.shape[0]

    log.info("Stitching denoised data ...")
    preds = ds_test.stitch_predictions(preds, window=params['infer']['window'], keep_k_dim=False)

    # rescale volume
    preds = preds * params['dataset']['std4norm'] + params['dataset']['mean4norm']

    # save volume
    log.info("Saving data to %s ..." % output_dir)
    if len(args.start_slice) == 0:
        tiffs_mod.save_stack(output_dir, preds)
    else:
        # Save the processed sub volume with the right tiff number
        tiffs_mod.save_stack(output_dir, preds, offset=int(args.start_slice))

    log.info("Done.")
