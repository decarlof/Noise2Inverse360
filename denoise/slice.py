#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Denoise a single CT slice using a trained Noise2Inverse model.
"""

import os
import yaml
import numpy as np
import torch
import tifffile
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from denoise.model import unet_ns_gn
from denoise.utils import save2img
from denoise.data_utils import extract_sliding_window_patches_25d, stitch_sliding_window_patches
from denoise import tiffs as tiffs_mod
from denoise import log


def run(args):

    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # 3D mode operates on full volumes — single-slice inference is not meaningful
    mode = getattr(args, 'mode', None) or params['train'].get('mode', '2.5d')
    if mode == '3d':
        raise RuntimeError(
            "The 'slice' command is not available in --mode 3d.\n"
            "3D denoising processes full volumes. Use:\n\n"
            "  denoise volume --config %s --mode 3d" % args.config
        )

    # create directory for denoised slices
    full_recon_name = params['dataset']['full_recon_name']
    base_name = full_recon_name[:-4] if full_recon_name.endswith('_rec') else full_recon_name
    out_path = params['dataset']['directory_to_reconstructions'] + '/' + base_name + '_denoised_slices'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # setup cuda device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in model
    n_slices = params['train']['n_slices']
    _ckpt_map = {'val': 'best_val_model.pth', 'lcl': 'best_lcl_model.pth', 'edge': 'best_edge_model.pth'}
    ckpt_name = _ckpt_map[getattr(args, 'checkpoint', 'lcl')]
    model_dir = getattr(args, 'model_dir', None) or \
        params['dataset']['directory_to_reconstructions'] + '/TrainOutput'
    path_to_mdl = os.path.join(model_dir, ckpt_name)
    log.info("Using checkpoint: %s" % ckpt_name)
    checkpoint = torch.load(path_to_mdl, map_location=torch.device('cpu'), weights_only=False)
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev)

    log.info("Loading slice %d" % args.slice_number)

    # path to data
    full_recon_path = params['dataset']['directory_to_reconstructions'] + '/' + params['dataset']['full_recon_name']

    # collect tiff files
    tiffs_collection = tiffs_mod.glob(full_recon_path)

    # supports 2.5D modeling
    S = len(tiffs_collection)
    left = n_slices // 2
    right = n_slices - 1 - left
    offsets = np.arange(-left, right + 1, dtype=int)
    idxs = args.slice_number + offsets
    idxs_mapped = np.clip(idxs, 0, S - 1)

    # get image slice
    list_of_images_to_process = [tiffs_collection[img_num] for img_num in idxs_mapped]

    # load in data
    images, _, _ = tiffs_mod.load_stack(list_of_images_to_process)
    images = torch.from_numpy(images[np.newaxis]).to(dev)

    # normalize image stack using training mean/std
    mean4norm = params['dataset']['mean4norm']
    std4norm = params['dataset']['std4norm']

    images = (images - mean4norm) / std4norm

    psz = params['train']['psz']
    patches, coords, meta = extract_sliding_window_patches_25d(
        images,
        patch_size=(psz, psz),
        overlap=params['infer']['overlap'],
        pad_mode="reflect",
        return_coords=True,
    )

    denoised_patches = torch.zeros((1, meta["P"], 1, psz, psz))

    # denoise image
    with torch.no_grad():
        for i in tqdm(range(patches.shape[1])):
            denoised = model(patches[:, i])
            denoised_patches[:, i] = denoised

    denoised = stitch_sliding_window_patches(
        denoised_patches, coords, meta, window=params['infer']['window']
    ).cpu().squeeze().numpy()

    # rescale back to original values
    denoised = denoised * std4norm + mean4norm

    # save denoised slice
    tifffile.imwrite(f'{out_path}/{args.slice_number:05d}.tiff', denoised)
    log.info("Saved denoised slice to %s/%05d.tiff" % (out_path, args.slice_number))
