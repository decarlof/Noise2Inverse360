#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise2Inverse model training using distributed data parallel (DDP).
"""

import os
import time
import shutil
import yaml
import numpy as np
import torch
from copy import deepcopy
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from denoise.model import unet_ns_gn
from denoise.loss import LCL
from denoise.data import TomoDatasetTrain
from denoise.utils import save2img
from denoise.eval import laplacian_score_batch
from denoise import log


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):

    # read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    START_TIME = time.time()

    # setup distributed training using PyTorch's DDP framework
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_nccl_available():
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    # create directory containing training results in the directory of the reconstructions
    path_to_reconstructions = params['dataset']['directory_to_reconstructions']
    odir = path_to_reconstructions + '/' + 'TrainOutput'
    if rank == 0:
        if os.path.isdir(odir):
            shutil.rmtree(odir)
        os.mkdir(odir)
        os.mkdir(f'{odir}/results')

    torch.distributed.barrier()

    log.info("local rank %d (global rank %d) of a world size %d started" % (local_rank, rank, world_size))

    torch.cuda.set_device(local_rank)

    log.info("Loading data into CPU memory, it will take a while ...")
    ds_train = TomoDatasetTrain(params=params, config_file=args.config)
    train_sampler = DistributedSampler(dataset=ds_train, shuffle=True, drop_last=True)
    dl_train = DataLoader(dataset=ds_train, batch_size=params['train']['mbsz'], sampler=train_sampler,
                          num_workers=4, drop_last=False, prefetch_factor=params['train']['mbsz'], pin_memory=True)

    log.info("Loaded %d samples into CPU memory for training." % len(ds_train))

    # initialize model from scratch
    n_slices = params['train']['n_slices']
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['lr'])
    log.info('Initializing model from scratch')

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    log.info("Number of model parameters: %s" % f"{count_parameters(model):,}")
    model_updates = 0

    # loss functions and warmup criteria
    criterion = torch.nn.L1Loss()
    criterion_lcl = LCL()
    beta = .01
    warmup = params['train']['warmup']
    continue_warmup = True

    train_loss, val_loss = [], []
    edge_values = []
    train_lcl_loss, val_lcl_loss = [], []

    best_val_loss, best_edge, best_lcl_loss = np.inf, 0, np.inf
    best_val_epoch, best_edge_epoch, best_lcl_epoch = 0, 0, 0

    # start training
    for epoch in range(1, params['train']['maxep'] + 1):

        step_losses, step_val_losses, step_lcl_loss, step_lcl_val_loss, step_edge_values = [], [], [], [], []

        tick_ep = time.time()

        model.train()
        dl_train.sampler.set_epoch(epoch)
        # training loop
        for X_mb, Y_mb in dl_train:
            optimizer.zero_grad()

            X_mb_dev = X_mb.cuda()
            Y_mb_dev = Y_mb.cuda()

            if model_updates <= warmup:
                optimizer.zero_grad()

                # Process first view
                pred_view1 = model(X_mb_dev)
                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices // 2)])
                loss_view1.backward()
                optimizer.step()

                optimizer.zero_grad()

                # Process second view
                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices // 2)])
                loss_view2.backward()
                optimizer.step()

                loss_lcl1 = torch.tensor(0.)
                loss_lcl2 = torch.tensor(0.)

            else:
                optimizer.zero_grad()

                # Process first view
                pred_view1 = model(X_mb_dev)
                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices // 2)])
                loss_lcl1 = criterion_lcl(pred_view1) * beta
                loss_total1 = loss_view1 + loss_lcl1
                loss_total1.backward()
                optimizer.step()

                optimizer.zero_grad()

                # Process second view
                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices // 2)])
                loss_lcl2 = criterion_lcl(pred_view2) * beta
                loss_total2 = loss_view2 + loss_lcl2
                loss_total2.backward()
                optimizer.step()

            loss = loss_view1 + loss_view2
            step_losses.append(loss.detach().cpu().numpy())
            loss_lcl = loss_lcl1 + loss_lcl2
            step_lcl_loss.append(loss_lcl.detach().cpu().numpy())
            model_updates += 1

        model.eval()
        with torch.no_grad():
            # validation loop
            for X_mb, Y_mb in dl_train:
                X_mb_dev = X_mb.cuda()
                Y_mb_dev = Y_mb.cuda()

                pred_view1 = model(X_mb_dev)
                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices // 2)])
                loss_lcl1 = criterion_lcl(pred_view1) * beta

                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices // 2)])
                loss_lcl2 = criterion_lcl(pred_view2) * beta

                loss = loss_view1 + loss_view2

                lap_score = laplacian_score_batch(pred_view1.cpu()) + laplacian_score_batch(pred_view2.cpu())
                step_edge_values.append(lap_score)

                step_val_losses.append(loss.detach().cpu().numpy())
                step_lcl_val_loss.append(loss_lcl.cpu().numpy())

        if rank != world_size - 1:
            continue

        ep_time = time.time() - tick_ep
        log.info('Epoch %d' % epoch)
        log.info('[Train] L1 loss:    %.6f, %.6f => %.6f, rate: %.2fs/ep' % (
            np.mean(step_losses), step_losses[0], step_losses[-1], ep_time))
        log.info('[Train] LCL loss:   %.6f, %.6f => %.6f, rate: %.2fs/ep' % (
            np.mean(step_lcl_loss), step_lcl_loss[0], step_lcl_loss[-1], ep_time))
        log.info('[Val]   L1 loss:    %.6f, %.6f => %.6f, rate: %.2fs/ep' % (
            np.mean(step_val_losses), step_val_losses[0], step_val_losses[-1], ep_time))
        log.info('[Val]   LCL loss:   %.6f, %.6f => %.6f, rate: %.2fs/ep' % (
            np.mean(step_lcl_val_loss), step_lcl_val_loss[0], step_lcl_val_loss[-1], ep_time))
        log.info('[Val] EDGE Value:   %.4f, %.4f => %.4f, rate: %.2fs/ep' % (
            np.mean(step_edge_values), step_edge_values[0], step_edge_values[-1], ep_time))

        train_loss.append(np.mean(step_losses))
        val_loss.append(np.mean(step_val_losses))
        train_lcl_loss.append(np.mean(step_lcl_loss))
        val_lcl_loss.append(np.mean(step_lcl_val_loss))
        edge_values.append(np.mean(step_edge_values))

        # Save the best model with the lowest lcl loss
        if np.mean(step_lcl_val_loss) < best_lcl_loss:
            best_lcl_loss = np.mean(step_lcl_val_loss)
            best_lcl_epoch = epoch
            mdl_fname = f"{odir}/best_lcl_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        # Save the best model with the lowest val loss
        if np.mean(step_val_losses) < best_val_loss:
            best_val_loss = np.mean(step_val_losses)
            best_val_epoch = epoch
            mdl_fname = f"{odir}/best_val_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        # Save the best model with the highest edge value
        if np.mean(step_edge_values) > best_edge:
            best_edge = np.mean(step_edge_values)
            best_edge_epoch = epoch
            mdl_fname = f"{odir}/best_edge_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        # Warm up period
        if model_updates > warmup and continue_warmup:
            best_edge, best_lcl_loss = 0, np.inf
            best_edge_epoch, best_lcl_epoch = 0, 0
            continue_warmup = False

        CRNT_TIME = time.time()
        log.info("[Info]  Training Time: %.2f seconds" % (CRNT_TIME - START_TIME))

        # option to view the denoising process during training
        if epoch % 5 == 0:
            ridx = np.random.randint(pred_view1.shape[0])
            save2img(pred_view1[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view1.png' % (odir, epoch))
            save2img(pred_view2[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view2.png' % (odir, epoch))
            save2img(X_mb_dev[ridx, int(n_slices // 2)].detach().cpu().numpy(), '%s/results/_%d_view1.png' % (odir, epoch))
            save2img(Y_mb_dev[ridx, int(n_slices // 2)].detach().cpu().numpy(), '%s/results/_%d_view2.png' % (odir, epoch))

        # Keep track of when/where the best model is
        log.info('Lowest model validation loss %.6f at epoch %d' % (best_val_loss, best_val_epoch))
        log.info('Lowest model LCL loss %.6f at epoch %d' % (best_lcl_loss, best_lcl_epoch))
        log.info('Highest model EDGE score %.6f at epoch %d' % (best_edge, best_edge_epoch))
        log.info('Number of model updates: %s' % f"{model_updates:,}")
        log.info('Is model warming up?: %s' % continue_warmup)

        # View the training/validation loss during training
        if epoch % 5 == 0:
            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(train_loss[:], label="Training Loss")
            plt.plot(val_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_training.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(train_lcl_loss[:], label="Training Loss")
            plt.plot(val_lcl_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_lcl_training.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(edge_values[:])
            plt.xlabel("Epoch")
            plt.ylabel("EDGE Gradient")
            plt.savefig(f'{odir}/results/__edge_training.png')
            plt.close()
