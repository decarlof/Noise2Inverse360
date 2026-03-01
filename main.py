
from model import unet_ns_gn
from loss import LCL
import torch, argparse, os, time, sys, shutil, logging, yaml
from data import TomoDatasetTrain
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from copy import deepcopy
from utils import save2img
from eval import laplacian_score_batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):

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
    
    # create output log
    logging.basicConfig(filename=f'{odir}/Noise2Inverse360.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"local rank {local_rank} (global rank {rank}) of a world size {world_size} started")

    torch.cuda.set_device(local_rank)

    logging.info("\nLoading data into CPU memory, it will take a while ... ...")
    ds_train = TomoDatasetTrain(params=params, config_file=args.config)
    train_sampler = DistributedSampler(dataset=ds_train, shuffle=True, drop_last=True)
    dl_train = DataLoader(dataset=ds_train, batch_size=params['train']['mbsz'], sampler=train_sampler,\
                          num_workers=4, drop_last=False, prefetch_factor=params['train']['mbsz'], pin_memory=True)

    logging.info(f"\nLoaded %d samples, {ds_train.samples}, into CPU memory for training." % (len(ds_train), ))

    # initialized model from scratch 
    n_slices = params['train']['n_slices']
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = params['train']['lr'])
    logging.info('\nInitializing model from scratch\n')
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print(f"Number of model parameters: {count_parameters(model):,}")
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
    for epoch in range(1, params['train']['maxep']+1):

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

                #Process first view
                pred_view1 = model(X_mb_dev)

                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices//2)])
                
                loss_view1.backward()
                optimizer.step()
                
                optimizer.zero_grad()

                #Process second view
                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices//2)])

                loss_view2.backward()
                optimizer.step()

                loss_lcl1 = torch.tensor(0.)
                loss_lcl2 = torch.tensor(0.)

            else:
                optimizer.zero_grad()

                #Process first view
                pred_view1 = model(X_mb_dev)

                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices//2)])
                loss_lcl1 = criterion_lcl(pred_view1)*beta
                loss_total1 = loss_view1 + loss_lcl1

                loss_total1.backward()
                optimizer.step()
                
                optimizer.zero_grad()

                #Process second view
                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices//2)])
                loss_lcl2 = criterion_lcl(pred_view2)*beta
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
                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, int(n_slices//2)])
                loss_lcl1 = criterion_lcl(pred_view1)*beta
                
                pred_view2 = model(Y_mb_dev)
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, int(n_slices//2)])
                loss_lcl2 = criterion_lcl(pred_view2)*beta

                loss = loss_view1 + loss_view2

                lap_score = laplacian_score_batch(pred_view1.cpu()) + laplacian_score_batch(pred_view2.cpu())
                step_edge_values.append(lap_score)

                step_val_losses.append(loss.detach().cpu().numpy())
                step_lcl_val_loss.append(loss_lcl.cpu().numpy())

        if rank != world_size-1: continue

        ep_time = time.time() - tick_ep
        logging.info(f'\nEpoch {epoch}')
        iter_prints = f'[Train] L1 loss:    {np.mean(step_losses):.6f}, {step_losses[0]:.6f} => {step_losses[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)
        iter_prints = f'[Train] LCL loss:   {np.mean(step_lcl_loss):.6f}, {step_lcl_loss[0]:.6f} => {step_lcl_loss[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)
        iter_prints = f'[Val]   L1 loss:    {np.mean(step_val_losses):.6f}, {step_val_losses[0]:.6f} => {step_val_losses[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)
        iter_prints = f'[Val]   LCL loss:   {np.mean(step_lcl_val_loss):.6f}, {step_lcl_val_loss[0]:.6f} => {step_lcl_val_loss[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)
        iter_prints = f'[Val] EDGE Value:   {np.mean(step_edge_values):.4f}, {step_edge_values[0]:.4f} => {step_edge_values[-1]:.4f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)

        train_loss.append(np.mean(step_losses))
        val_loss.append(np.mean(step_val_losses))
        train_lcl_loss.append(np.mean(step_lcl_loss))
        val_lcl_loss.append(np.mean(step_lcl_val_loss))
        edge_values.append(np.mean(step_edge_values))
        
        #Save the best model with the lowest lcl loss
        if np.mean(step_lcl_val_loss) < best_lcl_loss:
            best_lcl_loss = np.mean(step_lcl_val_loss)
            best_lcl_epoch= epoch
            mdl_fname = f"{odir}/best_lcl_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        #Save the best model with the lowest val loss
        if np.mean(step_val_losses) < best_val_loss:
            best_val_loss = np.mean(step_val_losses)
            best_val_epoch = epoch
            mdl_fname = f"{odir}/best_val_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        #Save the best model with the highest edge value
        if np.mean(step_edge_values) > best_edge:
            best_edge = np.mean(step_edge_values)
            best_edge_epoch = epoch
            mdl_fname = f"{odir}/best_edge_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        #Warm up period
        if model_updates > warmup and continue_warmup:

                best_edge, best_lcl_loss = 0, np.inf
                best_edge_epoch, best_lcl_epoch = 0, 0
                
                continue_warmup = False

        CRNT_TIME = time.time()
        logging.info(f"[Info]  Training Time: {CRNT_TIME-START_TIME:.2f} seconds")

        
        #option to view the denoising process during training
        if epoch % 5 == 0:
            ridx = np.random.randint(pred_view1.shape[0])
            
            save2img(pred_view1[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view1.png' % (odir, epoch))
            save2img(pred_view2[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view2.png' % (odir, epoch))
            save2img(X_mb_dev[ridx, int(n_slices//2)].detach().cpu().numpy(), '%s/results/_%d_view1.png' % (odir, epoch))
            save2img(Y_mb_dev[ridx, int(n_slices//2)].detach().cpu().numpy(), '%s/results/_%d_view2.png' % (odir, epoch))

        #Keep track of when/where the best model is
        logging.info(f'Lowest model validation loss {best_val_loss:.6f} at epoch {best_val_epoch}')
        logging.info(f'Lowest model LCL loss {best_lcl_loss:.6f} at epoch {best_lcl_epoch}')
        logging.info(f'Highest model EDGE score {best_edge:.6f} at epoch {best_edge_epoch}')
        logging.info(f'Number of model updates: {model_updates:,}')
        logging.info(f'Is model warming up?: {continue_warmup}')

        #View the training/validation loss during training
        if epoch % 5 == 0:
            plt.figure(figsize=(12,8))
            plt.title("Training Progress")
            plt.plot(train_loss[:], label="Training Loss")
            plt.plot(val_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_training.png')
            plt.close()

            plt.figure(figsize=(12,8))
            plt.title("Training Progress")
            plt.plot(train_lcl_loss[:], label="Training Loss")
            plt.plot(val_lcl_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_lcl_training.png')
            plt.close()

            plt.figure(figsize=(12,8))
            plt.title("Training Progress")
            plt.plot(edge_values[:])
            plt.xlabel("Epoch")
            plt.ylabel("EDGE Gradient")
            plt.savefig(f'{odir}/results/__edge_training.png')
            plt.close()


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Noise2Inverse with 2.5D')
    parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
    parser.add_argument('--local_rank', type=int, help='local rank for DDP')
    parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')
    parser.add_argument('-config', type=str, required=True, help='path to config yaml file')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ...' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logical_cpus = os.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(logical_cpus)

    logging.getLogger('matplotlib.font_manager').disabled = True

    main(args)