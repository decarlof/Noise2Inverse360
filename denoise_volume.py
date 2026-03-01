from model import unet_ns_gn
import torch, argparse, os, time, sys, shutil, logging, yaml
from pathlib import Path
from data import TomoDatasetInfer
from data_utils import InferenceBatchSizeOptimizer
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiffs

import warnings
warnings.filterwarnings("ignore")

def main(args):
    
    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # setup output directory
    output_dir = params['dataset']['directory_to_reconstructions'] + '/' 'denoised_volume'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # setup cuda device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in model
    n_slices = params['train']['n_slices']
    path_to_mdl = params['dataset']['directory_to_reconstructions'] + '/' + 'TrainOutput' + '/' + 'best_lcl_model.pth'
    checkpoint = torch.load(path_to_mdl, map_location=torch.device('cpu'))
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev).eval()

    print("\nLoading data into CPU memory, it will take a while ... ...")

    # load in data
    ds_test = TomoDatasetInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
    print(f"\nLoaded in {ds_test.vol.shape[0]} slices of size {ds_test.vol.shape[1]}x{ds_test.vol.shape[2]}.\n")

    # determine maximum batch size given GPU system memory, model size, and image size
    optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=ds_test.vol[0].shape, device=dev, 
                                                     max_batch_size=512, precision='fp32')
    stats = optimal_batch_size.profile()
    mbsz = stats['optimal_batch_size']

    dl_test= DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,\
                          num_workers=4, drop_last=False, prefetch_factor=6, pin_memory=True)
    
    # initialize empty array for denoised volume
    preds = np.zeros((dl_test.dataset.total_patches, params['train']['psz'], params['train']['psz']))
    print(f'\nPatch volume size: {preds.shape[0]}x{preds.shape[1]}x{preds.shape[2]}')
    insert_cnt = 0

    # denoise volume
    print('\nProcessing data ...')
    with torch.no_grad():
        for X, _ in tqdm(dl_test):
            output = model(X.to(dev)).cpu().squeeze(dim=1).numpy()

            preds[insert_cnt:(insert_cnt+X.shape[0])] = output
            insert_cnt += X.shape[0]

    print('\nStitching denoised data ...')
    preds = ds_test.stitch_predictions(preds, window=params['infer']['window'], keep_k_dim=False)

    # rescale volume
    preds = preds*params['dataset']['std4norm'] + params['dataset']['mean4norm']

    # save volume
    print(f'\nSaving data ...')
    if len(args.start_slice) == 0:
        tiffs.save_stack(output_dir, preds)
    else:
        #Save the processed sub volume with the right tiff number
        tiffs.save_stack(output_dir, preds, offset=int(args.start_slice))


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Inference for 2.5D Noise2Inverse')
    parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-start_slice',type=str, default=0, help='minibatch size')    
    parser.add_argument('-end_slice',type=str, default=None, help='minibatch size')    
    parser.add_argument('-config', type=str, required=True, help='path to config yaml file')
    parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')


    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
    start_time = time.time()
    main(args)
    inference_time = time.time() - start_time
    print(f"\nInference Time: {inference_time:.4f} seconds\n")