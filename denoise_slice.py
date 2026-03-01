
from model import unet_ns_gn
import torch, argparse, os, time, sys, shutil, logging, yaml
import numpy as np
import tifffile
import tiffs
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from utils import save2img
from data_utils import extract_sliding_window_patches_25d, stitch_sliding_window_patches 

def main(args):

    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # create directory for denoised slices
    out_path = params['dataset']['directory_to_reconstructions'] + '/' 'denoised_slices'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # setup cuda device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in model
    n_slices = params['train']['n_slices']
    path_to_mdl = params['dataset']['directory_to_reconstructions'] + '/' + 'TrainOutput' + '/' + 'best_lcl_model.pth'
    checkpoint = torch.load(path_to_mdl, map_location=torch.device('cpu'))
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev)

    # get data
    print(f"\nLoading in slice {args.slice_number}.\n")

    # path to data
    full_recon_path = params['dataset']['directory_to_reconstructions'] + '/' + params['dataset']['full_recon_name']

    # collect tiff files
    tiffs_collection = tiffs.glob(full_recon_path)

    # supports 2.5D modeling
    S = len(tiffs_collection)
    left = n_slices // 2
    right = n_slices - 1 - left
    offsets = np.arange(-left, right + 1, dtype=int)  # length == num_slices
    idxs = args.slice_number + offsets
    idxs_mapped = np.clip(idxs, 0, S - 1)

    # get image slice
    list_of_images_to_process = [tiffs_collection[img_num] for img_num in idxs_mapped]

    # load in data
    images, _, _ = tiffs.load_stack(list_of_images_to_process)
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
    denoised = denoised*std4norm + mean4norm

    # save denoised slice
    tifffile.imwrite(f'{out_path}/{args.slice_number:05d}.tiff', denoised)

    #save2img(images[0, int(n_slices/2)].cpu().numpy(), f'_original_{args.slice_number:05d}.png')
    #save2img(denoised, f'_denoised_{args.slice_number:05d}.png')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Denoise CT slice with 2.5D N2I')
    parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-slice_number',type=int, required=True, help='test image')
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
        
    main(args)