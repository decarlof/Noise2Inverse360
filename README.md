# 2.5D Noise2Inverse for Denoising CT Data    

## Overview    
This project provides an implementation of the Noise2Inverse (N2I) framework for denoising CT data without requiring ground truth images. It implements the
N2I framework following the 2.5D approach utilizing adjacent slices in the deep learning model. This project utilizes a simple U-Net model with leaky relu
and group norm as the CNN model for denoising.

## Assumptions    
-All output of this project, training results, inference results, trained models are saved inside the directory of the reconstructions. For example:
    -User: John Smith
        -Sample 1 Directory:
            -Provided by the User:
                -Full Reconstruction (Directory)
                -Sub-Reconstruction 1 (Directory)
                -Sub Reconstruction 2 (Directory)
            -Provided by N2I:
                -config.yaml 
                -TrainOutput (Directory)
                -denoised_slices (Directory)
                -denoised_volume (Directory)
-Data for training/inference is already created
    -This project does not contain the capacity to generate the necessary reconstructions for training/inference
-Data for training/inference is saved as .tiff files
    -Can be either .tif or .tiff
-Model type/size is the same for each dataset
    -We've found the U-Net model with no skip connections, leaky relu, and group norm to be a robust choice for different samples
-Model can be used for inference even when training is still running

## Features    
-Automatic batch size optimization for A100/V100 GPUs
    -CT image size, GPU system memory, and model size all contribute to how much GPU RAM is used. Therefore, we've included a feature that autmatically determines 
    the maximum batch size to reduce the chance of a OOM error.
-Support for 2.5D inference with PyTorch
-Flexible plug-and-play for different samples across different users

## Installation Instructions    
-This project utilizes miniconda with a dedicated virtual environment

    -conda env create -f n2i_environment.yml

-Packages include:
    -albumentations
        -Used for data augmentation during training
    -pytorch (2.4.0)
    -cuda (11.8)
    -tifffile
    -tqdm
    -matplotlib
    -skimage

## Project Structure    
N2I
    -Python Files
        -data.py
        -data_util.py
        -denoise_slice.py
        -denoise_volume.py
        -eval.py
        -loss.py
        -main.py
        -model.py
        -tiffs.py
        -utils.py
    -Bash Scripts
        -train.sh
        -denoise_slice.sh
        -denoise_volume.sh
    -Yaml Files
        -environment.yaml
        -baseline_config.yaml

## Getting Started    

### Bash scripts    
You will need to add the path to the virtual environment in the .sh bash scripts    

Usage of this project includes training, denoising an individual slice, and denoising a full volume with each utilizing a bash script requiring a config file that specifies the location of the data

Config file: 
    -This project contains a baseline config file to be used for training and inference. Simply copy the file into the directory of reconstructions and add the path to the directory and the name of directories containing the full reconstruction, sub reconstruction 1, and sub reconstruction 2 into the appropriate spots of the config file.

Training:
    -Training is done utilizing the `train.sh` bash script. This file first deactivates the base conda env and activates the env designed for this project. The script then calls the main.py file that runs distributed training utilizing 2 GPUs. The only item that needs to be specified is the location of the config file used for training/inference that contains the location of the data.
    -Example usage:
        `bash train.sh /path/to/config.yaml`
    -Training Logic:
        -Load in training parameters specifed by config file
        -Setup DDP training
        -Create a directory for the training output
        -Load in dataset for training
        -Initialize model/optimizer
        -Randomly select a patch size specified in the config file
        -Brief warmup period using L1Loss before including LCL loss that helps the model focus on edges
        -Each epoch prints the average loss
        -Models are saved based on the lowest lcl and validation losses and highest edge value
        -After warmup, training statistics are reset
        -Predicted images are saved every 5 epochs to help visualize the denoising process

Denoise Slice:
    -Given APS produced volumes can be quite large, this project includes the option to denoise a single slice to quickly examine the denoised images. This can be done by using the `denoise_slice.sh` script. Like train.sh, you will need to specify the path to the config file and the slice to denoise. The output will then be saved as a .tiff file in the denoised_slices folder. 
    -Example usage:
        `bash denoise_slice.sh /path/to/config.yaml 500`
    -Denoise Slice Logic:
        -Load in parameters specifed by config file
        -Create directory for denoised slices. This is not deleted and recreated after each use; thus denoised slices will accumulate
        -Load in pre-trained model
        -Fetch the slice to denoise
            -Since the project follows the 2.5D approach the previous and next slices are also fethced but this is hidden from the user
        -The slice is patched using a sliding window approach similar to training
        -Load in the mean/standard deviation used during training and normalize the image
        -Denoise the image
        -Rescale the image back to its original value
        -Save as .tiff
    
Denoise Volume:
    -Denoising the full volume involves running the `denoise_volume.sh` script. To help faciliate quick evaluation, this project contains the option to denoise a subset of slices i.e., 500-600 specified on the command line. If no range is specified/left blank, it is assumed the full volume is to be denoised. The output will be save in the denoised_volume folder. This folder is deleted and recreated each time.
    -Example usage:
        `bash denoise_volume.sh /path/to/config.yaml/ 500 600`
    -Denoised volume logic:
        -Load in parameters specifed by config file
        -Create directory for denoised volume. This will be deleted and recreated for each run
        -Load in model
        -Load in data
            -Assume .tiff files
            -Data is normalized
            -Data is patched using a sliding window approach to match the patch size seen during training
        -Calculate the optimal batch size
        -Initialize empty array of size (#CT slices, image height, image width)
        -Denoise volume in mini-batches and insert them into array
        -Rescale denoised volume
        -Save denoised volume as .tiffs
            -If a subset was selected, an offset is specifed so that the save file names match the start of the subset


## Contributing    
Contributions are welcomed that either: improve the results and/or improve the flexibility of the workflow. A few areas for improvement include:
    -Training by fine-tuning an old model
        -Each training run trains a model from scratch. While effective, this can be time consuming. For users who create multiple reconstructions from a single sample or similar sample, fine-tuning a model from a previous acquisition provides a tremendous speedup.
        -We've seen a reductions from 8-12 hours (train from scratch) done to 30-60 minutes (fine-tuning)
    -Different models
        -Current project utilizes the U-Net which works well but of course other/newer architectures may provide some improvements
