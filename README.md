# 2.5D Noise2Inverse for Denoising CT Data

## Overview

This project provides an implementation of the Noise2Inverse (N2I)
framework for denoising CT data without requiring ground truth images.
It implements the 2.5D approach utilizing adjacent slices in the deep
learning model. A simple U-Net model with leaky ReLU and group norm is
used for denoising.

<p align="center">
  <img src="docs/source/img/workflow.svg" width="100%">
</p>

## Installation

Create the conda environment:

``` bash
git clone https://github.com/AISDC/Noise2Inverse360 denoise
cd denoise
conda env create -f envs/n2i_environment.yml
conda activate n2i
pip install . --no-deps
```

Dependencies include:

-   albumentations (data augmentation)
-   pytorch (2.0.1)
-   cuda (11.7)
-   tifffile
-   tqdm
-   matplotlib
-   skimage


## Assumptions

-   All output (training results, inference results, trained models) are
    saved inside the reconstruction directory.

    -   User: John Smith
        -   Sample 1 Directory:
            -   Provided by the User:
                -   Full Reconstruction (Directory)
            -   Created by `denoise prepare`:
                -   Sub-Reconstruction 0 (Directory)
                -   Sub-Reconstruction 1 (Directory)
                -   `config.yaml`
            -   Created by `denoise train` / inference:
                -   TrainOutput (Directory)
                -   `denoised_slices/`
                -   `denoised_volume/`

-   Data is saved as `.tiff` files (`.tif` or `.tiff`).

-   Model type/size is consistent across datasets.

    -   U-Net without skip connections + leaky ReLU + group norm has
        proven robust.

-   Inference can run while training is still in progress.

## Features

-   Automatic batch size optimization for A100/V100 GPUs
    -   Accounts for image size, GPU memory, and model size to reduce
        OOM errors.
-   Support for 2.5D inference with PyTorch
-   Flexible plug-and-play workflow across different samples/users

## Project Structure

    denoise/
    ├── denoise/
    │   ├── __init__.py
    │   ├── __main__.py      # CLI entry point (prepare / train / slice / volume)
    │   ├── log.py           # colored logging module
    │   ├── train.py         # DDP training loop
    │   ├── slice.py         # single-slice inference
    │   ├── volume.py        # full-volume inference
    │   ├── data.py          # dataset classes
    │   ├── data_utils.py    # patch extraction / stitching
    │   ├── model.py         # U-Net architecture
    │   ├── loss.py          # LCL loss
    │   ├── eval.py          # evaluation metrics
    │   ├── tiffs.py         # TIFF I/O utilities
    │   └── utils.py         # image utilities
    ├── docs/                # Sphinx documentation
    │   └── source/img/      # workflow and example figures
    ├── envs/
    │   ├── n2i_environment.yml
    │   └── requirements.txt
    ├── baseline_config.yaml
    ├── LICENSE
    ├── setup.py
    └── VERSION

## Getting Started

### Installation

``` bash
git clone https://github.com/AISDC/Noise2Inverse360 denoise
cd denoise
conda env create -f envs/n2i_environment.yml
conda activate n2i
pip install . --no-deps
```

### Data preparation

Create the two sub-reconstructions and config file in one step (run in
the `tomocupy` environment):

``` bash
denoise prepare --out-path-name /path/to/experiment_rec \
    [... your usual tomocupy recon options ...]
```

This produces `experiment_rec_0/`, `experiment_rec_1/`, and
`experiment_rec_config.yaml` alongside the full reconstruction.

### Training

``` bash
PYTHONNOUSERSITE=1 torchrun --nproc_per_node=2 -m denoise train \
    --config /path/to/experiment_rec_config.yaml --gpus 0,1
```

Training workflow:

1.  Load parameters from config file
2.  Setup DDP (2 GPUs)
3.  Create training output directory
4.  Load dataset
5.  Initialize model/optimizer
6.  Randomly select patch size
7.  Warmup with L1Loss before enabling LCL loss
8.  Save best models based on validation + edge metrics
9.  Save predicted images every 5 epochs

### Denoise Slice

``` bash
denoise slice --config /path/to/config.yaml --slice-number 500
```

-   Loads pretrained model
-   Fetches slice ± neighboring slices (2.5D)
-   Applies sliding window patching
-   Normalizes using training statistics
-   Saves `.tiff` to `denoised_slices/`

### Denoise Volume

``` bash
denoise volume --config /path/to/config.yaml
denoise volume --config /path/to/config.yaml --start-slice 500 --end-slice 600
```

-   Optionally denoise slice subset
-   Directory `denoised_volume/` is recreated each run
-   Automatic batch size calculation
-   Sliding window patching
-   Mini-batch inference
-   Saves output `.tiffs`

### Denoised Example 

<p align="center">
  <img src="docs/source/img/denoised_example4.svg" width="800">
</p>

## Contributing

Areas for improvement:

-   Fine-tuning from previous models (reduces training time from 8--12
    hrs to \~30--60 min)
-   Exploring alternative architectures beyond U-Net


## Citations
Relevant Citations:    
```
@article{https://doi.org/10.1109/TCI.2020.3019647,
  title={Noise2inverse: Self-supervised deep convolutional denoising for tomography},
  author={Hendriksen, Allard Adriaan and Pelt, Dani{\"e}l Maria and Batenburg, K Joost},
  journal={IEEE Transactions on Computational Imaging},
  volume={6},
  pages={1320--1335},
  year={2020},
  publisher={IEEE}
}

@article{https://doi.org/10.1038/s41598-021-91084-8,
  title={Deep denoising for multi-dimensional synchrotron X-ray tomography without high-quality reference data},
  author={Hendriksen, Allard A and B{\"u}hrer, Minna and Leone, Laura and Merlini, Marco and Vigano, Nicola and Pelt, Dani{\"e}l M and Marone, Federica and Di Michiel, Marco and Batenburg, K Joost},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={11895},
  year={2021},
  publisher={Nature Publishing Group UK London}
}

@article{https://doi.org/10.1016/j.tmater.2025.100075,
  title={Boosting Noise2Inverse via enhanced model selection for denoising computed tomography data},
  author={Yunker, Austin and Kenesei, Peter and Sharma, Hemant and Park, Jun-Sang and Miceli, Antonino and Kettimuthu, Rajkumar},
  journal={Tomography of Materials and Structures},
  pages={100075},
  year={2025},
  publisher={Elsevier}
}    

````   
