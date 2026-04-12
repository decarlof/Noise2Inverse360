# Noise2Inverse: 2.5D and 3D CT Denoising

[![Documentation Status](https://readthedocs.org/projects/noise2inverse360/badge/?version=latest)](https://noise2inverse360.readthedocs.io/en/latest/)

## Overview

This project provides an implementation of the Noise2Inverse (N2I)
framework for self-supervised denoising of CT reconstructions — no
clean ground-truth images required.  Two convolution modes are
available, selected with `--mode`:

| Mode | Flag | Network | Best for |
|------|------|---------|----------|
| **2.5D** (default) | `--mode 2.5d` | 2D U-Net, stacks N adjacent slices as channels | General synchrotron CT; fast, memory-efficient |
| **3D** | `--mode 3d` | Full 3D U-Net with skip connections | Coherent X-ray / XNH data; removes structured 3D noise and ring artifacts |

### 2.5D mode
Uses a lightweight 2D U-Net (no skip connections, GroupNorm, LeakyReLU)
that takes a stack of adjacent axial slices as input channels.  This
suppresses ring and streak artifacts while remaining fast and
memory-efficient — suitable for large synchrotron CT datasets.

### 3D mode
Uses a full 3D U-Net with skip connections (Laugros et al., bioRxiv
2025) that operates on cubic sub-volumes.  By processing all three
spatial dimensions simultaneously, it can remove structured 3D noise
such as probe-object mixing artifacts in X-ray holographic
nanotomography (XNH) — noise that slice-by-slice processing cannot
reach.  3D mode requires more GPU memory and is best suited for
coherent X-ray microscopy data.

<p align="center">
  <img src="docs/source/img/workflow.svg" width="100%">
</p>

## Installation

Create the conda environment:

``` bash
git clone https://github.com/AISDC/Noise2Inverse360 denoise
cd denoise
conda env create -f envs/denoise_environment.yml
conda activate denoise
pip install .
```

Dependencies include:

-   albumentations (data augmentation)
-   pytorch >= 2.0 (with CUDA support)
-   tifffile
-   tqdm
-   matplotlib
-   scikit-image
-   scipy
-   pyyaml


## Assumptions

-   All output (training results, inference results, trained models) are
    saved inside the reconstruction directory.

    -   User: John Smith
        -   Sample 1 Directory:
            -   Provided by the User:
                -   Full Reconstruction (Directory)
            -   Created by `tomocupy recon_steps` (tomocupy env):
                -   Sub-Reconstruction 0 (Directory)
                -   Sub-Reconstruction 1 (Directory)
            -   Created by `denoise prepare` (denoise env):
                -   `config.yaml`
            -   Created by `denoise train` / inference:
                -   TrainOutput (Directory)
                -   `<sample>_denoised_slices/`
                -   `<sample>_denoised_volume/`

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
    │   ├── __main__.py      # CLI entry point (prepare / train / slice / volume / register / search)
    │   ├── registry.py      # local model registry (~/.denoise/registry/)
    │   ├── log.py           # colored logging module
    │   ├── train.py         # DDP training loop (2.5D and 3D)
    │   ├── slice.py         # single-slice inference (2.5D only)
    │   ├── volume.py        # full-volume inference (2.5D and 3D)
    │   ├── data.py          # 2.5D dataset classes
    │   ├── data3d.py        # 3D dataset classes (cubic patches, 3D stitching)
    │   ├── data_utils.py    # patch extraction / stitching utilities
    │   ├── model.py         # 2.5D U-Net (no skip connections)
    │   ├── model3d.py       # 3D U-Net with skip connections (Laugros et al. 2025)
    │   ├── loss.py          # LCL loss
    │   ├── eval.py          # evaluation metrics
    │   ├── tiffs.py         # TIFF I/O utilities
    │   └── utils.py         # image utilities
    ├── docs/                # Sphinx documentation
    │   └── source/img/      # workflow and example figures
    ├── envs/
    │   ├── denoise_environment.yml
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
conda env create -f envs/denoise_environment.yml
conda activate denoise
pip install .
```

### Data preparation

**Step 1 — write the config YAML** (run in the `denoise` environment):

``` bash
# 2.5D mode (default)
(denoise) $ denoise prepare --file-name /data/sample.h5

# 3D mode — generates a YAML pre-configured for 3D training
(denoise) $ denoise prepare --file-name /data/sample.h5 --mode 3d
```

This writes `sample_rec_config.yaml` (with instrument metadata read from
the HDF5) and prints the two `tomocupy recon_steps` commands you need to
run next.

> **Note:** `denoise prepare` does **not** create the sub-reconstruction
> directories.  Due to a NumPy compatibility issue between the `denoise`
> and `tomocupy` environments, the sub-reconstructions must be created
> manually by running the printed commands in the `tomocupy` environment.

**Step 2 — create the sub-reconstructions** (run in the `tomocupy` environment):

``` bash
# even-indexed projections (0, 2, 4, ...)
(tomocupy) $ tomocupy recon_steps \
                 --file-name /data/sample.h5 \
                 --start-proj 0 --proj-step 2 \
                 --out-path-name /data/sample_rec_0 \
                 [... same options as the full reconstruction ...]

# odd-indexed projections (1, 3, 5, ...)
(tomocupy) $ tomocupy recon_steps \
                 --file-name /data/sample.h5 \
                 --start-proj 1 --proj-step 2 \
                 --out-path-name /data/sample_rec_1 \
                 [... same options as the full reconstruction ...]
```

`denoise prepare` prints the exact paths for `--out-path-name` derived
from `--file-name`, so you can copy-paste them directly.

### Training

Before launching a new training run, `denoise train` automatically
searches the local model registry (`~/.denoise/registry/`) for a model
trained under the same instrument conditions.  If a match is found, it
is listed and you are asked whether to proceed:

``` bash
# 2.5D mode (default — used when no --mode flag or mode stored in YAML)
(denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1

# 3D mode
(denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1 --mode 3d

Registry search found 1 matching model(s):
  [1] 2BM_pink_30keV_FLIROryx_20260219_143000  (9/9 criteria match — 100%)
       beamline:   2-BM  |  mode: pink  |  energy: 30.0 keV  |  ...
       registry path: /home/user/.denoise/registry/2BM_pink_30keV_FLIROryx_...

Train a new model anyway? [y/N]
```

Enter **N** to skip training and use the existing model, or **y** to
train anyway.  To bypass the search entirely, add `--no-search`.

To stop automatically when the validation loss plateaus, add `patience` to
the `train` section of the config YAML (default `0` = disabled):

```yaml
train:
  patience: 200   # stop if val loss does not improve for 200 epochs
```

Resume interrupted training with `--resume`:

``` bash
(denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1 --resume
```

To run two training jobs in parallel on the same node (e.g. two datasets
on a 4-GPU machine), assign a different `--master-port` to each job to
avoid a port conflict:

``` bash
# job 1 — GPUs 0,1, default port
(denoise) $ denoise train --config /data/delta_config.yaml --gpus 0,1 --no-search

# job 2 — GPUs 2,3, different port
(denoise) $ denoise train --config /data/beta_config.yaml --gpus 2,3 --no-search --master-port 29501
```

### Registering a trained model

After training, register the model so it can be found automatically in
future sessions:

``` bash
(denoise) $ denoise register \
                --config /data/sample_rec_config.yaml \
                --model-dir /data/sample_rec/TrainOutput
```

Models are stored in `~/.denoise/registry/` (never committed to git).
On APS machines where tocai and tomo4 share a GPFS home directory, a
model registered on tocai is immediately visible on tomo4.

### Searching the registry

``` bash
(denoise) $ denoise search --config /data/new_sample_rec_config.yaml
```

Prints all registry entries that match the noise fingerprint of the
given config, ranked by score (fraction of criteria matched).

### Denoise Slice (2.5D only)

``` bash
denoise slice --config /data/sample_rec_config.yaml --slice-number 500
```

-   Loads pretrained model, fetches slice ± neighboring slices (2.5D stack)
-   Applies sliding-window patching, normalizes, saves `.tiff` to `<sample>_denoised_slices/`
-   **Not available in 3D mode** — use `denoise volume` instead

### Denoise Volume

``` bash
# 2.5D (default)
denoise volume --config /data/sample_rec_config.yaml
denoise volume --config /data/sample_rec_config.yaml --start-slice 500 --end-slice 600

# 3D
denoise volume --config /data/sample_rec_config.yaml --mode 3d
```

-   Processes the full volume or a sub-volume (both modes)
-   Automatic batch size calculation, sliding-window patching with Hann blending
-   3D mode uses cubic patches and 3D overlap-add stitching
-   Saves output `.tiff` stack to `<sample>_denoised_volume/`

### Denoised Example

<p align="center">
  <img src="docs/source/img/denoised_example4.svg" width="800">
</p>

<p align="center">
  <img src="docs/source/img/brain.png" width="800">
  <br><em>Left: denoised &nbsp;|&nbsp; Right: noisy reconstruction (brain CT, APS 2-BM)</em>
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
