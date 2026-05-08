=======
Install
=======

This section covers the basics of how to download and install
`denoise <https://github.com/AISDC/Noise2Inverse360>`_.

.. contents:: Contents:
   :local:

Installing from source
======================

Clone the `denoise <https://github.com/AISDC/Noise2Inverse360>`_ repository::

    (base) $ git clone https://github.com/AISDC/Noise2Inverse360 denoise
    (base) $ cd denoise

One-command environment setup (APS machines, linux-64)
-------------------------------------------------------

A fully-pinned environment file is provided for reproducible installs on
APS hardware (tocai / tomo4).  It includes Python 3.11, PyTorch 2.6 with
CUDA 12.4, and all dependencies::

    (base) $ conda env create -f envs/denoise_environment.yml
    (base) $ conda activate denoise
    (denoise) $ pip install .

Manual environment setup
------------------------

For other machines or when you need a different CUDA version, create the
environment manually.

Install from `Anaconda <https://www.anaconda.com/distribution/>`_ (Python 3.11 recommended)::

    (base) $ conda create -n denoise python=3.11
    (base) $ conda activate denoise

Install PyTorch with CUDA support (adjust the ``cu124`` tag to match your driver)::

    (denoise) $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Install the package.

On a machine **with internet access** (e.g. tocai)::

    (denoise) $ pip install .

On a machine **without internet access** (e.g. tomo4), use ``--no-build-isolation``
so pip reuses the already-installed build tools instead of trying to download them::

    (denoise) $ pip install --no-build-isolation .

Test the installation
=====================

::

    (denoise) $ denoise -h
    usage: denoise [-h] ...

    Noise2Inverse CT denoising library

    options:
      -h, --help  show this help message and exit

    Commands:

        prepare   Write a denoise config YAML from an HDF5 file path
        train     Train the Noise2Inverse model
        slice     Denoise a single CT slice (2.5D mode only)
        volume    Denoise the entire CT volume
        register  Register a trained model in the local registry (~/.denoise/registry/)
        search    Search the registry for models matching a config noise fingerprint

Configuration
=============

All parameters are stored in a YAML configuration file.  ``denoise prepare``
writes a complete config (with both 2.5D and 3D fields) from an HDF5 file
path; alternatively, copy ``baseline_config.yaml`` from the repository root
and edit it for your dataset::

    dataset:
      directory_to_reconstructions: /path/to/reconstructions
      sub_recon_name0: recon_view0
      sub_recon_name1: recon_view1
      full_recon_name: recon_full
    train:
      psz: 256              # 2.5D patch size
      n_slices: 5           # 2.5D stack depth
      psz_3d: 96            # 3D cubic patch size (must be divisible by 2**n_blocks_3d)
      nb_patches_3d: 17600  # 3D random patches per epoch
      n_blocks_3d: 4        # 3D U-Net encoder depth
      start_filts_3d: 56    # 3D first-block filter count
      mbsz: 32              # batch size (use 4–8 in 3D mode)
      lr: 0.001
      warmup: 2000
      maxep: 2000
      patience: 0
      # mode: '2.5d'        # written automatically at first train; '2.5d' or '3d'
    infer:
      overlap: 0.5
      window: "cosine"      # use "hann" for 3D mode

The fields ``mean4norm`` and ``std4norm`` under ``dataset`` are written automatically
by the training script from the first sub-reconstruction statistics and must be
present before running inference.  The ``mode`` field under ``train`` is also
written automatically the first time you run ``denoise train`` and is read by
``slice``/``volume`` so you do not need to repeat ``--mode`` at inference time.

Update
======

To update your locally installed version, pull the latest code and reinstall.

On a machine **with internet access** (e.g. tocai)::

    (denoise) $ cd denoise
    (denoise) $ git pull
    (denoise) $ pip install .

On a machine **without internet access** (e.g. tomo4), use ``--no-build-isolation``
so pip reuses the already-installed build tools instead of trying to download them::

    (denoise) $ cd denoise
    (denoise) $ git pull
    (denoise) $ pip install --no-build-isolation .

If the conda environment is on a shared filesystem (e.g. NFS/GPFS), installing
on the internet-facing machine is sufficient — the update is immediately visible
on all other machines sharing the same environment.

Dependencies
============

The full dependency list is in ``envs/requirements.txt``. Key packages::

    pytorch >= 2.0 (with CUDA support)
    tifffile
    tqdm
    pyyaml
    albumentations
    matplotlib
    scikit-image
    scipy

.. note::

   ``opencv-python-headless`` is pulled in automatically as a transitive
   dependency of ``albumentations``.  This is harmless because the
   ``numpy<2.0`` pin ensures numpy 1.x is installed, which is compatible
   with both torch and opencv.  Do **not** install ``opencv-python``
   (the GUI variant) — it conflicts with the headless version.
