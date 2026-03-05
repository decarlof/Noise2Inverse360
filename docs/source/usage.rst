=====
Usage
=====

denoise provides a command-line interface for training the Noise2Inverse model and
denoising CT reconstructions.

Data preparation
================

Noise2Inverse requires **two independent sub-reconstructions** produced from
complementary angular subsets of the raw projections (e.g. even-indexed angles
and odd-indexed angles). These two sub-reconstructions serve as the training
pairs: the network learns to predict one from the other, which forces it to
remove noise without access to any clean reference images.

.. warning::

   A single full-angle reconstruction is **not sufficient** to run denoise.
   You must create the two sub-reconstructions from the raw projection data
   before proceeding.

The sub-reconstructions should be created with the same pre-processing steps
(ring removal, phase retrieval, normalisation) as the full reconstruction.
Using `tomopy <https://tomopy.readthedocs.io>`_ and
`dxchange <https://dxchange.readthedocs.io>`_::

    import tomopy
    import dxchange

    # load raw projections, flat fields, dark fields, and angles
    proj, flat, dark, theta = dxchange.read_aps_2bm('raw_data.h5')

    # flat-field correction
    proj = tomopy.normalize(proj, flat, dark)

    # split into two interleaved angular subsets
    proj0  = proj[0::2]    # even-indexed projections
    proj1  = proj[1::2]    # odd-indexed projections
    theta0 = theta[0::2]
    theta1 = theta[1::2]

    # reconstruct each subset independently
    rec0 = tomopy.recon(proj0, theta0, algorithm='gridrec')
    rec1 = tomopy.recon(proj1, theta1, algorithm='gridrec')

    # reconstruct the full dataset
    rec_full = tomopy.recon(proj, theta, algorithm='gridrec')

    # save as tiff stacks — each into its own subdirectory
    dxchange.write_tiff_stack(rec0,    'recon_view0/recon')
    dxchange.write_tiff_stack(rec1,    'recon_view1/recon')
    dxchange.write_tiff_stack(rec_full,'recon_full/recon')

The three output directories (``recon_view0/``, ``recon_view1/``,
``recon_full/``) must all reside inside the same parent directory, which is
then set as ``directory_to_reconstructions`` in the configuration file.

Initialization
==============

Create a configuration file by copying the baseline template::

    (n2i) $ cp baseline_config.yaml my_experiment.yaml

Edit ``my_experiment.yaml`` and set at minimum::

    dataset:
      directory_to_reconstructions: /path/to/your/reconstructions
      sub_recon_name0: recon_view0   # folder with even-angle sub-reconstruction tiffs
      sub_recon_name1: recon_view1   # folder with odd-angle sub-reconstruction tiffs
      full_recon_name: recon_full    # folder with full-angle reconstruction tiffs

Training
========

denoise train
-------------

Train the Noise2Inverse model. Because the training uses PyTorch DDP, it must be
launched via ``torchrun``.

.. note::

   ``torchrun`` is installed inside the ``n2i`` conda environment. If you used a
   different environment (e.g. ``tomocupy``) to create the sub-reconstructions,
   make sure to switch back before running training::

       $ conda activate n2i

Launch training with two GPUs::

    (n2i) $ torchrun --nproc_per_node=2 -m denoise train \\
                --config my_experiment.yaml --gpus 0,1

For single-GPU training::

    (n2i) $ torchrun --nproc_per_node=1 -m denoise train \\
                --config my_experiment.yaml --gpus 0

Progress is logged to ``~/logs/denoise_<timestamp>.log`` and printed to the console
with colour-coded levels. Training output (checkpoints, loss curves) is saved to
``<directory_to_reconstructions>/TrainOutput/``.

::

    (n2i) $ denoise train -h
    usage: denoise train [-h] --config FILE [--gpus IDS]

    Train the Noise2Inverse model (launch via torchrun for DDP)

    options:
      -h, --help     show this help message and exit
      --config FILE  Path to the YAML configuration file
      --gpus IDS     Comma-separated list of visible GPU IDs (default: 0)

Inference
=========

denoise slice
-------------

Denoise a single CT slice::

    (n2i) $ denoise slice --config my_experiment.yaml --slice-number 500
    2025-01-01 10:00:00,000 - Loading slice 500
    2025-01-01 10:00:05,000 - Saved denoised slice to .../denoised_slices/00500.tiff

The denoised slice is saved as a TIFF in
``<directory_to_reconstructions>/denoised_slices/``.

::

    (n2i) $ denoise slice -h
    usage: denoise slice [-h] --config FILE [--gpus IDS] --slice-number N

    Denoise a single CT slice

    options:
      -h, --help        show this help message and exit
      --config FILE     Path to the YAML configuration file
      --gpus IDS        Comma-separated list of visible GPU IDs (default: 0)
      --slice-number N  Index of the CT slice to denoise

denoise volume
--------------

Denoise the entire CT volume::

    (n2i) $ denoise volume --config my_experiment.yaml
    2025-01-01 10:00:00,000 - Loading data into CPU memory, it will take a while ...
    2025-01-01 10:00:30,000 - Loaded 1000 slices of size 2048x2048
    2025-01-01 10:00:30,100 - Patch volume size: 65536x256x256
    2025-01-01 10:00:30,200 - Processing data ...
    ...
    2025-01-01 10:05:00,000 - Stitching denoised data ...
    2025-01-01 10:05:10,000 - Saving data to .../denoised_volume ...
    2025-01-01 10:05:30,000 - Done.

To denoise only a sub-volume (slices 200 to 400)::

    (n2i) $ denoise volume --config my_experiment.yaml --start-slice 200 --end-slice 400

The denoised volume is saved as individual TIFF files in
``<directory_to_reconstructions>/denoised_volume/``.

::

    (n2i) $ denoise volume -h
    usage: denoise volume [-h] --config FILE [--gpus IDS] [--start-slice N] [--end-slice N]

    Denoise the entire CT volume

    options:
      -h, --help       show this help message and exit
      --config FILE    Path to the YAML configuration file
      --gpus IDS       Comma-separated list of visible GPU IDs (default: 0)
      --start-slice N  Start slice index (default: first slice)
      --end-slice N    End slice index (default: last slice)

Command Reference
=================

::

    (n2i) $ denoise -h
    usage: denoise [-h] ...

    Noise2Inverse CT denoising library

    options:
      -h, --help  show this help message and exit

    Commands:

        train     Train the Noise2Inverse model (launch via torchrun for DDP)
        slice     Denoise a single CT slice
        volume    Denoise the entire CT volume
