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

Using tomocupy (APS 2-BM)
--------------------------

At APS 2-BM the standard tool is `tomocupy <https://tomocupy.readthedocs.io>`_.
Run the full reconstruction first (as you normally would), then run two
additional reconstructions using alternating projection subsets via
``--start-proj`` and ``--proj-step``::

    # even-indexed projections (0, 2, 4, ...)
    (tomocupy) $ tomocupy recon --start-proj 0 --proj-step 2 \
                    --out-path-name /path/to/experiment_rec_0 \
                    [... your usual tomocupy options ...]

    # odd-indexed projections (1, 3, 5, ...)
    (tomocupy) $ tomocupy recon --start-proj 1 --proj-step 2 \
                    --out-path-name /path/to/experiment_rec_1 \
                    [... your usual tomocupy options ...]

This produces three directories alongside each other::

    /path/to/
        experiment_rec/       ← full reconstruction (already exists)
        experiment_rec_0/     ← even-angle sub-reconstruction
        experiment_rec_1/     ← odd-angle sub-reconstruction

.. note::

   Use the same pre-processing options (ring removal, phase retrieval,
   normalisation) for both sub-reconstructions as for the full reconstruction.

denoise prepare
---------------

The ``denoise prepare`` command automates the two ``tomocupy recon`` calls and
writes the configuration file in a single step.  It can be run in the
``tomocupy`` environment (where ``tomocupy`` is available), before switching to
the ``n2i`` environment for training::

    (tomocupy) $ denoise prepare --out-path-name /path/to/experiment_rec \
                     [... all your usual tomocupy recon options ...]

This runs ``tomocupy recon`` twice (even and odd projections), producing::

    /path/to/
        experiment_rec/            ← full reconstruction (already exists)
        experiment_rec_0/          ← even-angle sub-reconstruction
        experiment_rec_1/          ← odd-angle sub-reconstruction
        experiment_rec_config.yaml ← ready-to-use denoise config

Do **not** include ``--start-proj``, ``--proj-step``, or ``--out-path-name``
in the extra arguments — they are handled automatically.

::

    (tomocupy) $ denoise prepare -h
    usage: denoise prepare [-h] --out-path-name PATH ...

    Create N2I sub-reconstructions with tomocupy and write a config file

    positional arguments:
      ...                   All other tomocupy recon arguments passed through verbatim

    options:
      -h, --help            show this help message and exit
      --out-path-name PATH  Base output path of the full reconstruction

Initialization
==============

If you used ``denoise prepare``, the config file is already written and you can
proceed directly to training.

If you prefer to set up the config manually, copy the baseline template to the
**parent directory** of your reconstructions and edit it::

    (n2i) $ cp /path/to/Noise2Inverse360/baseline_config.yaml \
               /path/to/experiment_config.yaml

Set at minimum:

.. code-block:: yaml

    dataset:
      directory_to_reconstructions: /path/to          # parent folder containing all three rec dirs
      sub_recon_name0: experiment_rec_0               # even-angle sub-reconstruction folder
      sub_recon_name1: experiment_rec_1               # odd-angle sub-reconstruction folder
      full_recon_name: experiment_rec                 # full reconstruction folder

The config file can live anywhere — the path is passed explicitly to every
``denoise`` command via ``--config``.

Training
========

denoise train
-------------

Train the Noise2Inverse model. ``denoise train`` automatically launches
``torchrun`` with the correct settings — no manual ``torchrun`` invocation
is required.

.. note::

   Make sure the ``n2i`` conda environment is active before running training.
   If you used a different environment (e.g. ``tomocupy``) to create the
   sub-reconstructions, switch back first::

       $ conda activate n2i

.. note::

   ``denoise train`` internally sets ``PYTHONNOUSERSITE=1`` to prevent
   user-local packages in ``~/.local/`` from shadowing the conda environment.

Launch training with two GPUs::

    (n2i) $ denoise train --config my_experiment.yaml --gpus 0,1

For single-GPU training::

    (n2i) $ denoise train --config my_experiment.yaml --gpus 0

Any number of GPUs can be used — just list all the IDs. For example, on a
4-GPU machine::

    (n2i) $ denoise train --config my_experiment.yaml --gpus 0,1,2,3

``denoise train`` counts the comma-separated IDs and sets ``--nproc_per_node``
accordingly. DDP splits the mini-batch across all GPUs, so doubling the number
of GPUs roughly halves the time per epoch.

.. tip::

   With more GPUs the effective per-GPU batch size is ``mbsz / n_gpus``.
   On a 4-GPU machine consider increasing ``mbsz`` in the config (e.g. 64 or
   128 instead of the default 32) to keep each GPU fully utilised::

       train:
         mbsz: 128

Progress is logged to ``~/logs/denoise_<timestamp>.log`` and printed to the console
with colour-coded levels. Training output (checkpoints, loss curves) is saved to
``<directory_to_reconstructions>/TrainOutput/``.

::

    (n2i) $ denoise train -h
    usage: denoise train [-h] --config FILE [--gpus IDS]

    Train the Noise2Inverse model

    options:
      -h, --help     show this help message and exit
      --config FILE  Path to the YAML configuration file
      --gpus IDS     Comma-separated list of visible GPU IDs (default: 0)

Inference
=========

denoise slice
-------------

.. tip::

   ``denoise slice`` can be run while training is still in progress.
   The model checkpoints in ``TrainOutput/`` are updated in-place whenever a
   new best is found, so you can check denoising quality at any point without
   waiting for training to finish.

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

.. note::

   The GPU batch size for inference is determined automatically by profiling the
   model against the configured patch size (``psz``) and the available GPU
   memory. On a modern GPU (e.g. A100 80 GB) the batch size will be maximised
   to saturate the GPU. For large volumes the initial RAM load (reading all TIFF
   files from disk) typically takes longer than the GPU inference itself.

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

Reusing a trained model
=======================

The trained model captures the **noise statistics of the specific dataset** it
was trained on — the CT system, detector, photon flux, and reconstruction
algorithm all influence what the network learns.

**Same beamline and similar acquisition conditions** (energy, exposure, sample
type): the model can be applied directly to a new reconstruction without
retraining. Copy the config file used for training, update
``directory_to_reconstructions`` and the reconstruction folder names, and run
inference as normal. The ``mean4norm`` and ``std4norm`` values written into the
config during training handle intensity normalisation automatically.

**Different acquisition conditions** (different energy, significantly different
noise level, different beamline): the model may still provide some improvement
but results will be suboptimal. Retraining on sub-reconstructions from the new
dataset will give the best results.

Command Reference
=================

::

    (n2i) $ denoise -h
    usage: denoise [-h] ...

    Noise2Inverse CT denoising library

    options:
      -h, --help  show this help message and exit

    Commands:

        prepare   Create N2I sub-reconstructions with tomocupy and write a config file
        train     Train the Noise2Inverse model
        slice     Denoise a single CT slice
        volume    Denoise the entire CT volume
