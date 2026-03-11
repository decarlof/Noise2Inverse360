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

The ``denoise prepare`` command automates the two ``tomocupy recon`` calls
needed for Noise2Inverse (N2I) — a self-supervised denoising method that
trains on pairs of independent sub-reconstructions — and writes the
configuration file in a single step.  It can be run in the ``tomocupy``
environment (where ``tomocupy`` is available), before switching to the
``denoise`` environment for training.

Pass ``--out-path-name`` pointing to the **full reconstruction** that already
exists, followed by all the ``tomocupy recon`` options you normally use
(``--file-name``, ``--reconstruction-type``, ``--binning``, etc.).  For
example::

    (tomocupy) $ denoise prepare \
                     --out-path-name /local/data/tomo/sample_rec \
                     --file-name /local/data/tomo/sample.h5 \
                     --reconstruction-type full \
                     --binning 1 \
                     --rotation-axis 1024.0

This runs ``tomocupy recon`` twice (even and odd projections) using exactly
the same options, producing::

    /local/data/tomo/
        sample_rec/            ← full reconstruction (already exists)
        sample_rec_0/          ← even-angle sub-reconstruction (N2I input A)
        sample_rec_1/          ← odd-angle sub-reconstruction  (N2I input B)
        sample_rec_config.yaml ← ready-to-use denoise config

Do **not** include ``--start-proj``, ``--proj-step``, or ``--out-path-name``
in the extra arguments — they are set automatically by ``denoise prepare``.

When finished, ``denoise prepare`` prints the full path to the config file
and the exact command to run next::

    Config written to: /local/data/tomo/sample_rec_config.yaml
    Next step:
      conda activate denoise
      denoise train --config /local/data/tomo/sample_rec_config.yaml --gpus 0,1

::

    (tomocupy) $ denoise prepare -h
    usage: denoise prepare [-h] --out-path-name PATH ...

    Create Noise2Inverse (N2I) sub-reconstructions with tomocupy and write a config file

    positional arguments:
      tomocupy_args         All other tomocupy recon arguments passed through verbatim

    options:
      -h, --help            show this help message and exit
      --out-path-name PATH  Base output path of the full reconstruction

Initialization
==============

If you used ``denoise prepare``, the config file is already written and you can
proceed directly to training.

If you prefer to set up the config manually, copy the baseline template to the
**parent directory** of your reconstructions and edit it::

    (denoise) $ cp /path/to/Noise2Inverse360/baseline_config.yaml \
               /data/sample_rec_config.yaml

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

   Make sure the ``denoise`` conda environment is active before running training.
   If you used a different environment (e.g. ``tomocupy``) to create the
   sub-reconstructions, switch back first::

       $ conda activate denoise

.. note::

   ``denoise train`` internally sets ``PYTHONNOUSERSITE=1`` to prevent
   user-local packages in ``~/.local/`` from shadowing the conda environment.

Before launching training, ``denoise train`` automatically searches the local
model registry for a model trained under the same instrument conditions and
prompts you with the result.

**Case 1 — a compatible model already exists:**

.. code-block:: text

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1

    Registry search found 1 matching model(s):
      [1] 2BM_pink_30keV_FLIROryx_22150530  (9/9 criteria match — 100%)
           beamline:            2-BM
           mode:                pink
           energy:              30.0 keV
           type:                GGG:Eu - ESRF
           serial_number:       22150530
           exposure_time:       0.035 s
           binning_x:           2
           binning_y:           2
           registry path:       /home/beams/TOMO/.denoise/registry/2BM_pink_30keV_FLIROryx_22150530
    A compatible model may already exist. Copy the registry path above as --model-dir for slice/volume inference.

    Train a new model anyway? [y/N]

Press **Enter** (or **N**) to cancel and reuse the existing model.
Enter **y** to proceed with a new training run.

**Case 2 — no compatible model found:**

.. code-block:: text

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1

    Registry search: no matching models found. Existing models:
      - 2BM_pink_30keV_FLIROryx_22150530

    Proceed with new training? [Y/n]

Press **Enter** (or **Y**) to start training.
Enter **n** to cancel and inspect the listed models manually.
If the registry is empty the message reads ``Registry search: registry is empty.``

To skip the registry search entirely, add ``--no-search``::

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1 --no-search

Launch training with two GPUs::

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1

For single-GPU training::

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0

Any number of GPUs can be used — just list all the IDs. For example, on a
4-GPU machine::

    (denoise) $ denoise train --config /data/sample_rec_config.yaml --gpus 0,1,2,3

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

Three checkpoints are saved independently during training — each updated
in-place whenever a new best is found:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - File
     - Criterion
     - Recommended use
   * - ``best_val_model.pth``
     - Lowest validation L1 loss
     - **Inference** (best visual quality)
   * - ``best_lcl_model.pth``
     - Lowest LCL regularisation loss
     - Default (conservative)
   * - ``best_edge_model.pth``
     - Highest Laplacian edge score
     - Sharpness-focused comparison

.. note::

   If training is interrupted (e.g. OOM or scheduler walltime), all three
   checkpoints reflect the best values seen up to that point and are fully
   usable for inference. For example, training killed at epoch 1710/2000
   produced ``best_val_model.pth`` at epoch 1705 (val loss 0.053) and
   ``best_edge_model.pth`` at epoch 1650 — both suitable for inference.

Resuming interrupted training
-----------------------------

Every epoch, ``denoise train`` writes a ``resume.pth`` checkpoint to
``TrainOutput/`` that captures the complete training state: model weights,
optimiser state (Adam momentum), epoch counter, ``model_updates``, all best
values, and the full loss history.  If training is interrupted for any reason,
restart from the last completed epoch with ``--resume``::

    (denoise) $ denoise train --config sample_rec_config.yaml --gpus 0,1 --resume

Training continues from epoch ``N+1`` where ``N`` is the last fully completed
epoch.  All three best-model checkpoints (``best_val_model.pth``,
``best_lcl_model.pth``, ``best_edge_model.pth``) are preserved and remain
usable for inference at any point.

.. note::

   ``resume.pth`` is overwritten at the end of every epoch.  It is safe to
   run ``denoise slice`` or ``denoise volume`` while training is in progress —
   the inference commands load the *best* checkpoints, not ``resume.pth``.

.. warning::

   Do **not** use ``--resume`` after changing the config (e.g. ``psz``,
   ``n_slices``).  You may safely increase ``maxep`` to extend training beyond
   the original limit.

::

    (denoise) $ denoise train -h
    usage: denoise train [-h] --config FILE [--gpus IDS] [--resume] [--no-search]

    Train the Noise2Inverse model

    options:
      -h, --help     show this help message and exit
      --config FILE  Path to the YAML configuration file
      --gpus IDS     Comma-separated list of visible GPU IDs (default: 0)
      --resume       Resume from the last completed epoch (requires resume.pth in TrainOutput/)
      --no-search    Skip registry search before training

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

    (denoise) $ denoise slice --config sample_rec_config.yaml --slice-number 500
    2025-01-01 10:00:00,000 - Loading slice 500
    2025-01-01 10:00:05,000 - Saved denoised slice to .../denoised_slices/00500.tiff

The denoised slice is saved as a TIFF in
``<directory_to_reconstructions>/denoised_slices/``.

::

    (denoise) $ denoise slice -h
    usage: denoise slice [-h] --config FILE [--gpus IDS] --slice-number N [--checkpoint {val,lcl,edge}]

    Denoise a single CT slice

    options:
      -h, --help                    show this help message and exit
      --config FILE                 Path to the YAML configuration file
      --gpus IDS                    Comma-separated list of visible GPU IDs (default: 0)
      --slice-number N              Index of the CT slice to denoise
      --checkpoint {val,lcl,edge}   Checkpoint to use (default: lcl)

denoise volume
--------------

.. note::

   The GPU batch size for inference is determined automatically by profiling the
   model against the configured patch size (``psz``) and the available GPU
   memory. On a modern GPU (e.g. A100 80 GB) the batch size will be maximised
   to saturate the GPU. For large volumes the initial RAM load (reading all TIFF
   files from disk) typically takes longer than the GPU inference itself.

Denoise the entire CT volume::

    (denoise) $ denoise volume --config sample_rec_config.yaml
    2025-01-01 10:00:00,000 - Loading data into CPU memory, it will take a while ...
    2025-01-01 10:00:30,000 - Loaded 1000 slices of size 2048x2048
    2025-01-01 10:00:30,100 - Patch volume size: 65536x256x256
    2025-01-01 10:00:30,200 - Processing data ...
    ...
    2025-01-01 10:05:00,000 - Stitching denoised data ...
    2025-01-01 10:05:10,000 - Saving data to .../denoised_volume ...
    2025-01-01 10:05:30,000 - Done.

To denoise only a sub-volume (slices 200 to 400)::

    (denoise) $ denoise volume --config sample_rec_config.yaml --start-slice 200 --end-slice 400

The denoised volume is saved as individual TIFF files in
``<directory_to_reconstructions>/denoised_volume/``.

::

    (denoise) $ denoise volume -h
    usage: denoise volume [-h] --config FILE [--gpus IDS] [--start-slice N] [--end-slice N] [--checkpoint {val,lcl,edge}]

    Denoise the entire CT volume

    options:
      -h, --help                    show this help message and exit
      --config FILE                 Path to the YAML configuration file
      --gpus IDS                    Comma-separated list of visible GPU IDs (default: 0)
      --start-slice N               Start slice index (default: first slice)
      --end-slice N                 End slice index (default: last slice)
      --checkpoint {val,lcl,edge}   Checkpoint to use (default: lcl)

Performance example
^^^^^^^^^^^^^^^^^^^

The table below shows measured wall-clock times for a real APS 2-BM dataset:

* **Volume**: 2426 slices × 3232 × 3232 voxels (~100 GB in RAM)
* **Training hardware**: 2× NVIDIA V100 32 GB
* **Inference hardware**: 1× NVIDIA A100 80 GB (batch size 512, auto-selected)

Two inference runs are shown — one with ``--checkpoint lcl`` (default) and one
with ``--checkpoint val`` — to illustrate that checkpoint choice does not affect
throughput: the model architecture is identical, only the weights differ.

.. list-table::
   :header-rows: 1
   :widths: 40 18 18

   * - Stage
     - ``--checkpoint lcl``
     - ``--checkpoint val``
   * - Training (1710/2000 epochs, killed by SIGKILL at ~85%)
     - 42 h 23 min
     - —
   * - Load + normalize TIFFs into RAM
     - 5 min 59 s
     - 4 min 23 s
   * - Pre-pad + build patch index
     - 17 s
     - 18 s
   * - GPU inference (2962 batches × 512 patches)
     - 41 min 45 s
     - 41 min 13 s
   * - Stitch patches → volume
     - 11 min 16 s
     - 16 min 58 s
   * - Save 2426 TIFFs to disk
     - 7 min 27 s
     - 7 min 41 s
   * - **Total inference**
     - **~67 min**
     - **~71 min**

GPU inference (~41 min, 60–62% of total) is stable across runs. Loading
variation reflects OS disk-cache state. Stitching variation (~11–17 min) is
due to system memory pressure. Training time dwarfs inference by ~38×; using
4 GPUs would reduce training to roughly 10–12 hours.

Reusing a trained model
=======================

The Noise2Inverse model learns the **noise statistics of the imaging system**,
not anything about the sample itself.  This means the model is fully
**sample-independent** and can be reused across datasets as long as the
acquisition conditions that determine the noise remain the same.

What the model learns
---------------------

The network characterises:

* Detector read noise and dark current
* Shot noise at the specific photon flux (energy × exposure × beam current)
* Scintillator glow and afterglow patterns
* Ring artifact signatures from detector pixel non-uniformity
* Any systematic noise from the lens and optic combination

What does **not** affect reusability:

* Sample composition, density, or size
* Rotation axis position
* Number of projections (within reason)
* Binning (as long as ``psz`` in the config matches the binned pixel size)

When to reuse vs. retrain
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Condition
     - Reuse model?
     - Action
   * - Same beamline, same detector, same energy, same exposure
     - **Yes**
     - Update config paths only
   * - Same setup, different sample
     - **Yes**
     - Update config paths only
   * - Same setup, different binning
     - **Yes** (if ``psz`` adjusted)
     - Update ``psz`` in config
   * - Same setup, slightly different exposure (±20%)
     - **Likely yes**
     - Try inference; retrain if quality is poor
   * - Different energy or significantly different flux
     - **No**
     - Retrain on new sub-reconstructions
   * - Different detector or scintillator
     - **No**
     - Retrain on new sub-reconstructions
   * - Different beamline
     - **No**
     - Retrain on new sub-reconstructions

How to reuse
------------

Copy the config file used for training, update the three path fields, and
run inference as normal::

    dataset:
      directory_to_reconstructions: /path/to/new_experiment   # ← update
      sub_recon_name0: new_experiment_rec_0                    # ← update
      sub_recon_name1: new_experiment_rec_1                    # ← update
      full_recon_name: new_experiment_rec                      # ← update
      mean4norm: 0.1234    # ← keep from original training config
      std4norm:  0.0567    # ← keep from original training config

The ``mean4norm`` and ``std4norm`` values written by the training script
handle intensity normalisation automatically — the new sample's absorption
does not matter.

Then run inference directly::

    (denoise) $ denoise volume --config new_sample_rec_config.yaml --checkpoint val

Model Registry
==============

``denoise`` includes a local model registry so that trained models can be
stored, searched, and reused without manual bookkeeping.  Models are stored
in ``~/.denoise/registry/`` (override with the ``DENOISE_REGISTRY``
environment variable).  Because tocai and tomo4 share the same home directory
over GPFS, a model registered on tocai is immediately available on tomo4.

.. note::

   The registry lives entirely in your home directory and is **never** tracked
   by git.

Instrument metadata in the config
----------------------------------

When ``denoise prepare`` is run with ``--file-name``, it reads key instrument
parameters directly from the raw HDF5 file and writes them into the
``metadata:`` block of the generated config YAML::

    metadata:
      start_date: '2026-02-19T09:02:27-0600'
      name: Chawla
      beamline: 2-BM
      current: 200.03 mA
      energy: 30.0 keV
      mode: pink
      type: GGG:Eu - ESRF
      active_thickness: 17.0 um
      magnification: 9.835
      resolution: 0.7015 um
      manufacturer: FLIR
      model: Oryx ORX-10G-310S9M
      serial_number: 22150530
      exposure_time: 0.035 s
      temperature: 25.3 C
      binning_x: 2.0
      binning_y: 2.0
      propagation_distance: 100.0 mm

This block is the basis for all registry matching.  The fields used to decide
whether two configs share the same noise fingerprint are:

``beamline``, ``mode``, ``energy``, ``type`` (scintillator),
``active_thickness``, ``serial_number``, ``exposure_time``,
``binning_x``, ``binning_y``, and ``temperature`` (when present in both).

``propagation_distance``, ``magnification``, ``current``, and experimenter
fields are recorded for provenance but are not used in matching.

Registering a trained model
----------------------------

After training is complete, register the model with its config::

    (denoise) $ denoise register \
                    --config /data/sample_rec_config.yaml \
                    --model-dir /data/sample_rec/TrainOutput

An entry is created automatically under a name derived from the metadata
(e.g. ``2BM_pink_30keV_22150530_35ms_20260219_143000``).  To use a
custom name::

    (denoise) $ denoise register \
                    --config /data/sample_rec_config.yaml \
                    --model-dir /data/sample_rec/TrainOutput \
                    --name 2BM_pink_30keV_FLIROryx

The registry entry contains a copy of the config and all three checkpoint
files (``best_val_model.pth``, ``best_lcl_model.pth``,
``best_edge_model.pth``).

::

    (denoise) $ denoise register -h
    usage: denoise register [-h] --config FILE --model-dir DIR [--name NAME]

    Register a trained model in the local registry for later reuse.

    options:
      -h, --help        show this help message and exit
      --config FILE     Path to the YAML configuration file used for training
      --model-dir DIR   Directory containing the trained checkpoints (TrainOutput/)
      --name NAME       Registry entry name (auto-generated from metadata if omitted)

Searching the registry
-----------------------

To manually search the registry for models compatible with a given config::

    (denoise) $ denoise search --config /data/new_sample_rec_config.yaml

Example output::

    Registry search found 1 matching model(s):
      [1] 2BM_pink_30keV_FLIROryx_20260219_143000  (9/9 criteria match — 100%)
           beamline:            2-BM
           mode:                pink
           energy:              30.0 keV
           type:                GGG:Eu - ESRF
           serial_number:       22150530
           exposure_time:       0.035 s
           binning_x:           2.0
           binning_y:           2.0
           registry path:       /home/user/.denoise/registry/2BM_pink_30keV_FLIROryx_20260219_143000

::

    (denoise) $ denoise search -h
    usage: denoise search [-h] --config FILE

    Search the registry for models trained under compatible instrument conditions.

    options:
      -h, --help     show this help message and exit
      --config FILE  Path to the YAML configuration file to match against

Auto-search before training
----------------------------

By default, ``denoise train`` searches the registry automatically before
launching a training run.

**Match found** — compatible model(s) exist::

    (denoise) $ denoise train --config /data/new_sample_rec_config.yaml --gpus 0,1

    Registry search found 1 matching model(s):
      [1] 2BM_pink_30keV_FLIROryx_20260219_143000  (9/9 criteria match — 100%)
           ...
           registry path: /home/user/.denoise/registry/2BM_pink_30keV_FLIROryx_20260219_143000
    A compatible model may already exist. Copy the registry path above as --model-dir for slice/volume inference.

    Train a new model anyway? [y/N]

Entering **N** (or pressing Enter) cancels training so you can use the
existing model.  Entering **y** proceeds with training as normal.

**No match found** — registry is searched but no compatible model exists::

    (denoise) $ denoise train --config /data/new_sample_rec_config.yaml --gpus 0,1

    Registry search: no matching models found. Existing models:
      - 2BM_pink_30keV_FLIROryx_22150530

    Proceed with new training? [Y/n]

Pressing Enter (or **Y**) starts training.  Entering **n** cancels so you can
inspect the listed models manually before deciding.  If the registry is empty
the message reads ``Registry search: registry is empty.`` and the same
``[Y/n]`` prompt is shown.

To skip the registry search entirely::

    (denoise) $ denoise train --config sample_rec_config.yaml --gpus 0,1 --no-search

Using a registered model for inference
----------------------------------------

Point ``--model-dir`` at the registry entry to use it for inference on a
new dataset::

    dataset:
      directory_to_reconstructions: /data/new_experiment
      sub_recon_name0: new_rec_0
      sub_recon_name1: new_rec_1
      full_recon_name: new_rec
      mean4norm: 0.1234    # ← from the original training config
      std4norm:  0.0567    # ← from the original training config

Then run inference using the registered checkpoints::

    (denoise) $ denoise volume \
                    --config /data/new_sample_rec_config.yaml \
                    --checkpoint val

.. note::

   Copy ``mean4norm`` and ``std4norm`` from the registered model's
   ``config.yaml`` into the new config before running inference.  These
   normalisation statistics must match the values used during training.

Command Reference
=================

::

    (denoise) $ denoise -h
    usage: denoise [-h] ...

    Noise2Inverse CT denoising library

    options:
      -h, --help  show this help message and exit

    Commands:

        prepare   Create Noise2Inverse (N2I) sub-reconstructions with tomocupy and write a config file
        train     Train the Noise2Inverse model
        slice     Denoise a single CT slice
        volume    Denoise the entire CT volume
        register  Register a trained model in the local registry (~/.denoise/registry/)
        search    Search the registry for models matching a config noise fingerprint
