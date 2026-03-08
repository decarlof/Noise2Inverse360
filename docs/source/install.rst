=======
Install
=======

This section covers the basics of how to download and install
`denoise <https://github.com/AISDC/Noise2Inverse360>`_.

.. contents:: Contents:
   :local:

Installing from source
======================

Install from `Anaconda <https://www.anaconda.com/distribution/>`_ (Python >= 3.9).

Create and activate a dedicated conda environment::

    (base) $ conda env create -f envs/n2i_environment.yml
    (base) $ conda activate n2i
    (n2i) $

Clone the `denoise <https://github.com/AISDC/Noise2Inverse360>`_ repository::

    (n2i) $ git clone https://github.com/AISDC/Noise2Inverse360 denoise

Install the package::

    (n2i) $ cd denoise
    (n2i) $ PYTHONNOUSERSITE=1 pip install .

Test the installation
=====================

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

Configuration
=============

All parameters are stored in a YAML configuration file. Copy
``baseline_config.yaml`` from the repository root and edit it for your dataset::

    dataset:
      directory_to_reconstructions: /path/to/reconstructions
      sub_recon_name0: recon_view0
      sub_recon_name1: recon_view1
      full_recon_name: recon_full
    train:
      psz: 256
      n_slices: 5
      mbsz: 32
      lr: 0.001
      warmup: 2000
      maxep: 2000
    infer:
      overlap: 0.5
      window: "cosine"

The fields ``mean4norm`` and ``std4norm`` under ``dataset`` are written automatically
by the training script from the first sub-reconstruction statistics and must be
present before running inference.

Update
======

To update your locally installed version::

    (n2i) $ cd denoise
    (n2i) $ git pull
    (n2i) $ PYTHONNOUSERSITE=1 pip install .

Dependencies
============

The full dependency list is in ``envs/n2i_environment.yml``. Key packages::

    pytorch >= 2.0 (with CUDA support)
    tifffile
    tqdm
    pyyaml
    albumentations
    matplotlib
    scikit-image
    scipy

.. warning::

   Do **not** add ``opencv-python-headless`` to the environment.
   That package requires ``numpy>=2``, which is incompatible with
   ``torch 2.0.1`` and causes::

       AttributeError: module 'numpy.core.multiarray' has no attribute 'signedinteger'

   It was likely left behind from notebook-based exploratory work and is not
   needed by any part of the pipeline. The ``numpy<2.0`` pin in
   ``envs/requirements.txt`` guards against accidental upgrades.
