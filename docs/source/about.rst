=====
About
=====

`denoise <https://github.com/AISDC/Noise2Inverse360>`_ provides a
self-supervised deep-learning pipeline for denoising CT reconstructions using the
`Noise2Inverse <https://arxiv.org/abs/2001.11801>`_ method with 2.5D convolutional
neural networks.

Method
------

Noise2Inverse leverages the fact that CT sinograms from alternating angular subsets
(two interleaved half-angle acquisitions) are statistically independent. A U-Net is
trained to predict one subset from the other, learning to remove noise without any
clean reference images.

The 2.5D extension stacks a small number of adjacent axial slices as input channels,
enabling the network to exploit inter-slice coherence and suppress ring and streak
artefacts more effectively than slice-by-slice (2D) processing.

Training
--------

Training uses PyTorch Distributed Data Parallel (DDP) across one or more GPUs. The
``denoise train`` command handles ``torchrun`` and ``PYTHONNOUSERSITE`` internally —
no manual ``torchrun`` invocation is required::

    denoise train --config baseline_config.yaml --gpus 0,1

The training loop alternates between two "views" (the two sub-reconstructions) and
minimises an L1 loss augmented by a Laplacian Contrast Loss (LCL) term after a warm-up
phase. Three model checkpoints are saved automatically:

* ``best_val_model.pth`` — lowest validation L1 loss
* ``best_lcl_model.pth`` — lowest LCL loss
* ``best_edge_model.pth`` — highest Laplacian edge score

Inference
---------

The ``denoise slice`` command denoises a single axial slice using sliding-window
patch-based inference with overlap-add blending::

    denoise slice --config baseline_config.yaml --slice-number 500

The ``denoise volume`` command processes the entire reconstructed volume (or a
user-specified sub-volume) and saves the result as a stack of TIFF files::

    denoise volume --config baseline_config.yaml

denoise is primarily intended for use with full-angle CT reconstructions produced by
`tomopy <https://tomopy.readthedocs.io>`_ or similar reconstruction packages, and
has been developed for use at synchrotron tomography beamlines.
