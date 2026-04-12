=====
About
=====

`denoise <https://github.com/AISDC/Noise2Inverse360>`_ provides a
self-supervised deep-learning pipeline for denoising CT reconstructions using the
`Noise2Inverse <https://arxiv.org/abs/2001.11801>`_ method.  Two convolution
modes are available: a fast **2.5D** mode (default) and a full **3D** mode for
coherent X-ray data where structured 3D noise cannot be removed slice-by-slice.

Method
------

Noise2Inverse leverages the fact that CT sinograms from alternating angular
subsets (two interleaved half-angle acquisitions) are statistically independent.
A U-Net is trained to predict one subset reconstruction from the other, learning
to remove noise without any clean reference images.

The self-supervised training objective is:

.. math::

   \mathcal{L} = \frac{1}{2}\bigl[
       \|f(x_0) - x_1\|_1 + \|f(x_1) - x_0\|_1
   \bigr]

where :math:`x_0` and :math:`x_1` are two statistically independent
sub-reconstructions and :math:`f` is the denoising network.  After a warm-up
phase, a Laplacian Contrast Loss (LCL) term is added to sharpen fine structures
without over-smoothing edges.

Convolution modes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 35 30

   * - Mode
     - Flag
     - Network
     - Best for
   * - **2.5D** (default)
     - ``--mode 2.5d``
     - 2D U-Net without skip connections, GroupNorm, LeakyReLU.
       Stacks *N* adjacent axial slices as input channels.
     - General synchrotron CT; fast, memory-efficient.
   * - **3D**
     - ``--mode 3d``
     - Full 3D U-Net with skip connections, layer norm.
       Processes cubic sub-volumes.
     - Coherent X-ray microscopy (XNH); removes structured 3D noise
       and ring artifacts.

2.5D mode
^^^^^^^^^

The 2.5D extension stacks a small number of adjacent axial slices (controlled
by ``n_slices`` in the config, default 5) as input channels to a 2D U-Net.
This enables the network to exploit inter-slice coherence and suppress ring and
streak artefacts more effectively than purely slice-by-slice (2D) processing,
while remaining fast enough for multi-terabyte clinical and synchrotron datasets.

The 2.5D network (``model.py``) uses no skip connections, GroupNorm, and
LeakyReLU — a configuration shown to be robust across synchrotron beamlines.

3D mode
^^^^^^^

The 3D mode uses a full volumetric U-Net with skip connections (encoder–decoder
architecture from Laugros et al., bioRxiv 2025, adapted from the ELEKTRONN3
implementation).  By convolving in all three spatial dimensions simultaneously,
it can remove structured 3D noise such as the probe-object mixing artifacts
present in X-ray holographic nanotomography (XNH) data — noise that is
correlated across slices and therefore invisible to a 2D or 2.5D network.

Key advantages of 3D mode:

* Removes ring artifacts and streak noise that extend across many slices
* Exploits full 3D context for denoising — no slice-by-slice approximation
* Training uses 24-rotation 3D geometric augmentation (all cube symmetries)
  to make the model invariant to volumetric orientation

Trade-offs:

* Requires significantly more GPU memory (cubic patches, default 64³)
* Inference is patch-based with 3D overlap-add stitching (slower than 2.5D)
* The ``slice`` command is not available in 3D mode — use ``volume`` instead

Training
--------

Training uses PyTorch Distributed Data Parallel (DDP) across one or more GPUs.
The ``denoise train`` command handles ``torchrun`` and ``PYTHONNOUSERSITE``
internally — no manual ``torchrun`` invocation is required::

    # 2.5D (default)
    denoise train --config baseline_config.yaml --gpus 0,1

    # 3D
    denoise train --config baseline_config.yaml --gpus 0,1 --mode 3d

The training loop alternates between two "views" (the two sub-reconstructions)
and minimises an L1 loss.  After a warm-up phase, a Laplacian Contrast Loss
(LCL) term is added.  In 3D mode, LCL is applied to the center axial slice of
each 3D prediction patch.

Three model checkpoints are saved automatically:

* ``best_val_model.pth``  — lowest validation L1 loss
* ``best_lcl_model.pth``  — lowest LCL regularisation loss
* ``best_edge_model.pth`` — highest Laplacian edge score

Inference
---------

**2.5D mode** — single slice::

    denoise slice --config baseline_config.yaml --slice-number 500

**2.5D mode** — full volume::

    denoise volume --config baseline_config.yaml

**3D mode** — full volume (``slice`` is not supported in 3D mode)::

    denoise volume --config baseline_config.yaml --mode 3d

Both modes use sliding-window patch extraction with overlap-add blending
(Hann window).  In 3D mode the patches and blending window are cubic.

denoise is primarily intended for use with full-angle CT reconstructions
produced by `tomopy <https://tomopy.readthedocs.io>`_ or similar reconstruction
packages, and has been developed for synchrotron tomography beamlines and
coherent X-ray microscopy facilities.

References
----------

* Hendriksen et al. (2020). *Noise2Inverse: Self-supervised deep convolutional
  denoising for tomography.* IEEE Transactions on Computational Imaging.
  https://doi.org/10.1109/TCI.2020.3019647

* Laugros et al. (2025). *Self-supervised image restoration in coherent X-ray
  neuronal microscopy.* bioRxiv.
  https://doi.org/10.1101/2025.02.10.633538
