=======
Credits
=======

Citations
=========

``denoise`` is built on four papers.  If you use this tool, please cite
all four (Laugros:25 only when using ``--mode 3d``).

:cite:`Hendriksen:20` — **the core Noise2Inverse method**

   This paper introduces the algorithm that ``denoise`` is built on.
   The key insight is that two CT reconstructions produced from
   interleaved angular subsets (even and odd projections) are
   statistically independent.  A U-Net trained to predict one from the
   other learns to suppress noise without any clean reference images.
   The paper provides the theoretical guarantee: when the measured noise
   is element-wise independent and zero-mean, this self-supervised
   training recovers a true denoising function.  On simulated CT
   datasets it outperforms Total-Variation Minimisation and
   state-of-the-art image denoising methods in both PSNR and SSIM.
   ``denoise prepare`` automates the creation of the two input
   sub-reconstructions required by this method.

:cite:`Hendriksen:21` — **2.5D extension and multi-dimensional generalisation**

   This paper extends Noise2Inverse from single-slice (2D) processing to
   multi-dimensional domains — space, time, and spectrum.  The 2.5D
   variant (a small stack of adjacent axial slices as input channels)
   exploits inter-slice coherence to suppress ring and streak artefacts
   more effectively than slice-by-slice processing.  ``denoise``
   implements this 2.5D architecture directly: the ``n_slices`` parameter
   in the config controls how many neighbouring slices are stacked.
   The paper also demonstrates applicability to dynamic micro-tomography
   and X-ray diffraction tomography on real-world synchrotron datasets.

:cite:`Laugros:25` — **3D U-Net architecture (used in --mode 3d)**

   This paper introduces a self-supervised image restoration pipeline for
   coherent X-ray neuronal microscopy (XNH), where structured 3D noise
   from probe-object mixing is correlated across slices and cannot be
   removed by 2D or 2.5D processing.  The architecture is a full 3D
   U-Net with skip connections, layer normalisation, and 24-rotation
   cubic-symmetry data augmentation.  ``denoise`` adopts this network
   directly in ``--mode 3d`` (see ``model3d.py``); the default
   hyperparameters written by ``denoise prepare``
   (``psz_3d=96``, ``n_blocks_3d=4``, ``start_filts_3d=56``,
   ``nb_patches_3d=17600``) match the SSD_3D reference configuration
   from this paper.

:cite:`Yunker:25` — **enhanced model selection and multi-checkpoint training**

   The original Noise2Inverse method trains for a fixed number of epochs,
   which can both miss the optimal model and waste computational
   resources.  This paper proposes saving multiple checkpoints based on
   different criteria evaluated during training: validation reconstruction
   similarity, structural similarity (SSIM), PSNR, and cosine similarity
   between the two sub-reconstruction outputs.  The best model identified
   this way improves SSIM and PSNR by up to 12.5% and 12.53%,
   respectively, while requiring only one-fifth of the training time.
   ``denoise`` implements this directly through three independently
   tracked checkpoints, each updated whenever a new best is found:

   * ``best_val_model.pth`` — lowest validation L1 loss
   * ``best_lcl_model.pth`` — lowest Laplacian Contrast Loss (LCL)
   * ``best_edge_model.pth`` — highest Laplacian edge score

   The LCL term, added after a warm-up phase, penalises loss of
   high-frequency edge detail — a further contribution of this work.
   All three checkpoints are usable for inference at any point, even
   after an interrupted training run.

.. bibliography:: bibtex/cite.bib
   :style: plain
   :labelprefix: A

Reference
---------

.. bibliography:: bibtex/ref.bib
   :style: plain
   :labelprefix: B
   :all:
