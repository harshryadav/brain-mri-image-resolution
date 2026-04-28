# Project proposal (MSML640)

> Reproduced from the original course proposal for reference. The repository
> implements the approach described below.

## Medical Image Super-Resolution for Enhanced Diagnostic Imaging

**Team:** Harsh Yadav (116683388), Deepika Ghotra (121314574),
Utkrisht Nath (121335289)

### Problem statement

Magnetic Resonance Image (aka MRI) scans are essential for clinical
diagnosis. Although they have revolutionized the way we understand medicine
today, there are limitations to the technology. For example, high-resolution
scans require patients to remain perfectly still for long periods and any
movements from them can distort the image causing blurriness around edges
that may hide finer brain structures. This usually results in lower
resolution images. Due to these potential inaccuracies, it could result in
misdiagnoses for patients.

Our goal for this project is to build a deep learning super-resolution (SR)
system that takes these low resolution images and reconstructs them into
sharper and higher resolution images that preserve anatomical details that a
clinician needs to properly diagnose a patient.

| Input                                                       | Output                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| Low-resolution 2D brain MRI axial slice (4x downsampled)    | High-resolution reconstruction (256x256) with sharp boundaries      |

The end product would be a successful system that quantitatively improves
image quality (PSNR and SSIM) while maintaining structural integrity required
for clinical use.

### Approach

Start with simple models and build up to a more complex attention-based one:

- **Baseline (Bicubic Interpolation).** Classical mathematical approach,
  fast, no training, the floor every model should beat.
- **Model 1 - SRCNN.** A lightweight 3-layer convolutional network built
  from scratch in PyTorch. Patch extraction, feature mapping, reconstruction.
- **Model 2 - AGUNet (Attention-Gated U-Net).** Our primary focus. Attention
  gates focus reconstruction on clinically important regions (grey/white
  matter boundaries, sulci, etc.) and filter out noise. Paired with a DCGAN
  critic for sharper outputs, ablated across experiments E3-E5. We reuse
  the open-source PyTorch code from
  [Li et al. (2022)](https://github.com/bryanlimy/clinical-super-mri) as
  inspiration and adapt the data pipeline for the FastMRI brain dataset
  (k-space conversion via IFFT + Root-Sum-of-Squares across coils, a custom
  loader for the converted scans).

### Ablation study

| Exp | Method                                  | Loss / config                       | Expected outcome                                             |
| --- | --------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| E1  | Bicubic Baseline                        | n/a                                 | Lower-bound PSNR/SSIM/NRMSE                                  |
| E2  | SRCNN                                   | MSE                                 | Moderate improvement over bicubic (~1-2 dB)                  |
| E3  | AGUNet (plain)                          | MSE only                            | Base U-Net performance; isolates skip-connection benefit     |
| E4  | AGUNet + Attention                      | MSE + attention gates               | Better focus on brain structures; sharper boundaries         |
| E5  | AGUNet + Attention + Critic             | MSE + attention + DCGAN critic      | Sharpest output; should approach Li et al. benchmark         |

LR inputs are generated synthetically by Gaussian blur (sigma 0.5-2.0) and
bicubic downsampling at 4x scale (or 2x as a secondary).

### Dataset

We use the [FastMRI brain dataset](https://fastmri.med.nyu.edu/), specifically
the `multicoil_test` subset (raw k-space `.h5`, dimensions slices x coils x H x W).
A custom reconstruction pipeline converts k-space to magnitude via per-coil
2D IFFT followed by RSS. We resize to 256x256 and split deterministically
70/20/10 by volume.

### Tools

- **PyTorch & Torchvision** - core framework, model architectures, training
- **h5py** - process the raw FastMRI k-space `.h5` files
- **scikit-image & torchmetrics** - PSNR, SSIM, NRMSE

### Self-implemented components

- **FastMRI data pipeline.** `fastmri_convert.py` reconstructs k-space `.h5`
  files into magnitude images via IFFT + RSS; the dataset auto-detects
  processed files and generates synthetic LR.
- **Degradation pipeline.** A custom module simulating LR via Gaussian
  blurring and bicubic downsampling (4x primary, 2x secondary).
- **SRCNN architecture.** 3-layer CNN trained with Adam + MSE, written from
  scratch in PyTorch.
- **Evaluation harness.** A script that computes PSNR/SSIM/NRMSE across all
  experiments and aggregates them into a single CSV.

### Evaluation metrics & success criteria

- **PSNR** - pixel-level accuracy vs ground truth. Target: PSNR improvement
  >= 2 dB over bicubic on the test set.
- **SSIM** - perception-based texture/structure similarity. Target:
  SSIM > 0.90 on the test split (reference benchmark Li et al.: 0.985).
- **NRMSE** - normalized error metric used by Li et al., enables direct
  comparison.
- Visual inspection confirms white/grey matter boundaries and brain sulci
  are preserved.

We expect E5 (full AGUNet) to produce the highest quality, approaching the
published Li et al. benchmark. E3 and E4 quantify how much attention gating
and adversarial training each contribute.

### Case study and prior work

- **Real-ESRGAN** (Wang et al. 2021) - degradation pipeline (blur, noise,
  downsampling) that mirrors real MRI noise; informs our LR generation.
- **Multi-stage GAN for medical images** (Ahmad et al. 2022, Sci. Reports)
  - confirms deep SR generalizes across medical imaging types and informs
  our AGUNet training strategy.
- **Deep Attention SR of brain MRI under clinical protocols**
  (Li et al. 2022, Frontiers in Computational Neuroscience) - direct source
  of our AGUNet architecture and target benchmark (PSNR 35.39, SSIM 0.985).

### Fallback plan

If AGUNet or the DCGAN critic prove computationally infeasible or unstable,
pivot to a deeper SRCNN (8-10 residual conv layers) and shift focus to a
controlled study of how depth affects recovery, plus a 2x vs 4x scale
comparison. The data pipeline work stays central either way.

| Risk                                       | Fallback action                                              |
| ------------------------------------------ | ------------------------------------------------------------ |
| AGUNet too complex to adapt                | Drop AGUNet; use a 5-layer SRCNN as Model 2; run 2x and 4x   |
| DCGAN critic causes training instability   | Remove the critic; run only E3 vs E4 (MSE vs MSE + attention)|
| Metrics below success threshold            | Reduce scale factor 4 -> 2 and re-run; report honestly       |
