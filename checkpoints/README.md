# checkpoints/

Pre-trained model weights that ship with the repo so the demo can run
inference end-to-end without retraining.

## Layout

One folder per experiment. Each must contain `best.pt` (model state dict)
plus the matching `config.resolved.yaml` (so the demo can rebuild the
exact model architecture without guessing).

```
checkpoints/
  e1_bicubic/
    config.resolved.yaml
  e2_srcnn/
    best.pt
    config.resolved.yaml
  e3_agunet_mse/
    best.pt
    config.resolved.yaml
  e4_agunet_attn/
    best.pt
    config.resolved.yaml
  e5_agunet_attn_dcgan/
    best.pt
    config.resolved.yaml
```

E1 (bicubic) has no trainable parameters, so `best.pt` is optional for it.
The demo still evaluates it from the YAML alone.

## How to populate (developer workflow)

After running training (locally or on Zaratan), copy the artifacts:

```bash
for exp in e1_bicubic e2_srcnn e3_agunet_mse e4_agunet_attn e5_agunet_attn_dcgan; do
    mkdir -p checkpoints/$exp
    [ -f runs/$exp/best.pt ] && cp runs/$exp/best.pt checkpoints/$exp/
    cp runs/$exp/config.resolved.yaml checkpoints/$exp/
done
```

Then commit. `*.pt` is normally gitignored, but `.gitignore` whitelists
`checkpoints/**/*.pt` so these stay tracked.

## Optional: hosted fallback

If `checkpoints/` is missing entirely (e.g. the user `git clone`d instead
of unzipping the submission), `scripts/download_model.py` can pull a
tarball from a URL set via `CHECKPOINTS_URL` in `.env`. This is a backup
path; the primary distribution is the bundled files in this folder.





Output:

➜  brain-mri-image-resolution git:(feature/dockerize-project) bash run.sh
==> Building Docker image (brainsr:latest)...
[+] Building 6.6s (30/30) FINISHED                                                                                                                                        
 => [internal] load local bake definitions                                                                                                                           0.0s
 => => reading from stdin 2.68kB                                                                                                                                     0.0s
 => [preprocess internal] load build definition from Dockerfile                                                                                                      0.0s
 => => transferring dockerfile: 1.21kB                                                                                                                               0.0s
 => [dev internal] load metadata for docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime                                                                         0.5s
 => [train internal] load .dockerignore                                                                                                                              0.0s
 => => transferring context: 319B                                                                                                                                    0.0s
 => [train  1/15] FROM docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d               0.0s
 => => resolve docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d                       0.0s
 => [preprocess internal] load build context                                                                                                                         0.2s
 => => transferring context: 24.06MB                                                                                                                                 0.2s
 => CACHED [tensorboard  2/15] RUN apt-get update && apt-get install -y --no-install-recommends         git         build-essential         libgl1         libglib2  0.0s
 => CACHED [tensorboard  3/15] WORKDIR /workspace                                                                                                                    0.0s
 => CACHED [tensorboard  4/15] COPY requirements.txt ./                                                                                                              0.0s
 => CACHED [tensorboard  5/15] RUN pip install -r requirements.txt                                                                                                   0.0s
 => CACHED [tensorboard  6/15] COPY pyproject.toml README.md ./                                                                                                      0.0s
 => CACHED [tensorboard  7/15] COPY src ./src                                                                                                                        0.0s
 => CACHED [tensorboard  8/15] RUN pip install -e .                                                                                                                  0.0s
 => CACHED [tensorboard  9/15] COPY configs ./configs                                                                                                                0.0s
 => CACHED [tensorboard 10/15] COPY scripts ./scripts                                                                                                                0.0s
 => CACHED [tensorboard 11/15] COPY tests ./tests                                                                                                                    0.0s
 => [demo 12/15] COPY checkpoints ./checkpoints                                                                                                                      0.0s
 => [preprocess 13/15] COPY data/sample ./data/sample                                                                                                                0.0s
 => [tensorboard 14/15] COPY Makefile ./Makefile                                                                                                                     0.0s
 => [tensorboard 15/15] RUN mkdir -p /data/raw     /workspace/data/processed     /workspace/runs                                                                     0.1s
 => [tensorboard] exporting to image                                                                                                                                 5.3s
 => => exporting layers                                                                                                                                              0.4s
 => => exporting manifest sha256:8a170d03838b6d15e6d2128522c6c5616ce03623760cab741f6e9baeaa770cc1                                                                    0.0s
 => => exporting config sha256:3bfe0d1a35246aaf28dc97ab0eadc0f3ded09db5a326879043b7ce1a3de2e6df                                                                      0.0s
 => => exporting attestation manifest sha256:510e21aee32ea5d515db3b97c37b1a19c99e1d42a6258d18330d33890c8cd57a                                                        0.0s
 => => exporting manifest list sha256:5bf705979838c74fb5a53fac287bdbf9c7f9e292acbf460516f7820da4a21422                                                               0.0s
 => => naming to docker.io/library/brainsr:latest                                                                                                                    0.0s
 => => unpacking to docker.io/library/brainsr:latest                                                                                                                 4.9s
 => [preprocess] exporting to image                                                                                                                                  5.3s
 => => exporting layers                                                                                                                                              0.4s
 => => exporting manifest sha256:24f24c79ff9de0a4c5f2f10dfdf492c3fab606dcbb54b337aa1c242bc8b0eeca                                                                    0.0s
 => => exporting config sha256:55c50ed36f8a67c2bdff61bb68947be8c6e46c44d0b230f5e315e36bf555f279                                                                      0.0s
 => => exporting attestation manifest sha256:4fec0e2cd09404ab41740444492f4b4aaf4577ea8f9bb9c3203b106e2f761386                                                        0.0s
 => => exporting manifest list sha256:be2d6b542b2fd5377da78c7aa5af280acab38b92047ae1f9e43afe018e64d169                                                               0.0s
 => => naming to docker.io/library/brainsr:latest                                                                                                                    0.0s
 => => unpacking to docker.io/library/brainsr:latest                                                                                                                 4.9s
 => [demo] exporting to image                                                                                                                                        5.4s
 => => exporting layers                                                                                                                                              0.4s
 => => exporting manifest sha256:3d9f2db434bf81c72e22d048f8f53bdeae26a57e9cf7dc22d7c813c476c84c19                                                                    0.0s
 => => exporting config sha256:c84b0be783445e42fa45dd32d2b426c83cc40da67251293d0c80c2877d2bb4a6                                                                      0.0s
 => => exporting attestation manifest sha256:84788a28a3deda26770ecd3ae4d91621e3fc5698c40dcb9a3df405bb3224f398                                                        0.0s
 => => exporting manifest list sha256:966811679899e2bf714e9c441dbee5cd2d92a44cab63dcff18a9a1f6f757b02d                                                               0.0s
 => => naming to docker.io/library/brainsr:latest                                                                                                                    0.0s
 => => unpacking to docker.io/library/brainsr:latest                                                                                                                 4.9s
 => [train] exporting to image                                                                                                                                       5.3s
 => => exporting layers                                                                                                                                              0.4s
 => => exporting manifest sha256:361ea00e6b5fae66cff237e3b27c9ce3690855a817b231f2cbd0094410f121e9                                                                    0.0s
 => => exporting config sha256:e9a200091f6f000e2774eed7fc8a052b4888e497f19ae163a7baec27362dbf41                                                                      0.0s
 => => exporting attestation manifest sha256:11671d1d30c20dcc2c9960506779174e58e3aca9552400dbba926a036e212a7d                                                        0.0s
 => => exporting manifest list sha256:9a7933cf500278d2f38a8cb47988bc96d36996e2c094bb917bf085edcd53b4a3                                                               0.0s
 => => naming to docker.io/library/brainsr:latest                                                                                                                    0.0s
 => => unpacking to docker.io/library/brainsr:latest                                                                                                                 4.9s
 => [dev] exporting to image                                                                                                                                         5.3s
 => => exporting layers                                                                                                                                              0.4s
 => => exporting manifest sha256:020b86cd225c688b45fc294f13b0f0e81bacd65ea10460c969a4bd3c3f9e4d7c                                                                    0.0s
 => => exporting config sha256:9f6de1bb1b35cbe837af8d58eab1ff560ba2ebcd94814795379d0a7b436a2c78                                                                      0.0s
 => => exporting attestation manifest sha256:d20b2339831052a858f878ac5f8ef412b4837da7a1589cf390bda1ad7a531818                                                        0.0s
 => => exporting manifest list sha256:cdad1932e45327948cd581d84f9d018e95b1027a15b25dc75cd656f6eb281164                                                               0.0s
 => => naming to docker.io/library/brainsr:latest                                                                                                                    0.0s
 => => unpacking to docker.io/library/brainsr:latest                                                                                                                 4.9s
 => [train] resolving provenance for metadata file                                                                                                                   0.0s
 => [dev] resolving provenance for metadata file                                                                                                                     0.0s
 => [tensorboard] resolving provenance for metadata file                                                                                                             0.0s
 => [preprocess] resolving provenance for metadata file                                                                                                              0.0s
 => [demo] resolving provenance for metadata file                                                                                                                    0.0s
[+] Building 1/1
 ✔ brainsr:latest  Built                                                                                                                                             0.0s 
==> Running inference demo (E1..E5 on test split)
[+] Creating 1/1
 ✔ Network brain-mri-image-resolution_default  Created                                                                                                               0.0s 
2026-05-04 04:30:38,081 INFO __main__ | Checkpoints already present at checkpoints
2026-05-04 04:30:38,082 INFO __main__ | Processed data already present at data/processed
2026-05-04 04:30:38,088 INFO __main__ | [e1_bicubic] Evaluating on 654 test slices (device=cpu)
2026-05-04 04:30:42,186 INFO __main__ | [e1_bicubic] PSNR=30.286  SSIM=0.8856  NRMSE=0.0816  (samples -> runs/e1_bicubic/samples)
/workspace/src/brainsr/cli/demo.py:139: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(weights_path, map_location="cpu")
2026-05-04 04:30:42,196 INFO __main__ | [e2_srcnn] Loaded weights from checkpoints/e2_srcnn/best.pt
2026-05-04 04:30:42,199 INFO __main__ | [e2_srcnn] Evaluating on 654 test slices (device=cpu)
2026-05-04 04:31:12,649 INFO __main__ | [e2_srcnn] PSNR=32.681  SSIM=0.9048  NRMSE=0.0620  (samples -> runs/e2_srcnn/samples)
2026-05-04 04:31:12,673 INFO __main__ | [e3_agunet_mse] Loaded weights from checkpoints/e3_agunet_mse/best.pt
2026-05-04 04:31:12,676 INFO __main__ | [e3_agunet_mse] Evaluating on 654 test slices (device=cpu)
2026-05-04 04:31:47,480 INFO __main__ | [e3_agunet_mse] PSNR=32.244  SSIM=0.9041  NRMSE=0.0652  (samples -> runs/e3_agunet_mse/samples)
2026-05-04 04:31:47,507 INFO __main__ | [e4_agunet_attn] Loaded weights from checkpoints/e4_agunet_attn/best.pt
2026-05-04 04:31:47,511 INFO __main__ | [e4_agunet_attn] Evaluating on 654 test slices (device=cpu)
2026-05-04 04:32:23,938 INFO __main__ | [e4_agunet_attn] PSNR=32.645  SSIM=0.9078  NRMSE=0.0622  (samples -> runs/e4_agunet_attn/samples)
2026-05-04 04:32:23,963 INFO __main__ | [e5_agunet_attn_dcgan] Loaded weights from checkpoints/e5_agunet_attn_dcgan/best.pt
2026-05-04 04:32:23,965 INFO __main__ | [e5_agunet_attn_dcgan] Evaluating on 654 test slices (device=cpu)
2026-05-04 04:33:00,545 INFO __main__ | [e5_agunet_attn_dcgan] PSNR=32.840  SSIM=0.9050  NRMSE=0.0608  (samples -> runs/e5_agunet_attn_dcgan/samples)

========================================================================
experiment                           PSNR     SSIM    NRMSE
------------------------------------------------------------------------
e1_bicubic                         30.286   0.8856   0.0816
e2_srcnn                           32.681   0.9048   0.0620
e3_agunet_mse                      32.244   0.9041   0.0652
e4_agunet_attn                     32.645   0.9078   0.0622
e5_agunet_attn_dcgan               32.840   0.9050   0.0608
========================================================================
Per-experiment samples: runs/<exp>/samples/
Aggregate metrics:      runs/results.csv