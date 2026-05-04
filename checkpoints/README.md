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
