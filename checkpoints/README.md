# checkpoints/

Pre-trained model weights bundled with the repo so the demo can run
end-to-end without retraining. Total ~24 MB.

## Layout

One folder per experiment. `best.pt` is the model state dict; the
`config.resolved.yaml` next to it is the exact config used to train, so
the demo can rebuild the architecture without guessing.

```
checkpoints/
  e1_bicubic/             config.resolved.yaml      (no weights - bicubic has no params)
  e2_srcnn/               best.pt + config.resolved.yaml
  e3_agunet_mse/          best.pt + config.resolved.yaml
  e4_agunet_attn/         best.pt + config.resolved.yaml
  e5_agunet_attn_dcgan/   best.pt + config.resolved.yaml
```

## Refreshing after a re-training run

```bash
for exp in e1_bicubic e2_srcnn e3_agunet_mse e4_agunet_attn e5_agunet_attn_dcgan; do
    mkdir -p checkpoints/$exp
    [ -f runs/$exp/best.pt ] && cp runs/$exp/best.pt checkpoints/$exp/
    cp runs/$exp/config.resolved.yaml checkpoints/$exp/
done
```

`*.pt` is normally gitignored, but `.gitignore` whitelists
`checkpoints/**/*.pt` so the bundled weights stay tracked.
