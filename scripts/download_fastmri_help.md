# Getting the FastMRI brain dataset

The NYU FastMRI dataset is **not redistributable**, so this repo can't ship
it. Grab it yourself:

1. Apply at <https://fastmri.med.nyu.edu/> - approval is usually automatic
   for academic use once you accept the DUA.
2. NYU emails you pre-signed AWS S3 URLs (valid ~14 days).
3. Pick the brain partitions you need. For this project we use the
   `multicoil_test` batches (smallest at ~9 GB each):

   - `brain_multicoil_test_batch_0.tar` (~9 GB)
   - `brain_multicoil_test_batch_1.tar` (~9 GB)
   - `brain_multicoil_test_batch_2.tar` (~9 GB)

   Any single batch is plenty for a class project; combine all three for
   ~6,600 slices total. Note these batches are 8x undersampled - see the
   caveat in the main README.

4. Download with `curl -O '<presigned-url>'` (quote the URL - it has `&`).
5. Untar to a directory and point the project at it:

   ```bash
   cp .env.example .env
   # edit FASTMRI_DIR= to that directory, or set FASTMRI_DIRS=
   # for multiple untarred batches
   ```

6. Preprocess once to populate `data/processed/`:

   ```bash
   make preprocess          # native
   make docker-preprocess   # Docker
   ```

We only ever touch the `.npy` cache after that, so you can move or delete
the raw `.h5` files once preprocessing finishes.
