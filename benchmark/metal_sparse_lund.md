# Metal Sparse Reconstruction Benchmark

Date: 2026-07-02

Dataset: `/Users/markus/GIT/OpenSfM/data/lund/images`

- Images: 29 JPEGs
- Resolution: 1024 x 768
- Build: `build-metal`, `METAL_ENABLED=ON`, `CUDA_ENABLED=OFF`, `GUI_ENABLED=OFF`, `OPENGL_ENABLED=OFF`
- Command:

```bash
/usr/bin/time -p build-metal/src/colmap/exe/colmap automatic_reconstructor \
  --workspace_path /private/tmp/colmap-metal-benchmark-lund \
  --image_path /Users/markus/GIT/OpenSfM/data/lund/images \
  --dense 0 \
  --use_gpu 1
```

## Results

The same automatic sparse workflow can be reproduced with:

```bash
python3 scripts/benchmark_sift_metal.py \
  /Users/markus/GIT/OpenSfM/data/lund/images \
  --automatic \
  --workspace /private/tmp/colmap-metal-benchmark-lund \
  --colmap build-metal/src/colmap/exe/colmap
```

| Metric | Value |
|---|---:|
| Total wall time | 7.36 s |
| User CPU time | 6.08 s |
| System CPU time | 0.24 s |
| Feature extraction | 0.023 min |
| Feature matching + verification | 0.065 min |
| Sparse reconstruction | 0.025 min |
| Images in database | 29 |
| Keypoint rows | 29 |
| Descriptor rows | 29 |
| Match rows | 406 |
| Two-view geometry rows | 406 |
| Total keypoints | 115,759 |
| Total raw matches | 10,298 |
| Registered images | 13 |
| Sparse points | 213 |
| Observations | 660 |
| Mean track length | 3.098592 |
| Mean reprojection error | 0.916758 px |

The log confirmed Metal was used for both sparse GPU stages:

- `Creating SIFT Metal GPU feature extractor`
- `Creating SIFT Metal GPU feature matcher`

This small real-image run shows matching and geometric verification dominate
the current sparse Metal workflow, followed by CPU sparse mapping. Feature
extraction is no longer the primary bottleneck on this dataset.

## Guided Matching Toggle

The automatic reconstructor now exposes `--guided_matching` as an opt-out for
quality presets that enable guided matching. On this dataset, disabling guided
matching reduced runtime but significantly reduced reconstruction coverage:

```bash
/usr/bin/time -p build-metal/src/colmap/exe/colmap automatic_reconstructor \
  --workspace_path /private/tmp/colmap-metal-benchmark-lund-no-guided \
  --image_path /Users/markus/GIT/OpenSfM/data/lund/images \
  --dense 0 \
  --use_gpu 1 \
  --guided_matching 0
```

| Metric | Guided matching | Guided matching disabled |
|---|---:|---:|
| Total wall time | 7.36 s | 5.64 s |
| Feature matching + verification | 0.065 min | 0.060 min |
| Registered images | 13 | 2 |
| Sparse points | 213 | 38 |
| Observations | 660 | 76 |
| Mean reprojection error | 0.916758 px | 0.825889 px |

For Gaussian splat training datasets, leave guided matching enabled by default
unless the capture set has enough redundancy that the lower registration rate is
acceptable.
