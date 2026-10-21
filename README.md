# TwinRecon

TwinRecon turns a single-object video into an object-only dense point cloud, a Gaussian-splat PLY, a colored Poisson mesh, and an explainable regional confidence atlas. A local web studio lets you start scans, inspect progress, orbit the point cloud or Gaussian splats, review uncertain regions and recapture directions, and download each artifact.

## Run

Activate the `twinrecon` Conda environment, which must contain Open3D, OpenCV, SciPy, NumPy, plyfile and Flask, plus a CUDA-enabled COLMAP build. For the optional refinement stage, install a CUDA-enabled PyTorch and `gsplat` 1.5.3 in the same environment. The CLI resolves `colmap` beside the active Python interpreter first.

```bash
conda activate twinrecon
python app.py
```

Open <http://127.0.0.1:5000>. The server listens on localhost by default. The page uses WebGL2 to render fine points, the consistency cloud, and actual elliptical Gaussians from the exported PLY. The Splats tab defaults to a 160k preview for responsive orbiting; choose **Max · up to 500k** to render every exported Gaussian from the standard presets. Older, larger exports are sampled to that ceiling for laptop use. This browser renderer uses the measured DC color and does not yet display higher-order SH. The web form defaults to a 4 GB balanced preset (50 frames, 800 px, 350k splats, Poisson depth 7), with a detail preset available under Fine tune capture.

The **Edge Detail** slider tightens the Gaussian footprint in the browser. It does not change the downloaded PLY or recover detail absent from the source frames. The browser currently displays DC color only; accepted view-dependent SH coefficients are stored in the exported PLY for SH-capable viewers.

For the CLI:

```bash
python scripts/run_pipeline.py --video data/inputs/crocs_object.mp4
```

Useful options:

- `--object-box 0.1 0.1 0.9 0.9` gives the mask a normalized bounding box when the object is off-center.
- `--segmenter rembg` selects optional model-based foreground isolation. Install `rembg` and `onnxruntime` separately; its model may download on first use. The default GrabCut mode runs with the listed dependencies.
- `--frames 50 --max-dim 800` lowers memory and run time on the 4 GB GPU.
- `--max-splats 350000 --sh-degree 0` reduces output size. The default exports 450,000 splats with level-3 SH fields.
- `--mesh-depth 7` reduces meshing time and system memory. `--no-mesh` skips the mesh and diagnostic. `--keep-work` preserves the COLMAP workspace.
- `--no-refine` skips the optional `gsplat` pass. `--refine-max-splats 40000 --refine-steps 20` tightens its memory and time budget. The default trains at most 80,000 sampled Gaussians at 320 px for 32 color/shape steps, then at most 12 SH steps. Validated corrections are spread locally through the dense PLY.

Every scan writes `reconstruction.json` and `pipeline.log`. On failure, the report records the error and leaves the workspace in place for diagnosis.

## Pipeline

1. Uniformly sample up to 60 frames. Segment the central object on CPU and save a first-frame mask preview.
2. Pass the masks to COLMAP SIFT so camera registration follows the object. Run PatchMatch Stereo and fuse its depth maps.
3. Project every fused 3D point into registered camera masks. Keep points supported as foreground in at least two views and at least 70% of visible views; remove statistical outliers. A RANSAC plane is removed only when it is much broader than the remaining object, protecting broad shoe soles. No percentile box crops the final object.
4. Fit local covariance ellipsoids in CPU chunks, preserve measured MVS color, and export a SuperSplat-compatible Gaussian PLY. If CUDA is available, run a short, masked `gsplat` pass on a bounded subset. Three held-out registered frames guard against color drift; the dense original PLY stays untouched unless their loss improves. Level-3 SH is trained only after DC/shape improves and accepted only after its own held-out improvement.
5. Fit a bounded Poisson mesh from at most 180,000 points, trim low-density skirts, and color mesh vertices from their nearest observed points.
6. Build an atlas from at most 120,000 cloud points and 12 registered views. A CPU depth buffer estimates visible camera support; masks measure silhouette agreement; projected point colors measure image mismatch; and raycasting measures cloud-to-mesh distance. Local cells get reason codes and floater flags. A heuristic capture compass ranks new azimuth/elevation directions by weak-region visibility and angular novelty. A frame audit flags unusual blur, mask disagreement, or image mismatch.

The **weak regions** percentage is the fraction of scored atlas points in regions below the documented 0.8 confidence threshold. The four evidence terms receive equal weight. This is an internal diagnostic, **not reconstruction accuracy**. The older mesh-distance view remains available. Measuring ground-truth accuracy needs a reference scan and an alignment plus distance metric. Recapture comparison reports independent summary changes, because separate SfM scans can have different scales and coordinate frames.

To add an atlas to a completed job whose COLMAP work directory has been removed:

```bash
python scripts/backfill_atlas.py outputs/<job_id>
```

This re-extracts frames and masks from the original video. The original video, `cameras.json`, cloud, and mesh must still exist. It does not rerun COLMAP.
Add `--refine` to also attempt validation-gated splat refinement on that existing job after CUDA is available.

## Current limits

When the NVIDIA driver or PyTorch CUDA is unavailable, refinement is recorded as skipped and measured MVS colors remain. The refinement has a strict sample cap and does not densify or retrain the whole object, so it may improve appearance only modestly. The current local WebGL viewer displays DC color; use a SH-capable viewer to inspect accepted higher-order coefficients. The atlas image term uses a z-buffered point-render proxy, not a full `gsplat` render. The compass is a transparent angular-coverage heuristic inspired by next-best-view research, not a P-Optimality implementation, and it cannot guarantee that recapture fixes an uncertain region. Regions missing entirely from both cloud and mesh cannot be scored.

GrabCut is a CPU fallback and can include shadows or patterned support surfaces. Check `segmentation/preview.jpg` before trusting a result. For difficult scenes, use the optional `rembg` backend and a tight object box, and capture the object against a contrasting matte background. The full COLMAP pipeline needs a working NVIDIA driver and CUDA; the web UI and CPU-only component checks work without one.
