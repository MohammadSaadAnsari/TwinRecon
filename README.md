# TwinRecon

TwinRecon turns a single-object video into four local artifacts: an object-only dense point cloud, a Gaussian-splat PLY, a colored Poisson mesh, and a cloud-to-mesh consistency map. A local web studio lets you start scans, inspect progress, orbit the point cloud or Gaussian splats, check the segmentation, and download each artifact.

## Run

Activate the `twinrecon` Conda environment, which must contain Open3D, OpenCV, SciPy, NumPy, plyfile and Flask, plus a CUDA-enabled COLMAP build. The CLI resolves `colmap` beside the active Python interpreter first.

```bash
conda activate twinrecon
python app.py
```

Open <http://127.0.0.1:5000>. The server listens on localhost by default. The page uses WebGL2 to render fine points, the consistency cloud, and actual elliptical Gaussians from the exported PLY. The Splats tab defaults to a 160k preview for responsive orbiting; choose **Max · up to 500k** to render every exported Gaussian from the standard presets. Older, larger exports are sampled to that ceiling for laptop use. This browser renderer uses the measured DC color and does not yet display higher-order SH. The web form defaults to a 4 GB balanced preset (50 frames, 800 px, 350k splats, Poisson depth 7), with a detail preset available under Fine tune capture.

The **Edge Detail** slider tightens the Gaussian footprint in the browser. Its default shows more texture detail than the original soft view; pushing it too high may reveal gaps and noisy MVS samples. It does not change the downloaded PLY or recover detail absent from the source frames.

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

Every scan writes `reconstruction.json` and `pipeline.log`. On failure, the report records the error and leaves the workspace in place for diagnosis.

## Pipeline

1. Uniformly sample up to 60 frames. Segment the central object on CPU and save a first-frame mask preview.
2. Pass the masks to COLMAP SIFT so camera registration follows the object. Run PatchMatch Stereo and fuse its depth maps.
3. Project every fused 3D point into registered camera masks. Keep points supported as foreground in at least two views and at least 70% of visible views; remove statistical outliers. A RANSAC plane is removed only when it is much broader than the remaining object, protecting broad shoe soles. No percentile box crops the final object.
4. Fit local covariance ellipsoids in CPU chunks, preserve measured MVS color, and export a SuperSplat-compatible Gaussian PLY. The point cloud remains full-resolution; the splat count is capped for laptop use.
5. Fit a bounded Poisson mesh from at most 180,000 points, trim low-density skirts, and color mesh vertices from their nearest observed points. Measure point-to-mesh distance and export a red/green consistency cloud.

The analysis percentage is **mesh disagreement**: the fraction of cloud points farther than the job's distance tolerance from its Poisson mesh. It is not reconstruction accuracy. Measuring ground-truth accuracy requires a reference scan and a specified distance metric after alignment. The website displays the tolerance alongside the percentage.

## Current limits

The level-3 SH properties are present for viewer compatibility, but higher-order coefficients are zero. This protects measured colors; genuine view-dependent reflection requires visibility-aware multi-view radiance fitting or optimization, which is not yet implemented. The consistency output identifies cloud points unsupported by the fitted mesh; it does not fill missing geometry. Targeted recapture recommendations remain future work.

GrabCut is a CPU fallback and can include shadows or patterned support surfaces. Check `segmentation/preview.jpg` before trusting a result. For difficult scenes, use the optional `rembg` backend and a tight object box, and capture the object against a contrasting matte background. The full COLMAP pipeline needs a working NVIDIA driver and CUDA; the web UI and CPU-only component checks work without one.
