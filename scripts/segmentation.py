"""CPU-only object masks and multi-view point validation.

Masks use GrabCut with color-based seeds. They are intentionally conservative at
the silhouette: a one-pixel expansion protects dark soles and thin straps.
"""

from pathlib import Path

import cv2
import numpy as np


def _largest_center_component(binary):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return None
    height, width = binary.shape
    score = stats[1:, cv2.CC_STAT_AREA].astype(float)
    distance = np.linalg.norm((centers[1:] - [width / 2, height / 2]) / [width, height], axis=1)
    score /= 1 + 3 * distance
    label = 1 + int(np.argmax(score))
    return (labels == label).astype(np.uint8)


def segment_frame(image, object_box=None):
    """Return an 8-bit foreground mask for a centered single object.

    object_box is optional (x0, y0, x1, y1) in normalized image coordinates.
    It is useful when the object is off center or touches the frame edge.
    """
    height, width = image.shape[:2]
    if height < 32 or width < 32:
        raise ValueError("Image is too small for object segmentation")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    border = max(4, round(min(width, height) * 0.045))
    rim = np.concatenate((lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
                          lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)))
    background = np.median(rim, axis=0)
    difference = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    border_difference = np.linalg.norm(rim.astype(np.float32) - background, axis=1)
    threshold = max(24.0, float(np.percentile(border_difference, 75)) * 1.35)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidate = ((difference > threshold) | (hsv[:, :, 1] > 80)).astype(np.uint8)
    candidate[:border] = 0
    candidate[-border:] = 0
    candidate[:, :border] = 0
    candidate[:, -border:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, width // 100 | 1), max(3, height // 100 | 1)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    component = _largest_center_component(candidate)

    if object_box is not None:
        x0, y0, x1, y1 = object_box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("--object-box needs four ordered values between 0 and 1")
        bounds = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
    elif component is not None and component.sum() > width * height * 0.015:
        ys, xs = np.where(component)
        margin = round(min(width, height) * 0.07)
        bounds = (max(border, int(xs.min()) - margin), max(border, int(ys.min()) - margin),
                  min(width - border, int(xs.max()) + margin), min(height - border, int(ys.max()) + margin))
    else:
        bounds = (round(width * .08), round(height * .07), round(width * .92), round(height * .93))

    x0, y0, x1, y1 = bounds
    gc = np.full((height, width), cv2.GC_BGD, np.uint8)
    gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
    dark = lab[:, :, 0].astype(float) < float(np.median(rim[:, 0])) - 45
    if dark.mean() > .06:
        distance_to_dark = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3)
        bright_background = (lab[:, :, 0] > np.median(rim[:, 0]) - 15) & (hsv[:, :, 1] < 90)
        gc[bright_background & (distance_to_dark > min(width, height) * .045)] = cv2.GC_BGD
    # Keep the central candidate as an anchor, while allowing GrabCut to grow
    # onto bright soles and other material that resembles the background.
    if component is not None:
        inner = cv2.erode(component, np.ones((7, 7), np.uint8))
        anchor = (inner > 0) & (difference > threshold * 1.2)
        anchor[:y0] = False
        anchor[y1:] = False
        anchor[:, :x0] = False
        anchor[:, x1:] = False
        gc[anchor] = cv2.GC_FGD
    if not np.any(gc == cv2.GC_FGD):
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        gc[max(y0, cy - 4):min(y1, cy + 4), max(x0, cx - 4):min(x1, cx + 4)] = cv2.GC_FGD
    cv2.grabCut(image, gc, None, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    component = _largest_center_component(mask)
    if component is None or component.sum() < width * height * .005:
        raise RuntimeError("Could not find an object mask; try --object-box")
    return cv2.dilate(component, np.ones((3, 3), np.uint8)).astype(np.uint8) * 255


def segment_images(images_dir, masks_dir, preview_path, object_box=None, backend="grabcut"):
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    session = None
    if backend == "rembg":
        try:
            from rembg import new_session
            session = new_session("isnet-general-use")
        except ImportError as exc:
            raise RuntimeError("rembg backend requires: pip install rembg onnxruntime") from exc
    coverages = []
    for index, path in enumerate(sorted(images_dir.glob("*.jpg"))):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        if session is None:
            mask = segment_frame(image, object_box)
        else:
            from rembg import remove
            rgba = remove(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), session=session)
            mask = (rgba[:, :, 3] > 100).astype(np.uint8) * 255
            component = _largest_center_component(mask > 0)
            if component is None:
                raise RuntimeError(f"Segmentation model found no object in {path}")
            mask = cv2.dilate(component, np.ones((3, 3), np.uint8)) * 255
        # COLMAP ImageReader.mask_path uses IMAGE_NAME.png and treats black as excluded.
        cv2.imwrite(str(masks_dir / f"{path.name}.png"), mask)
        coverages.append(float(np.mean(mask > 0)))
        if index == 0:
            overlay = image.copy()
            overlay[mask == 0] = (overlay[mask == 0] * .25).astype(np.uint8)
            Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), overlay)
    if not coverages:
        raise RuntimeError("No frames available for segmentation")
    return {"backend": backend, "frames": len(coverages), "mean_foreground_fraction": round(float(np.mean(coverages)), 3),
            "min_foreground_fraction": round(min(coverages), 3), "max_foreground_fraction": round(max(coverages), 3)}


def read_colmap_text(model_dir):
    """Read registered poses and SIMPLE_RADIAL/PINHOLE intrinsics from COLMAP TXT."""
    model_dir = Path(model_dir)
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cameras[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]), np.array(parts[4:], dtype=np.float64))
    views = []
    lines = [line for line in (model_dir / "images.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    for line in lines[::2]:
        parts = line.split()
        if len(parts) != 10 or not parts[0].isdigit():
            continue
        q = np.array(parts[1:5], dtype=np.float64)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rotation = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                             [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                             [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
        views.append((parts[9], cameras[int(parts[8])], rotation, np.array(parts[5:8], dtype=np.float64)))
    return views


def _project(points, camera, rotation, translation):
    model, width, height, params = camera
    xyz = points @ rotation.T + translation
    z = xyz[:, 2]
    valid = z > 1e-5
    x = xyz[:, 0] / np.maximum(z, 1e-5)
    y = xyz[:, 1] / np.maximum(z, 1e-5)
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        radial = 1 + k * (x*x + y*y)
        u, v = f * radial * x + cx, f * radial * y + cy
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        u, v = f*x + cx, f*y + cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        u, v = fx*x + cx, fy*y + cy
    else:
        raise ValueError(f"Unsupported camera model for mask projection: {model}")
    return u, v, valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def filter_points_by_masks(points, views, masks_dir, min_votes=2, min_fraction=.7, chunk_size=50000):
    """Vote across registered views without putting dense points on the GPU."""
    masks_dir = Path(masks_dir)
    loaded = []
    for name, camera, rotation, translation in views:
        mask = cv2.imread(str(masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            loaded.append((mask, camera, rotation, translation))
    if len(loaded) < 2:
        raise RuntimeError("Need at least two registered masked views for object-only filtering")
    keep = np.zeros(len(points), dtype=bool)
    support = np.zeros(len(points), dtype=np.uint8)
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start+chunk_size]
        fg = np.zeros(len(chunk), dtype=np.uint16)
        observed = np.zeros(len(chunk), dtype=np.uint16)
        for mask, camera, rotation, translation in loaded:
            u, v, valid = _project(chunk, camera, rotation, translation)
            ids = np.flatnonzero(valid)
            if len(ids):
                yy = np.clip(np.rint(v[ids]).astype(int), 0, mask.shape[0]-1)
                xx = np.clip(np.rint(u[ids]).astype(int), 0, mask.shape[1]-1)
                observed[ids] += 1
                fg[ids] += mask[yy, xx] > 127
        keep[start:start+len(chunk)] = (fg >= min_votes) & (fg / np.maximum(observed, 1) >= min_fraction)
        support[start:start+len(chunk)] = np.minimum(fg, 255).astype(np.uint8)
    return keep, support
