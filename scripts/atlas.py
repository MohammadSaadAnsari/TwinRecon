"""Explainable regional reconstruction confidence and recapture directions.

The scores are evidence diagnostics, not calibrated accuracy probabilities.
Every measurement stays on CPU and uses a bounded cloud sample.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from segmentation import _project


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _scene_distance(points, mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    result = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), 25000):
        part = points[start:start+25000].astype(np.float32)
        result[start:start+len(part)] = scene.compute_distance(o3d.core.Tensor(part)).numpy()
    return result


def _view_evidence(points, colors, center, views, images_dir, masks_dir, depth_slack):
    count = len(points)
    observed = np.zeros(count, np.uint16)
    silhouette = np.zeros(count, np.uint16)
    errors = np.zeros(count, np.float32)
    error_views = np.zeros(count, np.uint16)
    chosen = np.linspace(0, len(views)-1, min(12, len(views)), dtype=int)
    used = []
    frame_audit = []
    for view_id in chosen:
        name, camera, rotation, translation = views[view_id]
        image = cv2.imread(str(Path(images_dir) / name))
        mask = cv2.imread(str(Path(masks_dir) / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        # A reduced point-render depth buffer rejects self-occluded points.
        u, v, valid = _project(points + center, camera, rotation, translation)
        xyz = (points + center) @ rotation.T + translation
        h, w = mask.shape
        scale = min(1., 320 / max(h, w))
        small_w, small_h = max(1, int(w*scale)), max(1, int(h*scale))
        uu = np.clip((u*small_w/w).astype(np.int32), 0, small_w-1)
        vv = np.clip((v*small_h/h).astype(np.int32), 0, small_h-1)
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        pixels = vv[ids]*small_w + uu[ids]
        depth = xyz[ids, 2]
        buffer = np.full(small_w*small_h, np.inf, np.float32)
        np.minimum.at(buffer, pixels, depth)
        visible_ids = ids[depth <= buffer[pixels] + depth_slack]
        if not len(visible_ids):
            continue
        px = np.clip(np.rint(u[visible_ids]).astype(int), 0, w-1)
        py = np.clip(np.rint(v[visible_ids]).astype(int), 0, h-1)
        foreground = mask[py, px] > 127
        observed[visible_ids] += 1
        silhouette[visible_ids] += foreground
        good_ids = visible_ids[foreground]
        if len(good_ids):
            rgb = image[py[foreground], px[foreground], ::-1].astype(np.float32) / 255.
            local_error = np.abs(rgb - colors[good_ids]).mean(axis=1)
            errors[good_ids] += local_error
            error_views[good_ids] += 1
            mean_error = float(np.mean(local_error))
        else:
            mean_error = 1.
        sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        frame_audit.append({"image": name, "visible_points": int(len(visible_ids)),
                            "mask_agreement": round(float(foreground.mean()), 3),
                            "point_render_error": round(mean_error, 3),
                            "sharpness": round(sharpness, 2)})
        used.append(name)
    image_error = np.divide(errors, error_views, out=np.ones_like(errors), where=error_views > 0)
    silhouette_ratio = np.divide(silhouette, observed, out=np.zeros(count, np.float32), where=observed > 0)
    return observed, silhouette_ratio, image_error, used, frame_audit


def _compass(regions, views, center):
    if not views or not regions:
        return []
    camera_positions = np.asarray([-rotation.T @ translation - center for _, _, rotation, translation in views])
    camera_dirs = np.asarray([_unit(position) for position in camera_positions])
    ups = np.asarray([-rotation.T[:, 1] for _, _, rotation, _ in views])
    up = _unit(ups.mean(axis=0))
    if np.linalg.norm(ups.mean(axis=0)) < .25:
        up = np.array([0., 1., 0.])
    current = camera_dirs[-1]
    forward = _unit(current - np.dot(current, up)*up)
    if np.linalg.norm(forward) < .5:
        forward = _unit(camera_dirs[0] - np.dot(camera_dirs[0], up)*up)
    right = _unit(np.cross(up, forward))
    weak = [region for region in regions if region["confidence"] < .8]
    if not weak:
        weak = sorted(regions, key=lambda item: item["confidence"])[:min(10, len(regions))]
    weak = sorted(weak, key=lambda item: item["priority"], reverse=True)[:80]
    candidates = []
    for azimuth in range(-180, 181, 30):
        for elevation in (-35, 0, 30, 55):
            az, el = np.deg2rad([azimuth, elevation])
            direction = _unit(np.cos(el)*(np.cos(az)*forward + np.sin(az)*right) + np.sin(el)*up)
            angular_gap = np.arccos(np.clip(camera_dirs @ direction, -1, 1)).min()
            novelty = float(np.clip(angular_gap / np.deg2rad(65), 0, 1))
            if novelty < .2:
                continue
            gains = []
            for region in weak:
                normal = np.asarray(region["normal"])
                visibility = max(0., float(np.dot(direction, normal)))
                gains.append(region["priority"] * visibility * novelty)
            score = float(sum(gains))
            if score <= 0:
                continue
            target = weak[int(np.argmax(gains))]
            candidates.append({"azimuth_degrees": azimuth, "elevation_degrees": elevation,
                               "direction": direction.round(4).tolist(), "score": score,
                               "target_region": target["id"], "target_center": target["center"],
                               "reason": target["reason"], "novelty": round(novelty, 3)})
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    picked = []
    for candidate in candidates:
        direction = np.asarray(candidate["direction"])
        if all(np.dot(direction, np.asarray(other["direction"])) < .86 for other in picked):
            picked.append(candidate)
        if len(picked) == 3:
            break
    if picked:
        peak = picked[0]["score"]
        for candidate in picked:
            candidate["relative_gain"] = round(candidate.pop("score") / peak, 3)
    return picked


def build_confidence_atlas(cloud, mesh, views, center, images_dir, masks_dir,
                           output_ply, output_json, max_points=120000):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    points_all = np.asarray(cloud.points)
    colors_all = np.asarray(cloud.colors)
    ids = np.linspace(0, len(points_all)-1, min(max_points, len(points_all)), dtype=int)
    points = points_all[ids]
    colors = colors_all[ids]
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    spacing = float(np.median(cKDTree(points).query(points[::max(1, len(points)//12000)], k=2)[0][:, 1]))
    distance = _scene_distance(points, mesh)
    threshold = max(spacing*2.5, 1e-6)
    observed, silhouette, image_error, used, frame_audit = _view_evidence(
        points, colors, np.asarray(center), views, images_dir, masks_dir, max(spacing*4, diagonal*.004))
    if len(used) < 2:
        raise RuntimeError("Atlas needs at least two registered images and masks")
    # Voxel cells are local evidence regions, not a claim of metric accuracy.
    cell_size = max(diagonal / 22, spacing * 6)
    origin = points.min(axis=0)
    cells = np.floor((points-origin)/cell_size).astype(np.int32)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_normals = np.asarray(mesh.vertex_normals)
    normal_tree = cKDTree(mesh_vertices)
    region_color = np.zeros((len(points), 3), np.float64)
    records = []
    for region_id in range(len(unique)):
        member = order[starts[region_id]:starts[region_id+1]]
        if len(member) < 12:
            continue
        camera_support = float(np.median(observed[member]))
        silhouette_agreement = float(np.mean(silhouette[member]))
        render_error = float(np.median(image_error[member]))
        mesh_distance = float(np.median(distance[member]))
        support_score = float(np.clip(camera_support / min(6., len(used)), 0, 1))
        render_score = float(np.exp(-render_error / .20))
        mesh_score = float(np.exp(-mesh_distance / threshold))
        confidence = float(.25*support_score + .25*silhouette_agreement +
                           .25*render_score + .25*mesh_score)
        reason_components = {"few camera views": 1-support_score,
                             "mask disagreement": 1-silhouette_agreement,
                             "image mismatch": 1-render_score,
                             "cloud / mesh gap": 1-mesh_score}
        reason = max(reason_components, key=reason_components.get)
        center_local = np.median(points[member], axis=0)
        normal = mesh_normals[normal_tree.query(center_local)[1]]
        if np.dot(normal, center_local) < 0:
            normal = -normal
        normal = _unit(normal)
        floater = bool((camera_support < 2.5 and mesh_distance > threshold) or
                       (silhouette_agreement < .5 and mesh_distance > threshold*.7))
        priority = float((1-confidence) * np.sqrt(len(member)) * (1.3 if floater else 1.))
        record = {"id": region_id, "center": center_local.round(5).tolist(),
                  "normal": normal.round(4).tolist(), "points": int(len(member)),
                  "confidence": round(confidence, 3), "camera_support": round(camera_support, 2),
                  "silhouette_agreement": round(silhouette_agreement, 3),
                  "image_render_error": round(render_error, 3),
                  "cloud_mesh_distance": round(mesh_distance, 6),
                  "floater_candidate": floater, "priority": round(priority, 4), "reason": reason}
        records.append(record)
        region_color[member] = ([.98, .34, .29] if floater else
                                [.98, .72, .31] if confidence < .8 else [.38, .86, .72])
    # Small cells borrow their closest scored region's color.
    missing = np.flatnonzero(~region_color.any(axis=1))
    if len(missing):
        region_color[missing] = [.65, .72, .74]
    diagnostic = o3d.geometry.PointCloud()
    diagnostic.points = o3d.utility.Vector3dVector(points)
    diagnostic.colors = o3d.utility.Vector3dVector(region_color)
    Path(output_ply).parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_ply), diagnostic, write_ascii=False):
        raise RuntimeError("Could not write confidence atlas PLY")
    compass = _compass(records, views, np.asarray(center))
    median_error = float(np.median([frame["point_render_error"] for frame in frame_audit]))
    median_sharpness = float(np.median([frame["sharpness"] for frame in frame_audit]))
    for frame in frame_audit:
        reasons = []
        if frame["point_render_error"] > max(.22, median_error*1.6):
            reasons.append("high image mismatch")
        if frame["sharpness"] < median_sharpness*.4:
            reasons.append("possibly blurred")
        if frame["mask_agreement"] < .65:
            reasons.append("mask disagreement")
        frame["review_reasons"] = reasons
    weak = sum(record["points"] for record in records if record["confidence"] < .8)
    total = sum(record["points"] for record in records)
    report = {"method": "CPU z-buffered point-render proxy + masks + Poisson distance",
              "ground_truth_accuracy_available": False, "sample_points": len(points),
              "views_used": used, "region_size_scene_units": round(cell_size, 6),
              "distance_threshold_scene_units": round(threshold, 6),
              "weak_confidence_threshold": .8,
              "weak_region_fraction_by_points": round(weak / max(total, 1), 4),
              "floater_candidates": sum(record["floater_candidate"] for record in records),
              "frame_audit": frame_audit,
              "regions": sorted(records, key=lambda record: record["priority"], reverse=True),
              "capture_compass": compass}
    Path(output_json).write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key not in ("regions", "frame_audit")}
