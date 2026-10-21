"""Short, validation-gated gsplat refinement for a 4 GB laptop GPU.

Only a deterministic subset of the exported Gaussians is optimized. The
original PLY is retained unless held-out masked views improve. This module is
imported lazily so CPU-only diagnostics still work without torch or gsplat.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


LOG = logging.getLogger("twinrecon")
SH_C0 = 0.28209479177387814


def _camera_frame(view, center, images_dir, masks_dir, max_side=320):
    name, (model, width, height, params), rotation, translation = view
    image = cv2.imread(str(Path(images_dir) / name))
    mask = cv2.imread(str(Path(masks_dir) / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        distortion = np.array([k, 0, 0, 0, 0], dtype=np.float32)
        image = cv2.undistort(image, K, distortion)
        mask = cv2.undistort(mask, K, distortion, None, K)
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        return None
    target_width = max(32, round(width * min(1., max_side / max(width, height))))
    target_height = max(32, round(height * target_width / width))
    K[0] *= target_width / width
    K[1] *= target_height / height
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = rotation @ center + translation
    return {"name": name, "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.,
            "mask": (mask > 127).astype(np.float32), "K": K, "viewmat": viewmat}


def _sample_views(views, center, images_dir, masks_dir):
    chosen = np.linspace(0, len(views)-1, min(12, len(views)), dtype=int)
    frames = [_camera_frame(views[i], center, images_dir, masks_dir) for i in chosen]
    frames = [frame for frame in frames if frame is not None and frame["mask"].mean() > .03]
    if len(frames) < 6:
        return [], []
    validation = [frame for i, frame in enumerate(frames) if i % 4 == 3]
    training = [frame for i, frame in enumerate(frames) if i % 4 != 3]
    return training, validation


def refine_splats(ply_path, views, center, images_dir, masks_dir, max_gaussians=80000,
                  max_steps=32, sh_steps=12):
    """Refine DC/shape, then accept SH only if separate views improve.

    Returns a JSON-safe stage report. OOM or unavailable CUDA leaves the PLY
    untouched. An accepted subset is patched into the dense original export.
    """
    result = {"status": "skipped", "reason": None, "held_out_views": [],
              "dc_accepted": False, "sh_accepted": False, "optimized_gaussians": 0}
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:
        result["reason"] = f"torch/gsplat unavailable: {exc}"
        return result
    if not torch.cuda.is_available():
        result["reason"] = "CUDA driver unavailable; retained measured MVS splat"
        return result

    training, validation = _sample_views(views, np.asarray(center), images_dir, masks_dir)
    if not validation:
        result["reason"] = "fewer than six usable registered, masked views"
        return result
    result["held_out_views"] = [frame["name"] for frame in validation]
    vertices = PlyData.read(str(ply_path))["vertex"].data.copy()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 900 * 1024**2:
        result["reason"] = "less than 900 MiB CUDA memory free"
        return result
    budget = min(max_gaussians, 40000 if free_bytes < 1600*1024**2 else max_gaussians)
    ids = np.linspace(0, len(vertices)-1, min(budget, len(vertices)), dtype=np.int64)
    result["optimized_gaussians"] = len(ids)
    device = "cuda"
    try:
        def column(names):
            return np.column_stack([vertices[name][ids] for name in names]).astype(np.float32)

        original_means = torch.from_numpy(column(("x", "y", "z"))).to(device)
        original_logs = torch.from_numpy(column(("scale_0", "scale_1", "scale_2"))).to(device)
        quats = torch.from_numpy(column(("rot_0", "rot_1", "rot_2", "rot_3"))).to(device)
        opacity = torch.sigmoid(torch.from_numpy(vertices["opacity"][ids].copy()).to(device))
        rgb = np.clip(.5 + SH_C0 * column(("f_dc_0", "f_dc_1", "f_dc_2")), .005, .995)
        original_rgb = torch.from_numpy(rgb).to(device)
        logit_rgb = torch.nn.Parameter(torch.logit(original_rgb.clone()))
        displacement = torch.nn.Parameter(torch.zeros_like(original_means))
        log_scale_delta = torch.nn.Parameter(torch.zeros_like(original_logs))
        extent = float(np.linalg.norm(np.ptp(column(("x", "y", "z")), axis=0)))
        max_move = max(extent * .002, 1e-5)
        source = training + validation
        for frame in source:
            frame["target"] = torch.from_numpy(frame["rgb"]).to(device)
            frame["foreground"] = torch.from_numpy(frame["mask"])[..., None].to(device)
            frame["K_gpu"] = torch.from_numpy(frame["K"])[None].to(device)
            frame["view_gpu"] = torch.from_numpy(frame["viewmat"])[None].to(device)

        def render(frame, sh=None):
            means = original_means + torch.tanh(displacement) * max_move
            scales = torch.exp(original_logs + torch.clamp(log_scale_delta, -.35, .35))
            if sh is None:
                colors = torch.sigmoid(logit_rgb)
                degree = None
            else:
                dc = (torch.sigmoid(logit_rgb) - .5) / SH_C0
                colors = torch.cat((dc[:, None, :], sh), dim=1)
                degree = 3
            return rasterization(means, quats, scales, opacity, colors, frame["view_gpu"],
                                 frame["K_gpu"], frame["rgb"].shape[1], frame["rgb"].shape[0],
                                 sh_degree=degree, packed=True, sparse_grad=False,
                                 near_plane=max(extent*.003, 1e-4),
                                 backgrounds=torch.zeros((1, 3), device=device))[:2]

        def loss(frame, sh=None):
            image, alpha = render(frame, sh)
            image, alpha = image[0], alpha[0]
            mask = frame["foreground"]
            visible = mask * (alpha.detach() > .12).float()
            # The black-background render is premultiplied by alpha. Compare
            # recovered foreground radiance so translucent edges cannot bleach.
            radiance = image / alpha.clamp_min(.12)
            color = (torch.abs(radiance - frame["target"]) * visible).sum() / (visible.sum()*3 + 1)
            silhouette = (torch.abs(alpha - mask) * .08).mean()
            return color + silhouette

        with torch.no_grad():
            baseline = float(np.mean([loss(frame).item() for frame in validation]))
        result["validation_before"] = round(baseline, 6)
        best_loss = baseline
        best = None
        optimizer = torch.optim.Adam([{"params": [logit_rgb], "lr": .006},
                                      {"params": [displacement], "lr": .012},
                                      {"params": [log_scale_delta], "lr": .003}])
        stalls = 0
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            frame = training[step % len(training)]
            objective = loss(frame) + .002 * displacement.square().mean() + .001 * log_scale_delta.square().mean()
            objective.backward()
            optimizer.step()
            if step % 4 == 3 or step == max_steps-1:
                with torch.no_grad():
                    score = float(np.mean([loss(view).item() for view in validation]))
                if np.isfinite(score) and score < best_loss - 1e-4:
                    best_loss = score
                    best = (logit_rgb.detach().clone(), displacement.detach().clone(),
                            log_scale_delta.detach().clone())
                    stalls = 0
                else:
                    stalls += 1
                if stalls >= 3:
                    break
        result["dc_steps"] = step + 1
        result["validation_after_dc"] = round(best_loss, 6)
        if best is None or best_loss >= baseline * .995:
            result["reason"] = "held-out views did not improve by 0.5%; original preserved"
            return result
        with torch.no_grad():
            logit_rgb.copy_(best[0]); displacement.copy_(best[1]); log_scale_delta.copy_(best[2])
        result["dc_accepted"] = True

        sh_best = None
        if "f_rest_44" in vertices.dtype.names and len(training) >= 6:
            rest = torch.nn.Parameter(torch.zeros((len(ids), 15, 3), device=device))
            sh_optimizer = torch.optim.Adam([rest], lr=.002)
            sh_score = best_loss
            stalls = 0
            for sh_step in range(sh_steps):
                sh_optimizer.zero_grad(set_to_none=True)
                objective = loss(training[sh_step % len(training)], rest) + .025*rest.square().mean()
                objective.backward()
                sh_optimizer.step()
                if sh_step % 4 == 3 or sh_step == sh_steps-1:
                    with torch.no_grad():
                        score = float(np.mean([loss(view, rest).item() for view in validation]))
                    if np.isfinite(score) and score < sh_score - 1e-4:
                        sh_score = score
                        sh_best = rest.detach().clone()
                        stalls = 0
                    else:
                        stalls += 1
                    if stalls >= 2:
                        break
            result["sh_steps"] = sh_step + 1
            result["validation_after_sh"] = round(sh_score, 6)
            result["sh_accepted"] = sh_best is not None and sh_score < best_loss * .995

        final_rgb = torch.sigmoid(logit_rgb).detach().cpu().numpy()
        final_means = (original_means + torch.tanh(displacement)*max_move).detach().cpu().numpy()
        final_logs = (original_logs + torch.clamp(log_scale_delta, -.35, .35)).detach().cpu().numpy()
        delta_rgb = final_rgb - rgb
        delta_means = final_means - original_means.cpu().numpy()
        delta_logs = final_logs - original_logs.cpu().numpy()
        original_positions = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
        tree = cKDTree(original_positions[ids])
        sampled = original_positions[ids][::max(1, len(ids)//10000)]
        sample_spacing = float(np.median(tree.query(sampled, k=2)[0][:, 1]))
        influence = max(sample_spacing*1.5, 1e-6)
        if result["sh_accepted"]:
            # Standard 3DGS PLY stores remaining coefficients channel-major.
            sh_cpu = sh_best.cpu().numpy().transpose(0, 2, 1).reshape(len(ids), 45)
        for start in range(0, len(vertices), 40000):
            stop = min(start+40000, len(vertices))
            distances, nearest = tree.query(original_positions[start:stop], k=1)
            weight = np.exp(-.5*(distances/influence)**2).astype(np.float32)
            weight[distances > influence*2.5] = 0
            original_color = np.column_stack([vertices[f"f_dc_{axis}"][start:stop] for axis in range(3)])*SH_C0 + .5
            adjusted_color = np.clip(original_color + delta_rgb[nearest]*weight[:, None], 0, 1)
            adjusted_means = original_positions[start:stop] + delta_means[nearest]*weight[:, None]
            for axis, name in enumerate(("x", "y", "z")):
                vertices[name][start:stop] = adjusted_means[:, axis]
                vertices[f"scale_{axis}"][start:stop] += delta_logs[nearest, axis]*weight
                vertices[f"f_dc_{axis}"][start:stop] = (adjusted_color[:, axis]-.5)/SH_C0
            if result["sh_accepted"]:
                for channel in range(45):
                    vertices[f"f_rest_{channel}"][start:stop] = sh_cpu[nearest, channel]*weight
        target = Path(ply_path)
        temporary = target.with_suffix(".refined.tmp")
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(temporary))
        temporary.replace(target)
        result["status"] = "accepted"
        result["validation_final"] = result.get("validation_after_sh") if result["sh_accepted"] else result["validation_after_dc"]
        return result
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        LOG.warning("Refinement ran out of CUDA memory; retaining original PLY")
        result["reason"] = "CUDA out of memory; original preserved"
        return result
    finally:
        torch.cuda.empty_cache()
