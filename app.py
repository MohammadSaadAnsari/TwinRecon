"""Local TwinRecon studio. Start with: python app.py"""

import json
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from scripts.preview import point_preview as pack_points, splat_preview as pack_splats


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "data" / "uploads"
ALLOWED = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
processes = {}


def _job_dir(job_id):
    if not re.fullmatch(r"job_[a-zA-Z0-9_-]+", job_id):
        abort(404)
    folder = OUTPUTS / job_id
    if not folder.is_dir():
        abort(404)
    return folder


def _report(folder):
    path = folder / "reconstruction.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    report["job_id"] = folder.name
    proc = processes.get(folder.name)
    if proc and proc.poll() is not None and report.get("status") == "running":
        report["status"] = "failed"
        report["error"] = f"Pipeline exited with status {proc.returncode}; see server.log"
    return report


def _artifact(folder, key):
    report = _report(folder)
    if not report:
        abort(404)
    relative = report.get("outputs", {}).get(key)
    if not relative:
        abort(404)
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        abort(404)
    return path


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/samples")
def samples():
    folder = ROOT / "data" / "inputs"
    return jsonify([p.name for p in sorted(folder.iterdir()) if p.suffix.lower() in ALLOWED] if folder.exists() else [])


@app.get("/api/capabilities")
def capabilities():
    return jsonify({"rembg": importlib.util.find_spec("rembg") is not None})


@app.get("/api/jobs")
def list_jobs():
    OUTPUTS.mkdir(exist_ok=True)
    folders = sorted((p for p in OUTPUTS.iterdir() if p.is_dir()),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    reports = [_report(p) for p in folders[:40]]
    return jsonify([r for r in reports if r])


@app.get("/api/jobs/<job_id>")
def job_detail(job_id):
    report = _report(_job_dir(job_id))
    return jsonify(report) if report else ("", 404)


@app.get("/api/jobs/<job_id>/log")
def job_log(job_id):
    folder = _job_dir(job_id)
    for name in ("pipeline.log", "server.log"):
        path = folder / name
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                return stream.read().decode("utf-8", "replace")
    return "Waiting for pipeline log…"


@app.post("/api/jobs")
def start_job():
    if any(proc.poll() is None for proc in processes.values()):
        return jsonify({"error": "A reconstruction is already using the GPU."}), 409
    sample = request.form.get("sample", "")
    upload = request.files.get("video")
    if upload and upload.filename:
        suffix = Path(secure_filename(upload.filename)).suffix.lower()
        if suffix not in ALLOWED:
            return jsonify({"error": "Use MP4, MOV, MKV, AVI or WebM video."}), 400
        UPLOADS.mkdir(parents=True, exist_ok=True)
        video = UPLOADS / f"{uuid4().hex}{suffix}"
        upload.save(video)
    elif sample:
        if Path(sample).name != sample or Path(sample).suffix.lower() not in ALLOWED:
            abort(400)
        video = ROOT / "data" / "inputs" / sample
        if not video.is_file():
            abort(404)
    else:
        return jsonify({"error": "Choose a video or a sample."}), 400
    job_id = f"job_{int(time.time()*1000)}_{uuid4().hex[:5]}"
    folder = OUTPUTS / job_id
    folder.mkdir(parents=True)
    command = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
               "--video", str(video), "--output-root", str(folder)]
    segmenter = request.form.get("segmenter", "grabcut")
    if segmenter not in ("grabcut", "rembg"):
        abort(400)
    command += ["--segmenter", segmenter]
    preset = request.form.get("preset", "balanced")
    settings = {"balanced": (50, 800, 350000, 7), "detail": (60, 960, 450000, 8)}.get(preset)
    if settings is None:
        return jsonify({"error": "Unknown reconstruction preset."}), 400
    frames, max_dim, max_splats, mesh_depth = settings
    command += ["--frames", str(frames), "--max-dim", str(max_dim),
                "--max-splats", str(max_splats), "--mesh-depth", str(mesh_depth)]
    if request.form.get("object_box", "").strip():
        try:
            box = [float(v) for v in request.form["object_box"].replace(",", " ").split()]
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Object box needs x0 y0 x1 y1, each between 0 and 1."}), 400
        command += ["--object-box", *map(str, box)]
    log = (folder / "server.log").open("w")
    try:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    finally:
        log.close()
    processes[job_id] = proc
    return jsonify({"job_id": job_id, "status": "starting"}), 202


@app.get("/api/jobs/<job_id>/files/<key>")
def download(job_id, key):
    if key not in {"point_cloud", "gaussian_splat", "poisson_mesh", "consistency_map",
                   "confidence_atlas", "atlas_data", "cameras", "segmentation_preview"}:
        abort(404)
    return send_file(_artifact(_job_dir(job_id), key), as_attachment=key != "segmentation_preview")


@app.get("/api/jobs/<job_id>/preview/<kind>")
def point_preview(job_id, kind):
    key = {"cloud": "point_cloud", "consistency": "consistency_map",
           "atlas": "confidence_atlas", "splats": "gaussian_splat"}.get(kind)
    if key is None:
        abort(404)
    folder = _job_dir(job_id)
    path = _artifact(folder, key)
    try:
        if kind == "splats":
            try:
                limit = min(max(int(request.args.get("limit", "160000")), 10000), 500000)
            except ValueError:
                return jsonify({"error": "Splat limit must be an integer."}), 400
            packed = pack_splats(folder, path, limit)
        else:
            packed = pack_points(folder, path)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 422
    return app.response_class(packed, mimetype="application/octet-stream")


@app.get("/api/jobs/<job_id>/atlas")
def atlas_data(job_id):
    return send_file(_artifact(_job_dir(job_id), "atlas_data"), mimetype="application/json")


@app.get("/api/compare")
def compare_jobs():
    before_id, after_id = request.args.get("before", ""), request.args.get("after", "")
    if before_id == after_id:
        return jsonify({"error": "Choose two different scans."}), 400
    before = json.loads(_artifact(_job_dir(before_id), "atlas_data").read_text())
    after = json.loads(_artifact(_job_dir(after_id), "atlas_data").read_text())

    def metrics(atlas):
        regions = atlas.get("regions", [])
        weights = [region["points"] for region in regions]
        total = max(sum(weights), 1)
        return {"weak_fraction": atlas["weak_region_fraction_by_points"],
                "floater_regions": atlas["floater_candidates"],
                "mean_camera_support": sum(region["camera_support"]*weight for region, weight in zip(regions, weights))/total,
                "mean_silhouette_agreement": sum(region["silhouette_agreement"]*weight for region, weight in zip(regions, weights))/total,
                "mean_image_error": sum(region["image_render_error"]*weight for region, weight in zip(regions, weights))/total,
                "mean_cloud_mesh_distance": sum(region["cloud_mesh_distance"]*weight for region, weight in zip(regions, weights))/total}

    return jsonify({"before_job": before_id, "after_job": after_id,
                    "before": metrics(before), "after": metrics(after),
                    "note": "Independent regional summaries. SfM scale and coordinates can differ; distances and region locations are not directly aligned."})


if __name__ == "__main__":
    app.run(host=os.environ.get("TWINRECON_HOST", "127.0.0.1"), port=int(os.environ.get("TWINRECON_PORT", "5000")), debug=False)
