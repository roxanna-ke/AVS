"""Evaluate saved BN3D NeRFs with PUN RefMetricTracker's image metric protocol.

Example: python tools/bn3d_eval.py --profile pun_rendered --subset avs20
No training or mesh extraction. Test RGBs and cameras come from test40 by default.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
sys.path.insert(0, str(REPO))

from bn3d_orbit_video import select_checkpoint


def read_test_data(path):
    """Read explicit BN3D intrinsics and OpenGL poses; do not normalize cameras."""
    path = path / "transforms.json" if path.is_dir() else path
    meta = json.loads(path.read_text())
    frames = meta["frames"]
    if not frames:
        raise ValueError("The test manifest has no frames")
    if meta.get("camera_model", "OPENCV") not in ("OPENCV", "PINHOLE"):
        raise ValueError("Only BN3D perspective cameras are supported")
    camera_keys = ("w", "h", "fl_x", "fl_y", "cx", "cy")
    cameras, images, paths = [], [], []
    for frame in frames:
        camera = {key: frame.get(key, meta.get(key)) for key in camera_keys}
        if any(v is None or not np.isfinite(v) for v in camera.values()):
            raise ValueError(f"Missing or invalid intrinsics: {camera}")
        if any(camera[k] <= 0 for k in ("w", "h", "fl_x", "fl_y")):
            raise ValueError(f"Non-positive camera dimensions or focal lengths: {camera}")
        if any(float(frame.get(k, meta.get(k, 0))) != 0 for k in ("k1", "k2", "k3", "k4", "p1", "p2")):
            raise ValueError("This BN3D adapter requires undistorted test images")
        pose = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("Invalid OpenGL camera-to-world matrix")
        image_path = (path.parent / frame["file_path"]).resolve()
        with Image.open(image_path) as image:
            if image.mode not in ("RGB", "RGBA"):
                raise ValueError(f"Expected RGB/RGBA, got {image.mode}: {image_path}")
            if image.size != (camera["w"], camera["h"]):
                raise ValueError(f"Image dimensions disagree with intrinsics: {image_path}")
            # Exactly RefMetricTracker: take RGB channels, float32 / 255, BCHW.
            # BN3D's default test PNGs are already composited over white.
            images.append(np.asarray(image, dtype=np.float32)[..., :3].transpose(2, 0, 1) / 255.0)
        cameras.append(dict(camera, pose=pose))
        paths.append(str(image_path))
    if len({image.shape for image in images}) != 1:
        raise ValueError("PUN batch aggregation requires equal test image dimensions")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate test image paths")
    return path.resolve(), cameras, np.stack(images), paths


def pun_image_metrics(model, prediction, reference):
    """Same batch calls as nvf.metric.MetricTracker.RefMetricTracker.get_metric.

    In particular, PSNR is computed over the batch, NOT averaged per-image PSNR.
    Reset stateful TorchMetrics so every invocation is an independent evaluation.
    """
    metrics = {}
    for name in ("psnr", "ssim", "lpips", "rgb_loss"):
        metric = getattr(model, name)
        if hasattr(metric, "reset"):
            metric.reset()
        value = metric(prediction.clip(0.0, 1.0), reference.clip(0.0, 1.0))
        metrics[name] = value.mean().item()
        if hasattr(metric, "reset"):
            metric.reset()
    metrics["mse"] = metrics["rgb_loss"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=["pun_rendered", "bn3d_faithful"])
    parser.add_argument("--subset", required=True, choices=["avs20", "random20"])
    parser.add_argument("--steps", type=int, default=30000, help="Completed training iterations for automatic run selection.")
    parser.add_argument("--config", type=Path, help="Use a particular training config.yml.")
    parser.add_argument("--checkpoint", type=Path, help="Explicit step-XXXXXXXXX.ckpt; overrides automatic step selection.")
    parser.add_argument("--test-data", type=Path, help="Test directory or transforms.json; defaults to profile/test40.")
    parser.add_argument("--runs-root", type=Path, default=WORKSPACE / "outputs/bn3d_building1")
    parser.add_argument("--building-dir", type=Path, default=WORKSPACE / "data/bn3d_processed/building1")
    parser.add_argument("--output", type=Path, help="JSON output; default: selected run/eval/pun_test40_metrics.json.")
    parser.add_argument("--rays-per-chunk", type=int, default=8192)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without loading a model or writing results.")
    args = parser.parse_args()
    if args.steps <= 0 or args.rays_per_chunk <= 0:
        parser.error("Steps and rays-per-chunk must be positive")
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        config_path = (args.config or checkpoint.parent.parent / "config.yml").resolve()
    else:
        config_path, checkpoint = select_checkpoint(args)
        config_path, checkpoint = config_path.resolve(), checkpoint.resolve()
    match = re.fullmatch(r"step-(\d{9})\.ckpt", checkpoint.name)
    if not match or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Expected a nonempty step-XXXXXXXXX.ckpt: {checkpoint}")
    step = int(match.group(1))
    metadata = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    if metadata["method_name"] != "instant-ngp":
        raise ValueError("This adapter is for the BN3D instant-ngp reconstruction runs")
    if metadata["experiment_name"] != f"bn3d_building1_{args.profile}_{args.subset}":
        raise ValueError("Config experiment_name disagrees with --profile/--subset")
    test_path, cameras, references, image_paths = read_test_data(
        args.test_data or args.building_dir / args.profile / "test40")
    # Inspect original training data, preserving its scene AABB rather than
    # replacing the dataparser with a test manifest that lacks aabb_scale.
    dm = metadata["pipeline"]["datamanager"]
    train_data = dm.get("data")
    if train_data in (None, "null", "None"):
        train_data = dm["dataparser"]["data"]
    train_data = Path(*train_data) if isinstance(train_data, list) else Path(train_data)
    train_path = train_data / "transforms.json" if train_data.is_dir() else train_data
    train_meta = json.loads(train_path.read_text())
    train_poses = np.asarray([f["transform_matrix"] for f in train_meta["frames"]])
    if float(dm["dataparser"]["scene_scale"]) != 1.0:
        raise ValueError("Expected canonical BN3D scene_scale=1.0")
    for camera in cameras:
        if np.any(np.all(np.isclose(train_poses, camera["pose"], rtol=0, atol=1e-5), axis=(1, 2))):
            raise ValueError("A test pose overlaps a training pose; refusing train/test leakage")
    output = (args.output or config_path.parent / "eval" / "pun_test40_metrics.json").resolve()
    if output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    print(f"Config: {config_path}\nCheckpoint: {checkpoint}\nTest data: {test_path}\n"
          f"Images: {len(cameras)}; shape: {references.shape}; no overlapping train poses\n"
          f"Output: {output}", flush=True)
    if args.check_only:
        return
    if output.exists():
        raise FileExistsError(f"Output exists; choose another --output: {output}")

    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.configs.method_configs import all_methods
    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Run in a GPU environment with the PUN dependencies")
    # Same setup as bundled eval_setup, but pin an explicit checkpoint instead
    # of silently replacing load_dir with the path encoded in the saved YAML.
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.pipeline.model.eval_num_rays_per_chunk = args.rays_per_chunk
    config.load_dir, config.load_step = checkpoint.parent, step
    pipeline = config.pipeline.setup(device=torch.device("cuda"), test_mode="inference")
    loaded_checkpoint, loaded_step = eval_load_checkpoint(config, pipeline)
    if loaded_checkpoint.resolve() != checkpoint or loaded_step != step:
        raise RuntimeError("Loaded checkpoint does not match the requested checkpoint")
    pipeline.eval()
    with torch.no_grad():
        predictions = []
        for index, camera in enumerate(cameras):
            ns_camera = Cameras(
                camera_to_worlds=torch.from_numpy(camera["pose"][None, :3, :4]),
                fx=float(camera["fl_x"]), fy=float(camera["fl_y"]),
                cx=float(camera["cx"]), cy=float(camera["cy"]),
                width=int(camera["w"]), height=int(camera["h"]),
                camera_type=CameraType.PERSPECTIVE,
            ).to("cuda")
            rays = ns_camera.generate_rays(camera_indices=0, aabb_box=None)
            rgb = pipeline.model.get_outputs_for_camera_ray_bundle(rays)["rgb"]
            predictions.append(rgb.permute(2, 0, 1).cpu())
            print(f"Rendered {index + 1}/{len(cameras)}", flush=True)
        prediction = torch.stack(predictions).to("cuda")
        reference = torch.from_numpy(references).to(prediction)
        results = pun_image_metrics(pipeline.model, prediction, reference)
    report = {
        "profile": args.profile, "subset": args.subset,
        "config": str(config_path), "checkpoint": str(checkpoint),
        "checkpoint_step": step, "training_iterations": step + 1,
        "test_transforms": str(test_path), "num_test_images": len(cameras),
        "test_images": image_paths, "image_shape_chw": list(references.shape[1:]),
        "protocol": "PUN nvf.metric.MetricTracker.RefMetricTracker image metrics",
        "aggregation": "One full BCHW batch; metric(pred.clip(0,1), gt.clip(0,1)).mean()",
        "reference_preprocessing": "RGB channels / 255; no extra alpha compositing, masking or cropping",
        "camera_source": "Explicit BN3D intrinsics and canonical OpenGL poses; no FOV approximation",
        "mse_definition": "Alias of PUN model.rgb_loss (torch.nn.MSELoss)",
        "training_aabb_scale": train_meta.get("aabb_scale", 1),
        "background_color": config.pipeline.model.background_color,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
