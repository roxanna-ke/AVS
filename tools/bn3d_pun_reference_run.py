"""Continuous PUN training with cached BN3D images and RefMetricTracker evaluation."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from PIL import Image

from bn3d_common import read_json, sha256, write_json
from bn3d_pun_native_v3 import PUN_SCALE, RADIUS, RANDOM100, set_seed

DEFAULT_ROOT = REPO.parents[1] / "outputs/bn3d_pun_v3_v2bright_random100/building1"
MILESTONES = {2000: (2000,), 20000: (2000, 5000, 10000, 15000, 20000)}
DIAGNOSTIC_INDICES = tuple(range(0, 100, 10))
RGB_METRICS = ("psnr", "ssim", "lpips", "rgb_loss")


def validate_capture(root: Path, lighting: str) -> dict:
    """Read the fixed capture and retain a digest of every consumed input."""
    if lighting not in ("original", "bright"):
        raise ValueError(lighting)
    asset = read_json(root / "asset_manifest.json")
    mesh_path = root / "asset/House.obj"
    if sha256(mesh_path) != asset["derived_sha256"] or asset["PUN_scale_on_derived_mesh"] != PUN_SCALE:
        raise ValueError("Derived mesh differs from the completed capture")
    source = root / lighting
    capture = read_json(source / "capture_complete.json")
    if (capture.get("lighting") != lighting or capture.get("num_training_views") != 100
            or RANDOM100 not in capture.get("policies", [])
            or capture.get("bright_lighting_mode") != "v2"):
        raise ValueError(f"Unexpected {lighting} random100 capture")
    camera = read_json(source / "eval_poses.json")["camera"]
    if (int(camera["w"]), int(camera["h"])) != (512, 512):
        raise ValueError("Expected PUN 512x512 camera")
    expected_focal = 256 / np.tan(np.deg2rad(30) / 2)
    if not np.allclose([camera["fl_x"], camera["fl_y"]], expected_focal, atol=1e-3):
        raise ValueError("Cached camera differs from PUN's 30-degree FOV")
    poses = np.load(source / "random100_poses.npy").astype(np.float32)
    eval_poses = np.load(source / "eval_poses.npy").astype(np.float32)
    if poses.shape != (100, 7) or eval_poses.shape != (40, 4, 4):
        raise ValueError("Expected 100 training poses and 40 evaluation poses")
    if not np.isfinite(poses).all() or not np.isfinite(eval_poses).all():
        raise ValueError("Non-finite camera pose")
    if np.linalg.norm(poses[:, None, 4:] - eval_poses[None, :, :3, 3], axis=-1).min() < 1e-5:
        raise ValueError("Training and evaluation poses overlap")
    digest = hashlib.sha256()
    images, references = [], []
    for split, count in (("training/random100/rgba", 100), ("eval/rgba", 40)):
        for index in range(count):
            path = source / split / f"{index:04d}.png"
            with Image.open(path) as file:
                if file.mode != "RGBA" or file.size != (512, 512):
                    raise ValueError(f"Invalid cached RGBA: {path}")
                rgba = np.asarray(file).copy()
            if not np.any(rgba[..., 3]):
                raise ValueError(f"Empty foreground: {path}")
            digest.update(str(path.relative_to(root)).encode())
            digest.update(bytes.fromhex(sha256(path)))
            (images if split.startswith("training") else references).append(rgba)
    for index, rgba in enumerate(references):
        for folder, expected in (("rgb_black", rgba[..., :3]), ("mask", rgba[..., 3])):
            path = source / "eval" / folder / f"{index:04d}.png"
            with Image.open(path) as file:
                actual = np.asarray(file)
            if not np.array_equal(actual, expected):
                raise ValueError(f"Evaluation GT does not match RGBA: {path}")
            digest.update(str(path.relative_to(root)).encode())
            digest.update(bytes.fromhex(sha256(path)))
    for path in (mesh_path, source / "capture_complete.json", source / "random100_poses.npy",
                 source / "eval_poses.npy", source / "eval_poses.json"):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return {"source": source, "mesh_path": mesh_path, "camera": camera,
            "images": images, "references": references, "poses": poses,
            "eval_poses": eval_poses, "input_sha256": digest.hexdigest()}


def validate_pair(root: Path) -> None:
    """Ensure lighting is the only intended difference between the two groups."""
    a, b = (root / name for name in ("original", "bright"))
    for filename in ("random100_poses.npy", "eval_poses.npy"):
        if not np.array_equal(np.load(a / filename), np.load(b / filename)):
            raise ValueError(f"Lighting groups have different {filename}")
    if read_json(a / "eval_poses.json")["camera"] != read_json(b / "eval_poses.json")["camera"]:
        raise ValueError("Lighting groups have different intrinsics")
    for index in range(40):
        relative = Path("eval/mask") / f"{index:04d}.png"
        if sha256(a / relative) != sha256(b / relative):
            raise ValueError(f"Lighting groups have different evaluation masks: {index}")


def make_config(run: Path, iterations: int, name: str):
    """Set scheduler and W&B before ActiveMapper constructs its trainer."""
    import yaml
    from nvf.active_mapping.active_mapping import ActiveMapper

    class ConfiguredMapper(ActiveMapper):
        def init_instant_ngp_config(self, config_home: str, dataset_path: str) -> str:
            path = super().init_instant_ngp_config(config_home, dataset_path)
            self.config.max_num_iterations = iterations
            self.config.optimizers["fields"]["scheduler"].max_steps = iterations
            self.config.vis = "wandb"
            self.config.experiment_name = name
            self.config.project_name = os.environ.get("WANDB_PROJECT", "bn3d-pun-reference")
            Path(path).write_text(yaml.dump(self.config))
            return path

    mapper = ConfiguredMapper()
    mapper.fov = 30
    mapper.train_img_size = (512, 512)
    os.environ["WANDB_NAME"] = name
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = str(run)
    config_path = mapper.initialize_config(str(run), str(run / "dataset"), model="instant-ngp")
    if (mapper.config.max_num_iterations != iterations or
            mapper.config.optimizers["fields"]["scheduler"].max_steps != iterations):
        raise RuntimeError("Trainer and scheduler lengths differ")
    return mapper, config_path


class CachedEnvironment:
    """The small subset of ShapeNetEnviroment used by RefMetricTracker."""

    def __init__(self, data: dict, mesh):
        camera = data["camera"]
        self.scene = SimpleNamespace(name="shapenet", mesh=mesh, width=int(camera["w"]),
                                     height=int(camera["h"]), fx=float(camera["fl_x"]),
                                     fy=float(camera["fl_y"]))
        self.references = data["references"]
        self.poses = data["eval_poses"]

    def get_images(self, mode: str, return_quat: bool = True):
        if mode != "eval" or return_quat:
            raise ValueError("Cached environment only serves matrix-form evaluation poses")
        import torch
        return self.references, [torch.from_numpy(pose.copy()) for pose in self.poses]


def reset_model_metrics(model) -> None:
    for name in RGB_METRICS:
        metric = getattr(model, name)
        if hasattr(metric, "reset"):
            metric.reset()


def rgb_evaluation(tracker, mapper) -> dict:
    """Use RefMetricTracker's camera rendering and exact RGB metric reduction."""
    import torch

    output = tracker.visualize(mapper).to(mapper.trainer.device).clip(0, 1)
    reference = tracker.ref_images.to(output).clip(0, 1)
    model = mapper.trainer.pipeline.model
    values = {}
    for name in RGB_METRICS:
        metric = getattr(model, name)
        if hasattr(metric, "reset"):
            metric.reset()
        values[name] = float(metric(output, reference).mean().item())
        if hasattr(metric, "reset"):
            metric.reset()
    return values


def training_diagnostics(data: dict, mapper) -> dict:
    """Foreground MSE on ten fixed, evenly spaced training views."""
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nvf.active_mapping.mapping_utils import to_transform

    model = mapper.trainer.pipeline.model
    camera = data["camera"]
    squared_error, pixels = 0.0, 0
    for index in DIAGNOSTIC_INDICES:
        pose = to_transform(torch.from_numpy(data["poses"][index])).cpu()
        ns_camera = Cameras(camera_to_worlds=pose[None, :3, :4],
                            fx=float(camera["fl_x"]), fy=float(camera["fl_y"]),
                            cx=float(camera["cx"]), cy=float(camera["cy"]),
                            width=512, height=512,
                            camera_type=CameraType.PERSPECTIVE).to(mapper.trainer.device)
        bundle = ns_camera.generate_rays(camera_indices=0, aabb_box=None)
        predicted = model.get_outputs_for_camera_ray_bundle(bundle)["rgb"].clamp(0, 1)
        rgba = data["images"][index]
        reference = torch.from_numpy(rgba[..., :3].astype(np.float32) / 255).to(predicted)
        mask = torch.from_numpy(rgba[..., 3] > 0).to(predicted.device)
        squared_error += float(((predicted - reference).square() * mask[..., None]).sum().item())
        pixels += int(mask.sum().item())
    mse = squared_error / (3 * pixels)
    return {"foreground_mse": mse, "foreground_psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "num_views": len(DIAGNOSTIC_INDICES)}


def full_geometry(tracker, mapper, mesh, poses, imported_axis_matrix) -> dict:
    """Compute native PUN values, then corrected values from the same point cloud."""
    import torch
    from nvf.metric.mesh_metrics import GeometryMeshMetric, VisibilityMeshMetric

    native, _ = tracker.get_metric(mapper)
    scaled = mesh.copy()
    scaled.apply_transform(np.asarray(imported_axis_matrix, dtype=np.float64))
    scaled.apply_scale(PUN_SCALE)
    scene = SimpleNamespace(**vars(tracker.env.scene))
    scene.mesh = scaled
    corrected_geometry = GeometryMeshMetric(scene=scene, num_sample_points=500000)
    corrected_geometry.nerf_pcd = tracker.geometry_metric.nerf_pcd
    acc, comp, ratio = corrected_geometry.get_outputs()
    corrected_visibility = VisibilityMeshMetric(scene=scene)
    corrected_visibility.update(torch.from_numpy(poses))
    vis, area = corrected_visibility.get_value()
    return {"native_pun": {key: float(value) for key, value in native.items()},
            "scaled_mesh": {"acc": float(acc), "comp": float(comp), "cr": float(ratio),
                            "vis": float(vis), "vis_area": float(area)},
            "mesh_scale": PUN_SCALE,
            "axis_transform": "asset_manifest.imported_axis_matrix"}


def run_one(root: Path, lighting: str, iterations: int, seed: int) -> None:
    import torch
    import trimesh
    import wandb
    from config import ExpConfig, ModelType, SceneType
    from fep_nbv.utils.utils import set_params
    from nvf.metric.MetricTracker import RefMetricTracker

    if iterations not in MILESTONES:
        raise ValueError("Supported trajectories: 2000 and 20000 steps")
    validate_pair(root)
    data = validate_capture(root, lighting)
    run = root / "runs_reference" / lighting / str(iterations)
    if run.exists():
        raise FileExistsError(f"Use a fresh run directory: {run}")
    if not torch.cuda.is_available():
        raise RuntimeError("PUN Instant-NGP training requires CUDA")
    if os.environ.get("WANDB_MODE", "online") == "offline":
        raise RuntimeError("This experiment requires W&B online mode")
    run.mkdir(parents=True)
    os.chdir(REPO)
    set_seed(seed)
    cfg = ExpConfig()
    cfg.env.scene = SceneType.shapenet
    cfg.model = ModelType.ngp
    cfg.env.fov = 30
    cfg.env.resolution = (512, 512)
    cfg.env.radius = cfg.radius = RADIUS
    cfg.env.scale = PUN_SCALE
    cfg.object_aabb = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    cfg.target_aabb = cfg.object_aabb.clone()
    cfg.camera_aabb = cfg.object_aabb * 2.8
    cfg.train_iter = iterations
    cfg.exp_folder = str(run)
    name = f"building1-{lighting}-random100-{iterations}"
    mapper, config_path = make_config(run, iterations, name)
    try:
        if wandb.run is None:
            raise RuntimeError("PUN's W&B writer did not start")
        mapper.reset()
        set_params(cfg, mapper)
        mapper.trainer.pipeline.model.renderer_entropy.set_iteration(0)
        mesh = trimesh.load(data["mesh_path"], force="mesh")
        tracker = RefMetricTracker(cfg, env=CachedEnvironment(data, mesh))
        training_poses = torch.from_numpy(data["poses"])
        tracker.visibility_metric.update(training_poses)
        write_json(run / "manifest.json", {
            "lighting": lighting, "iterations": iterations, "seed": seed,
            "input_sha256": data["input_sha256"], "capture": str(data["source"]),
            "mesh_sha256": sha256(data["mesh_path"]), "config": config_path,
            "scheduler_max_steps": mapper.config.optimizers["fields"]["scheduler"].max_steps,
            "wandb_run_id": wandb.run.id, "milestones": MILESTONES[iterations],
            "training_diagnostic_indices": DIAGNOSTIC_INDICES,
        })
        wandb.config.update({"lighting": lighting, "seed": seed, "iterations": iterations,
                             "input_sha256": data["input_sha256"], "num_training_views": 100,
                             "num_eval_views": 40, "scheduler_max_steps": iterations},
                            allow_val_change=True)

        def on_step(step: int, force_final: bool = False) -> None:
            if step not in MILESTONES[iterations]:
                return
            # ActiveMapper.add_image also trains PUN's NVF heads after the NeRF
            # loop. Evaluate/save the final state after that native extra stage.
            if step == iterations and not force_final:
                return
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all()
            pipeline = mapper.trainer.pipeline
            was_training = pipeline.training
            try:
                with torch.no_grad():
                    checkpoint = run / f"step_{step:05d}.ckpt"
                    mapper.save_ckpt(str(checkpoint), step=step)
                    rgb = rgb_evaluation(tracker, mapper)
                    train = training_diagnostics(data, mapper)
                    result = {"step": step, "rgb": rgb, "train": train,
                              "checkpoint": str(checkpoint)}
                    if step == iterations:
                        asset = read_json(root / "asset_manifest.json")
                        result["geometry"] = full_geometry(
                            tracker, mapper, mesh, data["poses"], asset["imported_axis_matrix"])
                    write_json(run / f"metrics_{step:05d}.json", result)
                    logs = {f"eval/{key}": value for key, value in rgb.items()}
                    logs.update({f"train_fixed/{key}": value for key, value in train.items()
                                 if key != "num_views"})
                    if "geometry" in result:
                        for family, metrics in result["geometry"].items():
                            if isinstance(metrics, dict):
                                logs.update({f"geometry/{family}/{key}": value
                                             for key, value in metrics.items()})
                    wandb.log(logs, step=step)
                    print(f"Milestone {step}: {rgb}", flush=True)
            finally:
                reset_model_metrics(pipeline.model)
                pipeline.train(was_training)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.random.set_rng_state(torch_state)
                torch.cuda.set_rng_state_all(cuda_states)

        mapper.trainer.milestone_callback = on_step
        mapper.add_image(images=data["images"], poses=list(training_poses), model_option=None)
        on_step(iterations, force_final=True)
        expected = [run / f"metrics_{step:05d}.json" for step in MILESTONES[iterations]]
        if not all(path.is_file() for path in expected):
            raise RuntimeError("One or more milestone evaluations did not complete")
        write_json(run / "complete.json", {"milestones": list(MILESTONES[iterations]),
                                            "wandb_run_id": wandb.run.id,
                                            "input_sha256": data["input_sha256"]})
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lighting", choices=("original", "bright"), required=True)
    parser.add_argument("--iterations", type=int, choices=(2000, 20000), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_one(args.root.resolve(), args.lighting, args.iterations, args.seed)


if __name__ == "__main__":
    main()
