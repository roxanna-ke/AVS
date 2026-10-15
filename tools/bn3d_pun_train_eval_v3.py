"""Train one BN3D/PUN v3 instant-ngp run with full and masked metrics."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from PIL import Image

from bn3d_common import read_json, write_json
from bn3d_pun_native_v3 import DEFAULT_ROOT, POLICIES, RANDOM100, RADIUS, set_seed


def read_training(folder: Path, count: int = 20) -> list[np.ndarray]:
    images = []
    for i in range(count):
        with Image.open(folder / f"{i:04d}.png") as image:
            if image.mode != "RGBA" or image.size != (512, 512):
                raise ValueError(f"Invalid PUN RGBA training image: {image.filename}")
            images.append(np.array(image))
    return images


def render_predictions(mapper, eval_poses: np.ndarray, camera: dict, folder: Path,
                       label: str = "evaluation") -> None:
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType

    model = mapper.trainer.pipeline.model
    mapper.trainer.pipeline.eval()
    folder.mkdir(parents=True, exist_ok=True)
    for i, pose in enumerate(eval_poses):
        ns_camera = Cameras(
            camera_to_worlds=torch.tensor(pose[None, :3, :4], dtype=torch.float32),
            fx=float(camera["fl_x"]), fy=float(camera["fl_y"]),
            cx=float(camera["cx"]), cy=float(camera["cy"]),
            width=int(camera["w"]), height=int(camera["h"]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(mapper.trainer.device)
        with torch.no_grad():
            bundle = ns_camera.generate_rays(camera_indices=0, aabb_box=None)
            rgb = model.get_outputs_for_camera_ray_bundle(bundle)["rgb"].cpu().numpy()
        if rgb.shape != (512, 512, 3) or not np.isfinite(rgb).all():
            raise ValueError(f"Invalid NeRF output for view {i}")
        np.save(folder / f"{i:04d}.npy", rgb.astype(np.float32))
        Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(folder / f"{i:04d}.png")
        print(f"NeRF {label} render {i + 1}/{len(eval_poses)}", flush=True)


def native_metrics(model, pred, ref) -> dict:
    values = {}
    for name in ("psnr", "ssim", "lpips", "rgb_loss"):
        fn = getattr(model, name)
        if hasattr(fn, "reset"):
            fn.reset()
        value = fn(pred.clamp(0, 1), ref.clamp(0, 1))
        values[name] = float(value.mean().item())
        if hasattr(fn, "reset"):
            fn.reset()
    return values


def foreground_metrics(model, pred, ref, mask) -> dict:
    """Masked reductions of PUN's MSE/SSIM/LPIPS definitions.

    SSIM uses the original TorchMetrics local map. LPIPS uses the same trained
    network and linear layers with its spatial output enabled. These masked
    reductions are extensions, not the repository's original full-image scores.
    """
    import torch
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure

    if mask.shape != pred[:, :1].shape or not bool(mask.any()):
        raise ValueError("Empty or mismatched foreground mask")
    error = (pred - ref).square()
    mse = (error * mask).sum() / (mask.sum() * 3)
    _, ssim_map = structural_similarity_index_measure(
        pred, ref, data_range=1.0, return_full_image=True)
    h, w = mask.shape[-2:]
    top = (ssim_map.shape[-2] - h) // 2
    left = (ssim_map.shape[-1] - w) // 2
    ssim_map = ssim_map[..., top:top + h, left:left + w]
    if ssim_map.shape[1] != 1:
        ssim_map = ssim_map.mean(dim=1, keepdim=True)
    lpips_net = model.lpips.net
    old_spatial = lpips_net.spatial
    lpips_net.spatial = True
    try:
        spatial = lpips_net(pred, ref, normalize=True)
    finally:
        lpips_net.spatial = old_spatial
    if spatial.shape[-2:] != (h, w):
        raise ValueError(f"Unexpected LPIPS spatial map: {spatial.shape}")
    return {
        "psnr": float((-10 * torch.log10(mse)).item()),
        "ssim": float(((ssim_map * mask).sum() / mask.sum()).item()),
        "lpips": float(((spatial * mask).sum() / mask.sum()).item()),
        "rgb_loss": float(mse.item()),
    }


def evaluate_saved(model, source: Path, predictions: Path, out: Path, count: int = 40) -> None:
    import torch
    pred_list, ref_list, masks = [], [], []
    for i in range(count):
        pred = np.load(predictions / f"{i:04d}.npy")
        with Image.open(source / "eval/rgb_black" / f"{i:04d}.png") as image:
            ref = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(source / "eval/mask" / f"{i:04d}.png") as image:
            alpha = np.array(image.convert("L"), dtype=np.uint8)
        if pred.shape != ref.shape or pred.shape != (512, 512, 3):
            raise ValueError(f"Prediction/GT shape mismatch at view {i}")
        if not np.any(alpha):
            raise ValueError(f"Empty GT mask at view {i}")
        pred_list.append(pred.transpose(2, 0, 1))
        ref_list.append(ref.transpose(2, 0, 1))
        masks.append((alpha > 0)[None])
    device = next(model.parameters()).device
    pred = torch.from_numpy(np.stack(pred_list)).to(device).clamp(0, 1)
    ref = torch.from_numpy(np.stack(ref_list)).to(device).clamp(0, 1)
    mask = torch.from_numpy(np.stack(masks)).to(device)
    with torch.no_grad():
        # Exactly RefMetricTracker's batch calls and reduction for full images.
        full = native_metrics(model, pred, ref)
        foreground = foreground_metrics(model, pred, ref, mask)
        per_view = []
        for i in range(count):
            per_view.append({"index": i,
                "full": native_metrics(model, pred[i:i + 1], ref[i:i + 1]),
                "foreground": foreground_metrics(model, pred[i:i + 1], ref[i:i + 1], mask[i:i + 1])})
    write_json(out / "metrics.json", {
        "full": full, "foreground": foreground, "per_view": per_view,
        "full_protocol": "PUN RefMetricTracker: model metric(pred.clip(0,1), gt.clip(0,1)).mean() over BCHW batch",
        "foreground_protocol": "GT alpha>0; pooled RGB MSE/PSNR; masked TorchMetrics SSIM local map; masked PUN LPIPS spatial map",
        "num_eval_views": count,
    })


def evaluate_training_saved(model, source: Path, predictions: Path, out: Path,
                            count: int) -> None:
    """Measure training-view fit on foreground pixels without stacking 100 images."""
    import torch

    device = next(model.parameters()).device
    weighted_mse = 0.0
    foreground_pixels = 0
    per_view = []
    for i in range(count):
        pred = np.load(predictions / f"{i:04d}.npy")
        with Image.open(source / "training" / RANDOM100 / "rgba" / f"{i:04d}.png") as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if pred.shape != (512, 512, 3) or rgba.shape != (512, 512, 4):
            raise ValueError(f"Training prediction/GT shape mismatch at view {i}")
        mask_np = rgba[..., 3] > 0
        pixels = int(mask_np.sum())
        if not pixels:
            raise ValueError(f"Empty training foreground at view {i}")
        predicted = torch.from_numpy(pred.transpose(2, 0, 1)[None]).to(device).clamp(0, 1)
        reference = torch.from_numpy(rgba[..., :3].transpose(2, 0, 1)[None].astype(np.float32) / 255).to(device)
        mask = torch.from_numpy(mask_np[None, None]).to(device)
        with torch.no_grad():
            values = foreground_metrics(model, predicted, reference, mask)
        weighted_mse += values["rgb_loss"] * pixels
        foreground_pixels += pixels
        per_view.append({"index": i, "foreground": values})
    mse = weighted_mse / foreground_pixels
    write_json(out / "train_metrics.json", {
        "foreground": {
            "psnr": float(-10 * np.log10(mse)),
            "ssim": float(np.mean([row["foreground"]["ssim"] for row in per_view])),
            "lpips": float(np.mean([row["foreground"]["lpips"] for row in per_view])),
            "rgb_loss": float(mse),
        },
        "per_view": per_view,
        "num_train_views": count,
        "protocol": "Same-pose PUN NeRF renders vs training RGBA on alpha>0; pooled RGB MSE/PSNR, mean per-view masked SSIM/LPIPS",
    })


def train_eval(root: Path, lighting: str, policy: str, seed: int,
               iterations: int = 10000, eval_limit: int = 40,
               runs_dir: str = "runs_10000", train_diagnostics: bool = False) -> None:
    import torch
    from config import ExpConfig, ModelType, SceneType
    from fep_nbv.utils.utils import set_params
    from nvf.active_mapping.active_mapping import ActiveMapper

    source = root / lighting
    if not (source / "capture_complete.json").is_file():
        raise FileNotFoundError("Finish capture before training")
    num_training_views = int(read_json(source / "capture_complete.json").get("num_training_views", 20))
    if num_training_views < 1 or (train_diagnostics and policy != RANDOM100):
        raise ValueError("Invalid training-view count or diagnostic policy")
    run = root / runs_dir / lighting / policy
    if (run / "complete.json").exists():
        raise FileExistsError(run / "complete.json")
    run.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    set_seed(seed)
    images = read_training(source / "training" / policy / "rgba", num_training_views)
    poses = torch.from_numpy(np.load(source / f"{policy}_poses.npy").astype(np.float32))
    if poses.shape != (num_training_views, 7):
        raise ValueError(f"Expected {num_training_views} PUN quaternion poses")
    eval_poses = np.load(source / "eval_poses.npy").astype(np.float32)
    if eval_poses.shape != (40, 4, 4):
        raise ValueError("Expected 40 PUN evaluation poses")
    train_xyz = poses[:, 4:].numpy()
    if np.min(np.linalg.norm(train_xyz[:, None, :] - eval_poses[None, :, :3, 3], axis=-1)) < 1e-5:
        raise ValueError("Evaluation pose overlaps a training pose")
    cfg = ExpConfig()
    cfg.env.scene = SceneType.shapenet
    cfg.model = ModelType.ngp
    cfg.env.fov = 30
    cfg.env.resolution = (512, 512)
    cfg.env.radius = cfg.radius = RADIUS
    cfg.object_aabb = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    cfg.target_aabb = cfg.object_aabb.clone()
    cfg.camera_aabb = cfg.object_aabb * 2.8
    cfg.train_iter = iterations
    cfg.train_use_tensorboard = False
    mapper = ActiveMapper()
    mapper.fov = cfg.env.fov
    mapper.train_img_size = cfg.env.resolution
    config_path = mapper.initialize_config(str(run), str(run / "dataset"), model=cfg.model)
    # initialize_config uses PUN's fixed 10000-step instant-ngp schedule.
    # Set both limits before reset rebuilds the optimizer and its scheduler.
    mapper.config.max_num_iterations = cfg.train_iter
    mapper.config.optimizers["fields"]["scheduler"].max_steps = cfg.train_iter
    mapper.reset()
    set_params(cfg, mapper)
    mapper.trainer.pipeline.model.renderer_entropy.set_iteration(0)
    # Preserve the effective configuration as well as PUN's initial YAML.
    import yaml
    (run / "effective_config.yml").write_text(yaml.dump(mapper.config))
    write_json(run / "run_manifest.json", {"lighting": lighting, "policy": policy,
        "seed": seed, "training_iterations": iterations, "eval_limit": eval_limit,
        "field_scheduler_max_steps": mapper.config.optimizers["fields"]["scheduler"].max_steps,
        "runs_dir": runs_dir, "num_training_views": num_training_views,
        "train_diagnostics": train_diagnostics,
        "native_config": config_path,
        "effective_config": str(run / "effective_config.yml"),
        "train_pose_source": str(source / f"{policy}_poses.npy")})
    mapper.add_image(images=images, poses=list(poses), model_option=None)
    checkpoint = run / f"final_{iterations}.ckpt"
    mapper.save_ckpt(str(checkpoint), step=iterations)
    camera = read_json(source / "eval_poses.json")["camera"]
    render_predictions(mapper, eval_poses[:eval_limit], camera, run / "predictions")
    evaluate_saved(mapper.trainer.pipeline.model, source, run / "predictions", run, eval_limit)
    if train_diagnostics:
        from nvf.active_mapping.mapping_utils import to_transform
        train_matrices = np.stack([to_transform(pose).cpu().numpy() for pose in poses])
        render_predictions(mapper, train_matrices, camera, run / "train_predictions", "training")
        evaluate_training_saved(mapper.trainer.pipeline.model, source,
                                run / "train_predictions", run, num_training_views)
    write_json(run / "complete.json", {"lighting": lighting, "policy": policy,
        "seed": seed, "training_iterations": iterations,
        "num_training_views": num_training_views, "config": config_path,
        "checkpoint": str(checkpoint), "eval_source": str(source / "eval_poses.json"),
        "metrics": str(run / "metrics.json"),
        "train_metrics": str(run / "train_metrics.json") if train_diagnostics else None})
    del mapper
    gc.collect()


def summarize(root: Path, runs_dir: str = "runs_10000", iterations: int = 10000,
              policies: tuple[str, ...] = POLICIES) -> None:
    rows = []
    train_rows = []
    complete = True
    for lighting in ("original", "bright"):
        for policy in policies:
            run = root / runs_dir / lighting / policy
            path = run / "metrics.json"
            completion = run / "complete.json"
            if not path.exists() or not completion.exists():
                complete = False
                continue
            metrics = read_json(path)
            finished = read_json(completion)
            if (finished.get("training_iterations") != iterations
                    or metrics.get("num_eval_views") != 40
                    or len(metrics.get("per_view", [])) != 40):
                complete = False
                continue
            if any(not all(metric in metrics.get(region, {})
                           for metric in ("psnr", "ssim", "lpips", "rgb_loss"))
                   for region in ("full", "foreground")):
                complete = False
                continue
            for region in ("full", "foreground"):
                rows.append(dict(lighting=lighting, policy=policy, region=region,
                                 **metrics[region]))
            if policy == RANDOM100:
                train_path = run / "train_metrics.json"
                if not train_path.is_file():
                    complete = False
                    continue
                train = read_json(train_path)
                if (train.get("num_train_views") != 100
                        or len(train.get("per_view", [])) != 100):
                    complete = False
                    continue
                train_rows.append(dict(lighting=lighting, policy=policy,
                                       region="train_foreground", **train["foreground"]))
    tag = Path(runs_dir).name.removeprefix("runs_")
    stem = "summary" if runs_dir == "runs" else f"summary_{tag}"
    write_json(root / f"{stem}.json", {"rows": rows,
                                         "train_rows": train_rows,
                                         "complete": complete and len(rows) == 4 * len(policies)
                                         and (RANDOM100 not in policies or len(train_rows) == 2),
                                         "training_iterations": iterations, "runs_dir": runs_dir})
    import csv
    with (root / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("lighting", "policy", "region", "psnr", "ssim", "lpips", "rgb_loss"))
        writer.writeheader()
        writer.writerows(rows)
    if RANDOM100 in policies:
        with (root / f"train_{stem}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("lighting", "policy", "region", "psnr", "ssim", "lpips", "rgb_loss"))
            writer.writeheader()
            writer.writerows(train_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-eval", "summarize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lighting", choices=("original", "bright"))
    parser.add_argument("--policy", choices=(*POLICIES, "random3", RANDOM100))
    parser.add_argument("--policies", nargs="+", choices=(*POLICIES, "random3", RANDOM100),
                        default=POLICIES, help="Policies required by summarize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10000, help="Debug override; full run uses 10000")
    parser.add_argument("--eval-limit", type=int, default=40, help="Debug override; full run uses 40")
    parser.add_argument("--train-diagnostics", action="store_true",
                        help="Render random100 training poses and report masked fit metrics")
    parser.add_argument("--runs-dir", default="runs_10000",
                        help="Isolated run directory under --root; defaults to runs_10000")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.root.resolve(), args.runs_dir, args.iterations, tuple(args.policies))
    else:
        if not args.lighting or not args.policy:
            parser.error("train-eval requires --lighting and --policy")
        if args.iterations < 1 or not 1 <= args.eval_limit <= 40:
            parser.error("--iterations must be positive and --eval-limit must be in [1,40]")
        train_eval(args.root.resolve(), args.lighting, args.policy, args.seed,
                   args.iterations, args.eval_limit, args.runs_dir, args.train_diagnostics)


if __name__ == "__main__":
    main()
