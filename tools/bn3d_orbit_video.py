"""Render one BN3D checkpoint as an RGB orbit video using bundled Nerfstudio.

Example (run in a GPU pod, no training is performed):
  /mnt/home/conda-envs/pun/bin/python tools/bn3d_orbit_video.py \
      --profile pun_rendered --subset avs20 --steps 30000

All profiles share the same default trajectory and PUN-rendered intrinsics.
The checkpoint's adjacent config.yml is required to reconstruct the model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
sys.path.insert(0, str(REPO))


def select_checkpoint(args):
    """Select a completed run of exactly the requested training length."""
    root = args.runs_root / args.profile / args.subset / "instant_ngp"
    configs = [args.config.resolve()] if args.config else list(root.rglob("config.yml"))
    matches = []
    for config in configs:
        # BaseLoader inspects YAML without instantiating Nerfstudio Python objects.
        metadata = yaml.load(config.read_text(), Loader=yaml.BaseLoader)
        if int(metadata["max_num_iterations"]) != args.steps:
            continue
        relative = metadata.get("relative_model_dir", "nerfstudio_models")
        # Saved Nerfstudio YAML represents pathlib paths as a tagged list.
        relative = Path(*relative) if isinstance(relative, list) else Path(relative)
        model_dir = config.parent / relative
        checkpoint = model_dir / f"step-{args.steps - 1:09d}.ckpt"
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            matches.append((config, checkpoint))
    if not matches:
        raise FileNotFoundError(
            f"No completed {args.steps}-step run for {args.profile}/{args.subset}. "
            f"Expected config.yml and step-{args.steps - 1:09d}.ckpt under {root}. "
            "Wait for this run to finish, or specify --steps 10000 for an older run."
        )
    return max(matches, key=lambda pair: pair[0].stat().st_mtime)


def make_trajectory(args):
    """Construct Z-up, OpenGL camera-to-world matrices with -Z looking inward."""
    scene = json.loads((args.building_dir / "scene_config.json").read_text())
    bounds = np.asarray(scene["object_aabb"], dtype=np.float64)
    center = bounds.mean(axis=0)
    candidates = json.loads((args.building_dir / "candidate_poses.json").read_text())["frames"]
    offsets = np.asarray([f["transform_matrix"] for f in candidates])[:, :3, 3] - center
    elevation = args.elevation_deg
    if elevation is None:
        elevation = float(np.rad2deg(np.median(np.arctan2(offsets[:, 2], np.linalg.norm(offsets[:, :2], axis=1)))))
    if not -85 < elevation < 85:
        raise ValueError("Elevation must be between -85 and 85 degrees.")
    radius = args.radius if args.radius is not None else float(np.median(np.linalg.norm(offsets, axis=1)))
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    # A shared display camera for fair cross-profile comparison; training K is untouched.
    camera = json.loads((args.building_dir / "pun_rendered" / "avs20" / "camera_config.json").read_text())
    K = np.asarray(camera["K"], dtype=float)
    fx, cx = K[0, 0] * args.width / camera["width"], K[0, 2] * args.width / camera["width"]
    fy, cy = K[1, 1] * args.height / camera["height"], K[1, 2] * args.height / camera["height"]
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))
    angles = np.deg2rad(args.start_azimuth_deg) + np.arange(args.frames) * 2 * np.pi / args.frames
    elev = np.deg2rad(elevation)

    def build(distance):
        positions = center + distance * np.column_stack((
            np.cos(elev) * np.cos(angles), np.cos(elev) * np.sin(angles),
            np.full(args.frames, np.sin(elev))))
        backward = (positions - center) / distance
        right = np.cross(np.array([0., 0., 1.]), backward)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        up = np.cross(backward, right)
        poses = np.tile(np.eye(4), (args.frames, 1, 1))
        poses[:, :3, :3] = np.stack((right, up, backward), axis=2)
        poses[:, :3, 3] = positions
        return poses

    def fits(poses):
        q = np.einsum("nki,nij->nkj", corners[None] - poses[:, None, :3, 3], poses[:, :3, :3])
        depth = -q[..., 2]
        if np.any(depth <= 0):
            return False
        x, y = fx * q[..., 0] / depth + cx, -fy * q[..., 1] / depth + cy
        return (x.min() >= .05 * args.width and x.max() <= .95 * args.width
                and y.min() >= .05 * args.height and y.max() <= .95 * args.height)

    poses = build(radius)
    if args.radius is None:
        # Fit all building AABB corners with a 5% border across the entire orbit.
        for _ in range(200):
            if fits(poses):
                break
            radius *= 1.03
            poses = build(radius)
        else:
            raise ValueError("Could not frame the building; check scene bounds.")
    elif not fits(poses):
        raise ValueError("Specified radius clips the building bounds; increase --radius or omit it for automatic framing.")
    return {
        "camera_convention": "OpenGL camera-to-world, -Z forward, world Z-up",
        "w": args.width, "h": args.height, "fl_x": float(fx), "fl_y": float(fy),
        "cx": float(cx), "cy": float(cy), "fps": args.fps,
        "center": center.tolist(), "radius": radius, "elevation_deg": elevation,
        "frames": [{"transform_matrix": pose.tolist()} for pose in poses],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, choices=["pun_rendered", "bn3d_faithful"])
    parser.add_argument("--subset", required=True, choices=["avs20", "random20"])
    parser.add_argument("--steps", type=int, default=30000, help="Completed training iterations, not video frames.")
    parser.add_argument("--config", type=Path, help="Pin a particular run's config.yml instead of finding the latest completed run.")
    parser.add_argument("--runs-root", type=Path, default=WORKSPACE / "outputs/bn3d_building1")
    parser.add_argument("--building-dir", type=Path, default=WORKSPACE / "data/bn3d_processed/building1")
    parser.add_argument("--output", type=Path, help="MP4 path; by default write under the selected run's videos directory.")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--radius", type=float, help="Distance to building center, in canonical scene units.")
    parser.add_argument("--elevation-deg", type=float, help="Default: median elevation of the 200 candidate cameras.")
    parser.add_argument("--start-azimuth-deg", type=float, default=-90)
    parser.add_argument("--rays-per-chunk", type=int, default=8192)
    parser.add_argument("--check-only", action="store_true", help="Check checkpoint and trajectory without loading a GPU model or writing video.")
    args = parser.parse_args()
    if min(args.steps, args.width, args.height, args.rays_per_chunk) <= 0 or args.frames < 2 or args.fps <= 0:
        parser.error("Steps, dimensions, ray chunk and FPS must be positive; frames must be >= 2.")
    if args.width % 2 or args.height % 2:
        parser.error("Use even dimensions for H.264 video.")
    config, expected_checkpoint = select_checkpoint(args)
    trajectory = make_trajectory(args)
    print(f"Config: {config}\nCheckpoint: {expected_checkpoint}", flush=True)
    print(f"Orbit: {args.frames} frames, {args.width}x{args.height}, {args.frames / args.fps:g} seconds; "
          f"radius={trajectory['radius']:.3f}, elevation={trajectory['elevation_deg']:.1f} degrees", flush=True)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg must be available on PATH in the rendering pod.")
    if args.check_only:
        print("Checkpoint selection, framing and ffmpeg checks passed. No model loaded.")
        return

    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.scripts.render import _render_trajectory_video
    from nerfstudio.utils.eval_utils import eval_setup

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script in a GPU pod with the PUN environment.")
    output = args.output or config.parent / "videos" / f"orbit_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
    if output.suffix.lower() != ".mp4":
        parser.error("--output must end with .mp4")
    trajectory_path, manifest_path = output.with_suffix(".trajectory.json"), output.with_suffix(".manifest.json")
    if any(path.exists() for path in (output, trajectory_path, manifest_path)):
        raise FileExistsError(f"Output already exists; choose a new --output path: {output}")

    _, pipeline, loaded_checkpoint, step = eval_setup(config, eval_num_rays_per_chunk=args.rays_per_chunk, test_mode="inference")
    if loaded_checkpoint.resolve() != expected_checkpoint.resolve() or step != args.steps - 1:
        raise RuntimeError("Loaded checkpoint does not match the selected completed run.")
    poses = np.asarray([frame["transform_matrix"] for frame in trajectory["frames"]], dtype=np.float32)
    cameras = Cameras(camera_to_worlds=torch.from_numpy(poses[:, :3, :4]),
                      fx=trajectory["fl_x"], fy=trajectory["fl_y"], cx=trajectory["cx"], cy=trajectory["cy"],
                      width=args.width, height=args.height, camera_type=CameraType.PERSPECTIVE)
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(json.dumps(trajectory, indent=2) + "\n")
    _render_trajectory_video(pipeline, cameras, output_filename=output,
                             rendered_output_names=["rgb"], seconds=args.frames / args.fps, output_format="video")
    manifest_path.write_text(json.dumps({"profile": args.profile, "subset": args.subset,
        "training_steps": args.steps, "checkpoint_step": step, "config": str(config),
        "checkpoint": str(loaded_checkpoint), "video": str(output), "trajectory": str(trajectory_path),
        "frames": args.frames, "fps": args.fps, "width": args.width, "height": args.height}, indent=2) + "\n")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
