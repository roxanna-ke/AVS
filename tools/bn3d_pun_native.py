"""BN3D building1 through PUN's online AVS and ShapeNet renderers.

Run with /mnt/home/conda-envs/pun/bin/python from the PUN repository root.
The original OBJ and the existing BN3D experiment are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "08-vit-train"), str(REPO / "tools")]

import numpy as np
from PIL import Image

from bn3d_common import MESH, read_json, sha256, write_json

DEFAULT_ROOT = WORKSPACE / "data/bn3d_pun_native/building1"
CANONICAL = WORKSPACE / "data/bn3d_processed/building1/canonical_transform.json"
CHECKPOINT = WORKSPACE / "models/UPNet/vit_small_patch16_224_PSNR_250425172703/best_vit_regressor.pth"
HDRI = REPO / "data/assets/hdri/gray_hdri.exr"
RADIUS = 2.7319998741149902


def prepare(root: Path) -> Path:
    """Make a derived OBJ in PUN coordinates, retaining UVs and MTL links."""
    canonical = read_json(CANONICAL)
    assert sha256(MESH) == canonical["source_sha256"][str(MESH)]
    assets = root / "asset"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / "House.obj"
    matrix = np.asarray(canonical["import_matrices"]["House"], dtype=np.float64)
    normalize = np.asarray(canonical["world_to_canonical"], dtype=np.float64)
    raw_to_raw = np.linalg.inv(matrix) @ normalize @ matrix
    if not target.exists():
        temporary = target.with_suffix(".obj.tmp")
        with MESH.open() as source, temporary.open("w") as output:
            for line in source:
                if line.startswith("v "):
                    fields = line.split()
                    xyz = raw_to_raw @ np.array([*map(float, fields[1:4]), 1.0])
                    fields[1:4] = [f"{value:.9f}" for value in xyz[:3]]
                    line = " ".join(fields) + "\n"
                output.write(line)
        temporary.replace(target)
    for name in ("House.mtl", "House_Diff_5k.png", "House_Nrm_5K.png", "House_Spec_5K.png"):
        dest = assets / name
        if not dest.exists():
            dest.symlink_to(MESH.parent / name)
    write_json(root / "asset_manifest.json", {
        "source_obj": str(MESH), "source_sha256": sha256(MESH),
        "derived_obj": str(target), "derived_sha256": sha256(target),
        "canonical_source": str(CANONICAL), "raw_to_raw": raw_to_raw.tolist(),
        "imported_axis_matrix": matrix.tolist(), "world_to_canonical": normalize.tolist(),
        "radius": RADIUS, "PUN_scale_on_derived_mesh": 1.0,
    })
    return target


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scene_state(stage: str) -> dict:
    import bpy
    scene = bpy.context.scene
    world = scene.world
    bg = world.node_tree.nodes.get("Background") if world and world.use_nodes else None
    camera = scene.camera
    return {
        "stage": stage, "blender_version": bpy.app.version_string,
        "engine": scene.render.engine, "samples": scene.cycles.samples,
        "world": world.name if world else None,
        "world_background_strength": float(bg.inputs["Strength"].default_value) if bg else None,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        "film_transparent": scene.render.film_transparent,
        "camera_focal_mm": camera.data.lens if camera else None,
        "camera_sensor_width_mm": camera.data.sensor_width if camera else None,
    }


def brighten_world(world) -> None:
    import bpy
    from mathutils import Vector
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    env = next((node for node in nodes if node.type == "TEX_ENVIRONMENT"), None)
    if env is None:
        env = nodes.new("ShaderNodeTexEnvironment")
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    env.image = bpy.data.images.load(str(HDRI), check_existing=True)
    scene = bpy.context.scene
    scene.world = world
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    # Stable key/fill illumination in addition to PUN's gray HDRI. Removing
    # only our own lights prevents doubling them when ShapeNetScene is created.
    for obj in list(scene.objects):
        if obj.type == "LIGHT" and obj.name.startswith("BN3D_PUN_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for name, location, energy in (
        ("BN3D_PUN_key", (4.0, -6.0, 8.0), 2.0),
        ("BN3D_PUN_fill", (-5.0, 3.0, 6.0), 1.0),
    ):
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = energy
        data.angle = 0.3
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (-Vector(location)).to_track_quat("-Z", "Y").to_euler()


def save_rgba(image: np.ndarray, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    assert image.dtype == np.uint8 and image.shape == (512, 512, 4)
    if not np.any(image[..., 3]):
        raise ValueError(f"Empty BN3D foreground from second PUN renderer: {target}")
    Image.fromarray(image).save(target)


def register_bn3d_instances() -> int:
    """PUN assumes a one-object ShapeNet OBJ; BN3D imports as many meshes."""
    import bpy
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("ShapeNetScene imported no BN3D meshes")
    for obj in meshes:
        obj["inst_id"] = 1000
    bpy.context.view_layer.update()
    return len(meshes)


def save_poses(root: Path, name: str, poses, camera: dict) -> None:
    from scipy.spatial.transform import Rotation
    from bn3d_common import write_json
    data = poses.detach().cpu().numpy() if hasattr(poses, "detach") else np.asarray(poses)
    np.save(root / f"{name}_poses.npy", data)
    frames = []
    for i, pose in enumerate(data):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat(pose[:4]).as_matrix()
        c2w[:3, 3] = pose[4:]
        frames.append({"index": i, "transform_matrix": c2w.tolist(), "pose_xyzw_xyz": pose.tolist()})
    write_json(root / f"{name}_poses.json", {"camera": camera, "frames": frames})


def capture(root: Path, lighting: str, seed: int, samples: int | None) -> None:
    """Run native PUN online selection, then its second Blender renderer."""
    import bpy
    import torch
    import timm
    from regress_model import ViTRegressor
    from blender_utils import blender_interface
    from config import ExpConfig, SamplerType, SceneType
    from fep_nbv.env.shapenet_env import set_env
    from fep_nbv.env.shapenet_scene import ShapeNetScene
    from fep_nbv.env.gen_data_fn import shapenet_eval
    from nvf.active_mapping.agents.Sampler import SphericalSampler
    from fep_nbv.baseline import our_policy_single as policy

    if not torch.cuda.is_available():
        raise RuntimeError("PUN UPNet requires CUDA")
    if lighting not in ("original", "bright"):
        raise ValueError(lighting)
    model_path = prepare(root)
    out = root / lighting
    if (out / "capture_complete.json").exists():
        raise FileExistsError(out / "capture_complete.json")
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    set_seed(seed)
    cfg = ExpConfig()
    cfg.env.scene = SceneType.shapenet
    cfg.sampler = SamplerType.spherical
    cfg.env.target_path = str(model_path)
    cfg.env.scale = 1.0
    cfg.radius = cfg.env.radius = RADIUS
    cfg.env.save_data = False  # Avoid PUN's shared data/shapenet/init path.
    if samples is not None:
        cfg.env.cycles_samples = samples
    # Original entrypoint initializes ShapeNetEnviroment before BlenderInterface.
    initial_env = set_env(cfg)
    del initial_env
    sampler = SphericalSampler(cfg)
    renderer = blender_interface.BlenderInterface(resolution=512)
    renderer.import_mesh(str(model_path), scale=1.0)
    if lighting == "bright":
        brighten_world(bpy.context.scene.world)
    states = [scene_state("avs")]
    # The full PUN checkpoint replaces every backbone parameter; avoid a
    # redundant network fetch from timm during offline reproduction.
    model = ViTRegressor(model_name="vit_small_patch16_224", output_dim=48,
                         pretrained=False).cuda()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cuda"), strict=True)
    model.eval()
    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    policy.radius = RADIUS
    first = sampler.sample(512)
    poses = [first[0]]
    relative = []
    steps = []
    avs_dir = out / "avs"
    avs_dir.mkdir(exist_ok=True)
    for index in range(20):
        obs = renderer.render(str(avs_dir / "latest"), poses[-1][None], write_cam_params=True)
        rgb = np.rint(obs[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(avs_dir / f"{index:04d}.png")
        with torch.inference_mode():
            prediction = model(transform(obs.permute(0, 3, 1, 2)).cuda())[0].cpu().numpy()
        if prediction.shape != (48,) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid UPNet prediction at step {index}")
        relative.append(prediction)
        if index == 19:
            break
        candidates = sampler(512)
        history = policy.interpolate_uncertainty(relative, torch.stack(poses), candidates)
        eligible = policy.filter_viewpoints_small(history, torch.stack(poses), candidates, "PSNR")
        if len(eligible) == 0:
            raise RuntimeError(f"Original PUN filter removed all candidates at step {index}")
        filtered_history = policy.interpolate_uncertainty(relative, torch.stack(poses), eligible)
        score = np.prod(filtered_history, axis=0)
        selected_index = int(np.argmin(score))
        selected = policy.view_select_all(filtered_history, eligible, "PSNR")
        assert torch.equal(selected, eligible[selected_index])
        poses.append(selected)
        np.savez_compressed(avs_dir / f"step_{index + 1:02d}.npz",
                            candidates=candidates.numpy(), eligible=eligible.numpy(),
                            scores=score, prediction=prediction)
        steps.append({"step": index + 1, "num_eligible": len(eligible),
                      "selected_index": selected_index, "selected_pose": selected.tolist()})
        print(f"{lighting}: AVS {index + 2}/20", flush=True)
    avs_poses = torch.stack(poses)
    np.save(avs_dir / "upnet_48.npy", np.stack(relative))
    write_json(avs_dir / "selection.json", {"steps": steps, "mode": "PSNR", "select": "all", "filter": "small"})
    # Independent native sampler stream; both lightings share random20 directions.
    np.random.seed(seed + 1)
    random_poses = torch.cat([avs_poses[:1], sampler.sample(19)], dim=0)
    assert len(random_poses) == len(avs_poses) == 20

    # Match original call order: a fresh ShapeNetScene after online selection.
    scene = ShapeNetScene(cfg.env)
    mesh_count = register_bn3d_instances()
    if lighting == "bright":
        brighten_world(next(world for world in bpy.data.worlds if world.node_tree == scene.rgb_node_tree))
    states.append(scene_state("training_and_gt"))
    camera = scene.get_camera_params()
    for policy_name, selected_poses in (("avs20", avs_poses), ("random20", random_poses)):
        save_poses(out, policy_name, selected_poses, camera)
    eval_poses = shapenet_eval()
    if len(eval_poses) != 40:
        raise ValueError(f"Expected original PUN's 40 evaluation poses, got {len(eval_poses)}")
    np.save(out / "eval_poses.npy", eval_poses)
    (out / "eval/rgb_black").mkdir(parents=True, exist_ok=True)
    (out / "eval/mask").mkdir(parents=True, exist_ok=True)
    write_json(out / "eval_poses.json", {"camera": camera, "frames": [
        {"index": i, "transform_matrix": pose.tolist()} for i, pose in enumerate(eval_poses)]})
    write_json(out / "selection_complete.json", {
        "lighting": lighting, "seed": seed, "random_seed": seed + 1,
        "states": states, "asset_manifest": str(root / "asset_manifest.json"),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "pun_sources": {str(path.relative_to(REPO)): sha256(path) for path in (
            REPO / "fep_nbv/baseline/our_policy_single.py",
            REPO / "nvf/active_mapping/agents/Sampler.py",
            REPO / "08-vit-train/blender_utils/blender_interface.py",
            REPO / "fep_nbv/env/shapenet_scene.py",
            REPO / "fep_nbv/env/gen_data_fn.py")},
        "hdri": str(HDRI), "hdri_sha256": sha256(HDRI),
        "native_second_renderer_samples": cfg.env.cycles_samples,
        "registered_bn3d_meshes": mesh_count,
        "bright_lighting": {"world_strength": 1.0, "key_sun_energy": 2.0,
                            "fill_sun_energy": 1.0, "view_transform": "Standard"} if lighting == "bright" else None,
        "backgrounds": {"avs_upnet": "white RGB", "training": "RGBA/random training blend", "eval": "black RGB"},
    })


def render_batch(root: Path, lighting: str, split: str, start: int, count: int) -> None:
    """Render at most one short batch per Blender process to avoid bpycv crashes."""
    import bpy
    import torch
    from config import EnvConfig, SceneType
    from blender_utils import blender_interface
    from fep_nbv.env.shapenet_scene import ShapeNetScene

    out = root / lighting
    record = read_json(out / "selection_complete.json")
    total = 40 if split == "eval" else 20
    if start < 0 or count < 1 or start + count > total:
        raise ValueError("Invalid render batch")
    model_path = prepare(root)
    os.chdir(REPO)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    # PUN BlenderInterface creates the world inherited by ShapeNetScene.
    first = blender_interface.BlenderInterface(resolution=512)
    if lighting == "bright":
        brighten_world(bpy.context.scene.world)
    cfg = EnvConfig(scene=SceneType.shapenet, target_path=str(model_path),
                    scale=1.0, radius=RADIUS,
                    cycles_samples=int(record["native_second_renderer_samples"]),
                    save_data=False)
    scene = ShapeNetScene(cfg)
    mesh_count = register_bn3d_instances()
    if lighting == "bright":
        brighten_world(next(world for world in bpy.data.worlds if world.node_tree == scene.rgb_node_tree))
    state = scene_state("training_and_gt")
    expected = record["states"][-1]
    for key in ("world_background_strength", "view_transform", "look",
                "exposure", "gamma", "camera_focal_mm"):
        if state[key] != expected[key]:
            raise ValueError(f"Second renderer state mismatch for {key}: {state[key]} != {expected[key]}")
    poses = np.load(out / ("eval_poses.npy" if split == "eval" else f"{split}_poses.npy"))
    for i in range(start, start + count):
        if split == "eval":
            target = out / "eval/rgba" / f"{i:04d}.png"
        else:
            target = out / "training" / split / "rgba" / f"{i:04d}.png"
        if target.exists():
            with Image.open(target) as cached:
                rgba = np.array(cached.convert("RGBA"))
            if np.any(rgba[..., 3]):
                if split != "eval" or all((out / "eval" / name / f"{i:04d}.png").exists()
                                          for name in ("rgb_black", "mask")):
                    continue
        pose = torch.from_numpy(poses[i]) if split != "eval" else poses[i]
        rgba = scene.render_pose(pose)
        save_rgba(rgba, target)
        if split == "eval":
            Image.fromarray(rgba[..., :3]).save(out / "eval/rgb_black" / f"{i:04d}.png")
            Image.fromarray(rgba[..., 3]).save(out / "eval/mask" / f"{i:04d}.png")
        print(f"{lighting} {split} rendered {i + 1}/{total}", flush=True)
    write_json(out / "render_batches" / f"{split}_{start:04d}.json",
               {"lighting": lighting, "split": split, "start": start,
                "count": count, "state": state, "registered_bn3d_meshes": mesh_count})


def finalize_capture(root: Path, lighting: str) -> None:
    out = root / lighting
    selection = read_json(out / "selection_complete.json")
    for split, total in (("avs20", 20), ("random20", 20), ("eval", 40)):
        for i in range(total):
            file = out / ("eval/rgba" if split == "eval" else f"training/{split}/rgba") / f"{i:04d}.png"
            with Image.open(file) as image:
                rgba = np.array(image.convert("RGBA"))
            if rgba.shape != (512, 512, 4) or not np.any(rgba[..., 3]):
                raise ValueError(f"Invalid foreground in {file}")
            if split == "eval":
                for folder in ("rgb_black", "mask"):
                    if not (out / "eval" / folder / f"{i:04d}.png").is_file():
                        raise FileNotFoundError(f"Missing eval {folder} for view {i}")
    write_json(out / "capture_complete.json", selection)


def preview(root: Path, lighting: str, seed: int) -> None:
    """One low-sample pose through both original PUN renderer classes."""
    import bpy
    from config import EnvConfig, SceneType
    from blender_utils import blender_interface
    from fep_nbv.env.shapenet_scene import ShapeNetScene
    from nvf.active_mapping.agents.Sampler import sample_poses_spherical

    model_path = prepare(root)
    os.chdir(REPO)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    set_seed(seed)
    pose = sample_poses_spherical(1, radius=RADIUS)
    out = root / "previews" / lighting
    out.mkdir(parents=True, exist_ok=True)
    first = blender_interface.BlenderInterface(resolution=512)
    first.import_mesh(str(model_path), scale=1.0)
    bpy.context.scene.cycles.samples = 1
    if lighting == "bright":
        brighten_world(bpy.context.scene.world)
    first_state = scene_state("avs")
    first_rgb = first.render(str(out / "first_render"), pose, write_cam_params=True)[0]
    Image.fromarray(np.rint(first_rgb.numpy() * 255).clip(0, 255).astype(np.uint8)).save(out / "avs_rgb.png")
    cfg = EnvConfig(scene=SceneType.shapenet, target_path=str(model_path),
                    scale=1.0, radius=RADIUS, cycles_samples=1, save_data=False)
    second = ShapeNetScene(cfg)
    mesh_count = register_bn3d_instances()
    if lighting == "bright":
        brighten_world(next(world for world in bpy.data.worlds if world.node_tree == second.rgb_node_tree))
    second_state = scene_state("training")
    save_rgba(second.render_pose(pose[0]), out / "training_rgba.png")
    write_json(out / "preview.json", {"pose": pose[0].tolist(),
                                    "first": first_state, "second": second_state,
                                    "registered_bn3d_meshes": mesh_count})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preview", "capture", "render", "finalize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lighting", choices=("original", "bright"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, help="Debug override for PUN ShapeNetScene Cycles samples; omit for native 10000")
    parser.add_argument("--split", choices=("avs20", "random20", "eval"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root.resolve()))
    elif args.command == "preview":
        if args.lighting is None:
            parser.error("preview requires --lighting")
        preview(args.root.resolve(), args.lighting, args.seed)
    elif args.command == "render":
        if args.lighting is None or args.split is None:
            parser.error("render requires --lighting and --split")
        render_batch(args.root.resolve(), args.lighting, args.split, args.start, args.count)
    elif args.command == "finalize":
        if args.lighting is None:
            parser.error("finalize requires --lighting")
        finalize_capture(args.root.resolve(), args.lighting)
    else:
        if args.lighting is None:
            parser.error("capture requires --lighting")
        capture(args.root.resolve(), args.lighting, args.seed, args.samples)


if __name__ == "__main__":
    main()
