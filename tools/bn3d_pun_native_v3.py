"""BN3D building1 v3 through PUN's online AVS and ShapeNet renderers.

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

from bn3d_common import MESH, contact_sheet, read_json, sha256, write_json

DEFAULT_ROOT = WORKSPACE / "outputs/bn3d_pun_native_v3/building1"
CANONICAL = WORKSPACE / "data/bn3d_processed/building1/canonical_transform.json"
CHECKPOINT = WORKSPACE / "models/UPNet/vit_small_patch16_224_PSNR_250425172703/best_vit_regressor.pth"
HDRI = REPO / "data/assets/hdri/gray_hdri.exr"
RADIUS = 2.7319998741149902
PUN_SCALE = 2.0
TARGET_MAX_DIST = 0.5
POLICIES = ("avs20", "random1", "random2")


def obj_vertices(path: Path) -> np.ndarray:
    with path.open() as source:
        vertices = [list(map(float, line.split()[1:4])) for line in source if line.startswith("v ")]
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def imported_vertices(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return raw @ matrix[:3, :3].T + matrix[:3, 3]


def bounds_center(vertices: np.ndarray) -> np.ndarray:
    return (vertices.min(axis=0) + vertices.max(axis=0)) / 2


def prepare(root: Path) -> Path:
    """Make a v3 unit-sphere OBJ in PUN coordinates, retaining UVs and MTL."""
    canonical = read_json(CANONICAL)
    assert sha256(MESH) == canonical["source_sha256"][str(MESH)]
    assets = root / "asset"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / "House.obj"
    manifest_path = root / "asset_manifest.json"
    matrix = np.asarray(canonical["import_matrices"]["House"], dtype=np.float64)
    raw = obj_vertices(MESH)
    world = imported_vertices(raw, matrix)
    center = bounds_center(world)
    max_dist = float(np.linalg.norm(world - center, axis=1).max())
    normalize = np.eye(4)
    normalize[:3, :3] *= TARGET_MAX_DIST / max_dist
    normalize[:3, 3] = -normalize[0, 0] * center
    raw_to_raw = np.linalg.inv(matrix) @ normalize @ matrix
    if target.exists():
        if not manifest_path.exists():
            raise ValueError(f"Existing v3 asset has no manifest: {target}")
        old = read_json(manifest_path)
        if (old.get("normalization") != "unit_sphere"
                or old.get("source_sha256") != sha256(MESH)
                or old.get("target_max_dist") != TARGET_MAX_DIST
                or old.get("PUN_scale_on_derived_mesh") != PUN_SCALE
                or "raw_to_raw" not in old
                or not np.allclose(old["raw_to_raw"], raw_to_raw, atol=1e-10)
                or old.get("derived_sha256") != sha256(target)):
            raise ValueError(f"Existing v3 asset does not match settings: {target}")
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
    derived = imported_vertices(obj_vertices(target), matrix)
    derived_center = bounds_center(derived)
    derived_max_dist = float(np.linalg.norm(derived - derived_center, axis=1).max())
    if not (np.allclose(derived_center, 0, atol=1e-6)
            and np.isclose(derived_max_dist, TARGET_MAX_DIST, atol=1e-6)):
        raise ValueError(f"Bad unit-sphere normalization: {derived_center}, {derived_max_dist}")
    for name in ("House.mtl", "House_Diff_5k.png", "House_Nrm_5K.png", "House_Spec_5K.png"):
        dest = assets / name
        if not dest.exists():
            dest.symlink_to(MESH.parent / name)
    write_json(root / "asset_manifest.json", {
        "source_obj": str(MESH), "source_sha256": sha256(MESH),
        "derived_obj": str(target), "derived_sha256": sha256(target),
        "canonical_source": str(CANONICAL), "raw_to_raw": raw_to_raw.tolist(),
        "imported_axis_matrix": matrix.tolist(), "world_to_unit_sphere": normalize.tolist(),
        "normalization": "unit_sphere", "target_max_dist": TARGET_MAX_DIST,
        "derived_max_dist": derived_max_dist, "bounds_center_imported": center.tolist(),
        "pun_scale": PUN_SCALE, "radius": RADIUS,
        "PUN_scale_on_derived_mesh": PUN_SCALE,
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
    env = next((node for node in world.node_tree.nodes if node.type == "TEX_ENVIRONMENT"), None) if world and world.use_nodes else None
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
        "hdri_path": str(Path(env.image.filepath).resolve()) if env and env.image else None,
        "lights": [{"name": obj.name, "type": obj.data.type,
                    "energy": float(obj.data.energy), "location": list(obj.location)}
                   for obj in scene.objects if obj.type == "LIGHT"],
    }


def brighten_world_hdri(world) -> None:
    import bpy
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 4.0
    env = next((node for node in nodes if node.type == "TEX_ENVIRONMENT"), None)
    if env is None:
        env = nodes.new("ShaderNodeTexEnvironment")
        world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    env.image = bpy.data.images.load(str(HDRI), check_existing=True)
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    assert not any(obj.type == "LIGHT" for obj in scene.objects)


def apply_bright_lighting(world, mode: str) -> None:
    if mode == "v2":
        from bn3d_pun_native import brighten_world
        brighten_world(world)
    else:
        brighten_world_hdri(world)


def v3_scene_class():
    """Adapt PUN's single-object scale to every imported BN3D mesh."""
    import bpy
    from fep_nbv.env.shapenet_scene import ShapeNetScene

    class BN3DShapeNetScene(ShapeNetScene):
        def add_object(self, key, obj_file_path, obj_matrix=np.eye(4), scale=1.):
            super().add_object(key, obj_file_path, obj_matrix=obj_matrix, scale=scale)
            meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            if not meshes:
                raise RuntimeError("No BN3D meshes imported")
            for obj in meshes:
                obj.scale = (scale, scale, scale)
            bpy.context.view_layer.update()
            return len(meshes)

    return BN3DShapeNetScene


def validate_mesh_scale(target: float = PUN_SCALE * TARGET_MAX_DIST) -> float:
    import bpy
    verts = np.asarray([tuple(obj.matrix_world @ vertex.co)
                        for obj in bpy.context.scene.objects if obj.type == "MESH"
                        for vertex in obj.data.vertices], dtype=np.float64)
    if not len(verts):
        raise ValueError("No Blender mesh vertices")
    center = bounds_center(verts)
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    if not (np.allclose(center, 0, atol=2e-4) and np.isclose(radius, target, atol=2e-4)):
        raise ValueError(f"BN3D mesh scale mismatch: center={center}, radius={radius}, expected={target}")
    return radius


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


def capture(root: Path, lighting: str, seed: int, samples: int | None,
            bright_lighting: str = "v3", random_groups: int = 2) -> None:
    """Run native PUN online selection, then its second Blender renderer."""
    import bpy
    import torch
    import timm
    from regress_model import ViTRegressor
    from blender_utils import blender_interface
    from config import ExpConfig, SamplerType, SceneType
    from fep_nbv.env.shapenet_env import set_env
    from fep_nbv.env import shapenet_env as env_module
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
    cfg.env.scale = PUN_SCALE
    cfg.radius = cfg.env.radius = RADIUS
    cfg.env.save_data = False  # Avoid PUN's shared data/shapenet/init path.
    if samples is not None:
        cfg.env.cycles_samples = samples
    # Original entrypoint initializes ShapeNetEnviroment before BlenderInterface.
    v3_scene = v3_scene_class()
    original_scene = env_module.ShapeNetScene
    env_module.ShapeNetScene = v3_scene
    try:
        initial_env = set_env(cfg)
    finally:
        env_module.ShapeNetScene = original_scene
    del initial_env
    sampler = SphericalSampler(cfg)
    renderer = blender_interface.BlenderInterface(resolution=512)
    renderer.import_mesh(str(model_path), scale=PUN_SCALE)
    avs_mesh_radius = validate_mesh_scale()
    if lighting == "bright":
        apply_bright_lighting(bpy.context.scene.world, bright_lighting)
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
    # Independent streams; both lightings share the same random directions.
    np.random.seed(seed + 1)
    random1 = torch.cat([avs_poses[:1], sampler.sample(19)], dim=0)
    np.random.seed(seed + 2)
    random2 = torch.cat([avs_poses[:1], sampler.sample(19)], dim=0)
    assert len(random1) == len(random2) == len(avs_poses) == 20
    assert not torch.equal(random1[1:], random2[1:])
    selected_policies = [("avs20", avs_poses), ("random1", random1), ("random2", random2)]
    if random_groups == 3:
        # Keep the existing v3 random stream for evaluation poses unchanged.
        state = np.random.get_state()
        np.random.seed(seed + 3)
        random3 = torch.cat([avs_poses[:1], sampler.sample(19)], dim=0)
        np.random.set_state(state)
        assert len(random3) == 20 and not torch.equal(random3[1:], random2[1:])
        selected_policies.append(("random3", random3))

    # Match original call order: a fresh ShapeNetScene after online selection.
    scene = v3_scene(cfg.env)
    mesh_count = register_bn3d_instances()
    training_mesh_radius = validate_mesh_scale()
    if lighting == "bright":
        apply_bright_lighting(next(world for world in bpy.data.worlds if world.node_tree == scene.rgb_node_tree), bright_lighting)
    states.append(scene_state("training_and_gt"))
    camera = scene.get_camera_params()
    for policy_name, selected_poses in selected_policies:
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
        "lighting": lighting, "seed": seed,
        "random_seeds": [seed + i for i in range(1, random_groups + 1)],
        "policies": [name for name, _ in selected_policies],
        "bright_lighting_mode": bright_lighting,
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
        "avs_mesh_radius": avs_mesh_radius,
        "training_mesh_radius": training_mesh_radius,
        "bright_lighting": ({"world_strength": 1.0, "key_sun_energy": 2.0,
                             "fill_sun_energy": 1.0, "view_transform": "Standard"}
                            if bright_lighting == "v2" else
                            {"world_strength": 4.0, "lights": 0,
                             "view_transform": "Standard"}) if lighting == "bright" else None,
        "backgrounds": {"avs_upnet": "white RGB", "training": "RGBA/random training blend", "eval": "black RGB"},
    })


def render_batch(root: Path, lighting: str, split: str, start: int, count: int) -> None:
    """Render at most one short batch per Blender process to avoid bpycv crashes."""
    import bpy
    import torch
    from config import EnvConfig, SceneType
    from blender_utils import blender_interface

    out = root / lighting
    record = read_json(out / "selection_complete.json")
    if split != "eval" and split not in record.get("policies", POLICIES):
        raise ValueError(f"Policy was not captured: {split}")
    bright_lighting = record.get("bright_lighting_mode", "v3")
    total = 40 if split == "eval" else 20
    if start < 0 or count < 1 or start + count > total:
        raise ValueError("Invalid render batch")
    model_path = prepare(root)
    os.chdir(REPO)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    # PUN BlenderInterface creates the world inherited by ShapeNetScene.
    first = blender_interface.BlenderInterface(resolution=512)
    if lighting == "bright":
        apply_bright_lighting(bpy.context.scene.world, bright_lighting)
    cfg = EnvConfig(scene=SceneType.shapenet, target_path=str(model_path),
                    scale=PUN_SCALE, radius=RADIUS,
                    cycles_samples=int(record["native_second_renderer_samples"]),
                    save_data=False)
    scene = v3_scene_class()(cfg)
    mesh_count = register_bn3d_instances()
    mesh_radius = validate_mesh_scale()
    if lighting == "bright":
        apply_bright_lighting(next(world for world in bpy.data.worlds if world.node_tree == scene.rgb_node_tree), bright_lighting)
    state = scene_state("training_and_gt")
    expected = record["states"][-1]
    for key in ("world_background_strength", "view_transform", "look",
                "exposure", "gamma", "camera_focal_mm", "samples", "hdri_path", "lights"):
        if state[key] != expected[key]:
            raise ValueError(f"Second renderer state mismatch for {key}: {state[key]} != {expected[key]}")
    if not np.isclose(mesh_radius, record["training_mesh_radius"], atol=2e-4):
        raise ValueError("Second renderer mesh scale changed")
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
                "count": count, "state": state, "registered_bn3d_meshes": mesh_count,
                "mesh_radius": mesh_radius})


def projection_report(root: Path, lighting: str) -> dict:
    """Record geometric crop at PUN's second-renderer intrinsics."""
    out = root / lighting
    matrix = np.asarray(read_json(root / "asset_manifest.json")["imported_axis_matrix"])
    verts = imported_vertices(obj_vertices(root / "asset/House.obj"), matrix) * PUN_SCALE
    camera = read_json(out / "eval_poses.json")["camera"]
    fx, fy = float(camera["fl_x"]), float(camera["fl_y"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    reports = {}
    for split in (*read_json(out / "selection_complete.json").get("policies", POLICIES), "eval"):
        frames = read_json(out / f"{split}_poses.json")["frames"]
        views = []
        for frame in frames:
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
            local = (verts - pose[:3, 3]) @ pose[:3, :3]
            depth = -local[:, 2]
            front = depth > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                x = fx * local[:, 0] / depth + cx
                y = -fy * local[:, 1] / depth + cy
            inside = front & (x >= 0) & (x < 512) & (y >= 0) & (y < 512)
            views.append({"index": frame["index"],
                          "outside_vertex_fraction": float(1 - inside.mean()),
                          "behind_camera_vertices": int((~front).sum()),
                          "projected_bbox_xyxy": [float(np.nanmin(x[front])), float(np.nanmin(y[front])),
                                                  float(np.nanmax(x[front])), float(np.nanmax(y[front]))]})
        reports[split] = views
    return {"camera": camera, "pun_scale": PUN_SCALE, "views": reports}


def finalize_capture(root: Path, lighting: str) -> None:
    out = root / lighting
    selection = read_json(out / "selection_complete.json")
    policies = selection.get("policies", POLICIES)
    for split, total in [(policy, 20) for policy in policies] + [("eval", 40)]:
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
    for policy in policies:
        paths = [out / "training" / policy / "rgba" / f"{i:04d}.png" for i in range(20)]
        sheet = out / policy / "selected_contact_sheet.png"
        sheet.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(paths, [str(i + 1) for i in range(20)], sheet)
        with Image.open(sheet) as image:
            if image.size != (960, 864):
                raise ValueError(f"Invalid contact sheet: {sheet}")
    write_json(out / "projection_report.json", projection_report(root, lighting))
    write_json(out / "capture_complete.json", selection)


def preview(root: Path, lighting: str, seed: int, bright_lighting: str = "v3") -> None:
    """One low-sample pose through both original PUN renderer classes."""
    import bpy
    from config import EnvConfig, SceneType
    from blender_utils import blender_interface
    from nvf.active_mapping.agents.Sampler import sample_poses_spherical

    model_path = prepare(root)
    os.chdir(REPO)
    bpy.ops.wm.read_factory_settings(use_empty=False)
    set_seed(seed)
    pose = sample_poses_spherical(1, radius=RADIUS)
    out = root / "previews" / lighting
    out.mkdir(parents=True, exist_ok=True)
    first = blender_interface.BlenderInterface(resolution=512)
    first.import_mesh(str(model_path), scale=PUN_SCALE)
    first_radius = validate_mesh_scale()
    bpy.context.scene.cycles.samples = 256
    if lighting == "bright":
        apply_bright_lighting(bpy.context.scene.world, bright_lighting)
    first_state = scene_state("avs")
    first_rgb = first.render(str(out / "first_render"), pose, write_cam_params=True)[0]
    Image.fromarray(np.rint(first_rgb.numpy() * 255).clip(0, 255).astype(np.uint8)).save(out / "avs_rgb.png")
    cfg = EnvConfig(scene=SceneType.shapenet, target_path=str(model_path),
                    scale=PUN_SCALE, radius=RADIUS, cycles_samples=256, save_data=False)
    second = v3_scene_class()(cfg)
    mesh_count = register_bn3d_instances()
    second_radius = validate_mesh_scale()
    if lighting == "bright":
        apply_bright_lighting(next(world for world in bpy.data.worlds if world.node_tree == second.rgb_node_tree), bright_lighting)
    second_state = scene_state("training")
    save_rgba(second.render_pose(pose[0]), out / "training_rgba.png")
    write_json(out / "preview.json", {"pose": pose[0].tolist(),
                                    "first": first_state, "second": second_state,
                                    "registered_bn3d_meshes": mesh_count,
                                    "avs_mesh_radius": first_radius,
                                    "training_mesh_radius": second_radius})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preview", "capture", "render", "finalize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lighting", choices=("original", "bright"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bright-lighting", choices=("v2", "v3"), default="v3")
    parser.add_argument("--random-groups", type=int, choices=(2, 3), default=2)
    parser.add_argument("--samples", type=int, help="Debug override for PUN ShapeNetScene Cycles samples; omit for native 10000")
    parser.add_argument("--split", choices=(*POLICIES, "random3", "eval"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root.resolve()))
    elif args.command == "preview":
        if args.lighting is None:
            parser.error("preview requires --lighting")
        preview(args.root.resolve(), args.lighting, args.seed, args.bright_lighting)
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
        capture(args.root.resolve(), args.lighting, args.seed, args.samples,
                args.bright_lighting, args.random_groups)


if __name__ == "__main__":
    main()
