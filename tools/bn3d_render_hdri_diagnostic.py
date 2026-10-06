"""Render BN3D building1 AVS view 0002 with HDRI illumination only.

Run from the PUN repository root with /mnt/home/conda-envs/pun/bin/python.
The default output is a new file beside the existing diagnostic images.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from bn3d_pun_native import DEFAULT_ROOT, HDRI, RADIUS, register_bn3d_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--view", type=int, default=2)
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--strength", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.strength <= 0 or args.samples < 1:
        parser.error("strength and samples must be positive")

    import bpy
    import torch
    from blender_utils import blender_interface
    from config import EnvConfig, SceneType
    from fep_nbv.env.shapenet_scene import ShapeNetScene

    os.chdir(Path(__file__).resolve().parents[1])
    bpy.ops.wm.read_factory_settings(use_empty=False)
    blender_interface.BlenderInterface(resolution=512)
    cfg = EnvConfig(
        scene=SceneType.shapenet,
        target_path=str(args.root / "asset/House.obj"),
        scale=1.0,
        radius=RADIUS,
        cycles_samples=args.samples,
        save_data=False,
    )
    renderer = ShapeNetScene(cfg)
    register_bn3d_instances()

    # PUN creates a second RGB world. Set both worlds so either render path
    # receives exactly the same environment light, and remove all scene lamps.
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for world in (bpy.context.scene.world,
                  next(w for w in bpy.data.worlds if w.node_tree == renderer.rgb_node_tree)):
        world.use_nodes = True
        nodes = world.node_tree.nodes
        bg = nodes.get("Background")
        bg.inputs["Strength"].default_value = args.strength
        env = next(n for n in nodes if n.type == "TEX_ENVIRONMENT")
        env.image = bpy.data.images.load(str(HDRI), check_existing=True)
    settings = bpy.context.scene.view_settings
    settings.view_transform = "Standard"
    settings.look = "None"
    settings.exposure = 0.0
    settings.gamma = 1.0
    assert not any(o.type == "LIGHT" for o in bpy.context.scene.objects)

    poses = np.load(args.root / "bright/avs20_poses.npy")
    start = args.view if args.start is None else args.start
    if start < 0 or args.count < 1 or start + args.count > len(poses):
        parser.error(f"requested views must be in [0, {len(poses) - 1}]")
    if args.output is not None and args.count != 1:
        parser.error("--output requires --count 1")
    for view in range(start, start + args.count):
        output = args.output or (args.root / "diagnostics" /
            (f"hdri_strength{args.strength:g}_avs_{view:04d}.png"
             if args.start is not None else
             f"brighter_hdri_only_same_pose_{view:04d}.png"))
        if output.exists():
            print(f"Keeping existing {output}", flush=True)
            continue
        rgba = renderer.render_pose(torch.from_numpy(poses[view]))
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(output)
        print(f"Saved {output}; HDRI strength={args.strength}, samples={args.samples}, lights=0", flush=True)


if __name__ == "__main__":
    main()
