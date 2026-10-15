"""Render cached evaluation views from completed PUN reference checkpoints."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("WANDB_MODE", "disabled")
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from PIL import Image, ImageDraw

from bn3d_common import read_json, write_json
from bn3d_pun_native_v3 import PUN_SCALE, RADIUS
from bn3d_pun_reference_run import validate_capture, validate_pair

ROOT_DEFAULT = REPO.parents[1] / "outputs/bn3d_pun_v3_v2bright_random100/building1"
STEPS = (2000, 5000, 10000, 15000, 20000)


def completed_run(root: Path, lighting: str, iterations: int) -> Path:
    base = root / "runs_reference" / lighting / str(iterations)
    candidates = [base] if (base / "complete.json").is_file() else sorted(
        path.parent for path in base.glob("attempts/*/complete.json")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one completed run for {lighting}/{iterations}, found: {candidates}")
    return candidates[0]


def init_mapper(config_home: Path, dataset_path: Path, iterations: int, lighting: str):
    import torch
    from config import ExpConfig, ModelType, SceneType
    from fep_nbv.utils.utils import set_params
    from nvf.active_mapping.active_mapping import ActiveMapper

    class RenderMapper(ActiveMapper):
        def init_instant_ngp_config(self, config_home: str, dataset_path: str) -> str:
            path = super().init_instant_ngp_config(config_home, dataset_path)
            import yaml
            self.config.max_num_iterations = iterations
            self.config.optimizers["fields"]["scheduler"].max_steps = iterations
            self.config.vis = "none"
            self.config.experiment_name = f"render-{lighting}-{iterations}"
            self.config.project_name = "bn3d-pun-reference-render"
            Path(path).write_text(yaml.dump(self.config))
            return path

    mapper = RenderMapper()
    mapper.fov = 30
    mapper.train_img_size = (512, 512)
    mapper.initialize_config(str(config_home), str(dataset_path), model="instant-ngp")

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
    cfg.exp_folder = str(config_home)

    mapper.reset()
    set_params(cfg, mapper)
    mapper.trainer.pipeline.model.renderer_entropy.set_iteration(0)
    return mapper, cfg


def render_checkpoint(root: Path, lighting: str, iterations: int, run: Path, step: int) -> None:
    import torch
    from nvf.metric.MetricTracker import RefMetricTracker

    checkpoint = run / f"step_{step:05d}.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data = validate_capture(root, lighting)
    # Keep inference setup files in /tmp; never rewrite the saved training config.
    with tempfile.TemporaryDirectory(prefix="bn3d-pun-render-") as temporary:
        mapper, cfg = init_mapper(Path(temporary), run / "config" / "dataset", iterations, lighting)
        try:
            mapper.load_ckpt(str(checkpoint))
            mapper.trainer.pipeline.eval()

            # RefMetricTracker.visualize only needs these fields. Avoid constructing its
            # mesh metrics, since rendering images does not need geometry evaluation.
            tracker = SimpleNamespace(
                ref_poses=torch.stack([torch.from_numpy(pose.copy()) for pose in data["eval_poses"]]),
                img_size=(512, 512),
                config=cfg,
            )
            with torch.no_grad():
                predictions = RefMetricTracker.visualize(tracker, mapper)
            pred_dir = run / "predictions" / f"step_{step:05d}"
            pred_dir.mkdir(parents=True, exist_ok=True)

            gt_dir = data["source"] / "eval" / "rgb_black"
            thumbs = []
            for index, tensor in enumerate(predictions):
                rgb = (tensor.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
                prediction = Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")
                prediction.save(pred_dir / f"{index:04d}.png")

                gt = Image.open(gt_dir / f"{index:04d}.png").convert("RGB")
                tile = Image.new("RGB", (256 * 2, 144 + 22), "white")
                draw = ImageDraw.Draw(tile)
                draw.text((4, 3), f"view {index:02d} | GT", fill="black")
                draw.text((260, 3), "prediction", fill="black")
                tile.paste(gt.resize((256, 144)), (0, 22))
                tile.paste(prediction.resize((256, 144)), (256, 22))
                thumbs.append(tile)

            cols, rows = 4, (len(thumbs) + 3) // 4
            sheet = Image.new("RGB", (cols * 512, rows * 166), "#222222")
            for index, tile in enumerate(thumbs):
                sheet.paste(tile, ((index % cols) * 512, (index // cols) * 166))
            sheet.save(run / "predictions" / f"contact_step_{step:05d}.jpg", quality=92)
            write_json(pred_dir / "manifest.json", {
                "lighting": lighting, "iterations": iterations, "step": step,
                "checkpoint": str(checkpoint), "num_eval_views": len(predictions),
                "camera_source": str(data["source"] / "eval_poses.npy"),
                "ground_truth_source": str(gt_dir),
                "rendering": "RefMetricTracker.visualize; RGB clipped to [0,1], saved as PNG",
            })
            print(f"Rendered {lighting}/{iterations} step {step}: {pred_dir}", flush=True)
        finally:
            del mapper
            torch.cuda.empty_cache()


def main() -> None:
    root = Path(os.environ.get("BN3D_CAPTURE_ROOT", str(ROOT_DEFAULT))).resolve()
    validate_pair(root)
    for lighting in ("original", "bright"):
        for iterations in (2000, 20000):
            run = completed_run(root, lighting, iterations)
            milestones = (2000,) if iterations == 2000 else STEPS
            for step in milestones:
                pred_dir = run / "predictions" / f"step_{step:05d}"
                if (pred_dir / "manifest.json").is_file():
                    print(f"Already rendered: {pred_dir}", flush=True)
                    continue
                render_checkpoint(root, lighting, iterations, run, step)


if __name__ == "__main__":
    main()
