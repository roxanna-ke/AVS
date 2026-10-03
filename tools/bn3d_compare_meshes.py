"""Create one interactive HTML page for the four BN3D NeRF mesh exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS = [
    ("PUN-rendered · AVS20", "pun_rendered", "avs20", "#d95f02"),
    ("PUN-rendered · Random20", "pun_rendered", "random20", "#1b9e77"),
    ("BN3D-faithful · AVS20", "bn3d_faithful", "avs20", "#7570b3"),
    ("BN3D-faithful · Random20", "bn3d_faithful", "random20", "#e7298a"),
]


def latest_mesh(profile: str, subset: str, steps=None) -> Path:
    if steps is not None:
        from bn3d_orbit_video import select_checkpoint
        config, _ = select_checkpoint(SimpleNamespace(
            profile=profile, subset=subset, steps=steps, config=None,
            runs_root=PROJECT_ROOT / "outputs" / "bn3d_building1"))
        mesh = config.parent / "mesh" / "tsdf_mesh.ply"
        if not mesh.is_file():
            raise FileNotFoundError(f"Checkpoint is complete but its mesh is missing: {mesh}")
        return mesh
    root = PROJECT_ROOT / "outputs" / "bn3d_building1" / profile / subset / "instant_ngp"
    matches = list(root.glob("**/mesh/tsdf_mesh.ply"))
    if not matches:
        raise FileNotFoundError(
            f"No mesh found for {profile}/{subset} under {root}. Run the corresponding reconstruction script first."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_mesh(path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    if max_faces > 0 and len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError(f"Mesh became empty after simplification: {path}")
    return vertices, triangles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["pun_rendered", "bn3d_faithful"])
    parser.add_argument("--subset", choices=["avs20", "random20"])
    parser.add_argument("--steps", type=int, help="Select the mesh paired with a completed run of exactly this many iterations.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output standalone HTML file.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=40000,
        help="Target face count for browser preview simplification (some meshes retain more); use 0 to disable.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    step_suffix = f"_{args.steps}" if args.steps else ""
    runs = [entry for entry in RUNS
            if (args.profile is None or entry[1] == args.profile)
            and (args.subset is None or entry[2] == args.subset)]
    if args.output is None:
        output_root = PROJECT_ROOT / "outputs" / "bn3d_building1"
        if len(runs) == 1:
            args.output = output_root / runs[0][1] / runs[0][2] / f"mesh{step_suffix}.html"
        else:
            name = "_".join(filter(None, [args.profile, args.subset])) or "all_meshes"
            args.output = output_root / f"{name}{step_suffix}_comparison.html"
    rows = 2 if len(runs) > 2 else 1
    cols = min(2, len(runs))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[entry[0] for entry in runs],
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    sources = []
    for index, (label, profile, subset, color) in enumerate(runs):
        path = latest_mesh(profile, subset, args.steps)
        vertices, triangles = load_mesh(path, args.max_faces)
        sources.append(dict(profile=profile, subset=subset, training_steps=args.steps,
                            mesh=str(path), displayed_triangles=len(triangles)))
        row, col = divmod(index, cols)
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color=color,
                opacity=1.0,
                flatshading=True,
                lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.15, "roughness": 0.8},
                hoverinfo="skip",
                name=label,
                showscale=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        print(f"{label}: {path} ({len(triangles):,} displayed triangles)")

    scene = dict(
        aspectmode="data",
        xaxis=dict(title="X", range=[-1, 1]),
        yaxis=dict(title="Y", range=[-1, 1]),
        zaxis=dict(title="Z (up)", range=[-1, 1]),
        camera=dict(eye=dict(x=1.6, y=-1.8, z=1.35)),
    )
    fig.update_layout(
        title=(f"BN3D building1 · {runs[0][0]}" if len(runs) == 1 else "BN3D building1 · interactive mesh comparison")
              + (f" · {args.steps} steps" if args.steps else ""),
        margin=dict(l=0, r=0, t=70, b=0),
        height=1000 if rows == 2 else 750,
        showlegend=False,
    )
    fig.update_scenes(**scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(
        {"html": str(args.output), "target_faces": args.max_faces, "sources": sources}, indent=2) + "\n")
    print(f"Saved interactive comparison: {args.output}")


if __name__ == "__main__":
    main()
