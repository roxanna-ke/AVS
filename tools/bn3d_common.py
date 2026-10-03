"""Shared BN3D stage 1/2 utilities; no ShapeNet/NVF initialization."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
DEFAULT_OUTPUT = WORKSPACE / 'data/bn3d_processed/building1'
MESH = WORKSPACE / 'data/buildnet3d/building_models/building1/House.obj'
METADATA = WORKSPACE / 'data/buildnet3d/pose/building1_meta_data.json'
PROFILES = ('pun_rendered', 'bn3d_faithful')

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')
    tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def bounds(v):
    return np.stack([v.min(0), v.max(0)])

def import_mesh(path):
    import bpy
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if bpy.app.version[0] < 4:
        bpy.ops.import_scene.obj(filepath=str(path), split_mode='OFF', axis_forward='-Z', axis_up='Y')
    else:
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis='NEGATIVE_Z', up_axis='Y')
    objects = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    assert objects, 'OBJ import produced no meshes'
    bpy.context.view_layer.update()
    verts = np.concatenate([np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices]) for o in objects])
    return objects, verts

def normalized_mesh(mesh_path, canonical):
    import bpy
    from mathutils import Matrix
    objects, verts = import_mesh(mesh_path)
    np.testing.assert_allclose(bounds(verts), canonical['imported_world_aabb'], atol=2e-5)
    N = np.asarray(canonical['world_to_canonical'])
    for o in objects: o.matrix_world = Matrix(N.tolist()) @ o.matrix_world
    bpy.context.view_layer.update()
    verts = np.concatenate([np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices]) for o in objects])
    np.testing.assert_allclose(bounds(verts), canonical['canonical_aabb'], atol=2e-5)
    return objects, verts

def project(vertices, pose, K):
    # OpenGL camera coordinates -> image-edge coordinates, pixel centers at i+0.5.
    q = (vertices - pose[:3, 3]) @ pose[:3, :3]
    depth = -q[:, 2]
    return np.column_stack((K[0,0]*q[:,0]/depth+K[0,2], -K[1,1]*q[:,1]/depth+K[1,2])), depth

def contact_sheet(paths, labels, output, columns=5, tile=192):
    rows = (len(paths)+columns-1)//columns
    sheet = Image.new('RGB', (columns*tile, rows*(tile+24)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (p, label) in enumerate(zip(paths, labels)):
        with Image.open(p) as im: thumb = im.convert('RGB').resize((tile,tile))
        x,y = (i%columns)*tile, (i//columns)*(tile+24)
        sheet.paste(thumb,(x,y)); draw.text((x+5,y+tile+4),str(label),fill='black')
    sheet.save(output)

def export_subset(profile_dir, name, indices, frames, camera):
    import os
    out = Path(profile_dir)/name
    (out/'images').mkdir(parents=True,exist_ok=True)
    (out/'masks').mkdir(exist_ok=True)
    exported=[]
    for i in indices:
        for folder, source in [('images',Path(profile_dir)/'candidates'/f'{i:04d}.png'),('masks',Path(profile_dir)/'candidates/masks'/f'{i:04d}.png')]:
            if not source.is_file(): raise FileNotFoundError(source)
            dest=out/folder/f'{i:04d}.png'
            if not dest.exists(): dest.symlink_to(os.path.relpath(source,dest.parent))
        exported.append({'candidate_id':i,'file_path':f'images/{i:04d}.png','alpha_path':f'masks/{i:04d}.png','transform_matrix':frames[i]['transform_matrix']})
    K=np.asarray(camera['K'])
    write_json(out/'transforms.json',{'camera_model':'OPENCV','w':512,'h':512,'fl_x':K[0,0],'fl_y':K[1,1],'cx':K[0,2],'cy':K[1,2], 'orientation_override':'none','frames':exported,'coordinate_system':'canonical OpenGL; no further normalization','usage':'NVF adapter must use every listed image; alpha_path is for visualization, not foreground-only training sampling'})
    write_json(out/'indices.json',indices)
    write_json(out/'camera_config.json',dict(camera, canonical_transform='../../canonical_transform.json'))
