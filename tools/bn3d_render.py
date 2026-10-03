"""Render cached BN3D candidates/test40 with explicit intrinsics; resumable per image."""
import argparse
import time
import numpy as np
from bn3d_common import *

def setup_scene(canonical, samples):
    import bpy
    objects,verts=normalized_mesh(MESH,canonical)
    scene=bpy.context.scene
    scene.render.engine='CYCLES';scene.cycles.samples=samples
    scene.cycles.use_adaptive_sampling=False;scene.cycles.use_denoising=True;scene.cycles.seed=42
    pref=bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type='CUDA';pref.get_devices()
    for d in pref.devices:d.use=d.type=='CUDA'
    assert any(d.use for d in pref.devices),'CUDA render device unavailable'
    scene.cycles.device='GPU';scene.render.resolution_x=512;scene.render.resolution_y=512
    scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1
    scene.render.film_transparent=True
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    world=scene.world or bpy.data.worlds.new('BN3D world');scene.world=world;world.use_nodes=True
    nodes=world.node_tree.nodes;nodes.clear()
    env=nodes.new('ShaderNodeTexEnvironment');env.image=bpy.data.images.load(str(REPO/'data/assets/hdri/gray_hdri.exr'),check_existing=True)
    bg=nodes.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=1.0
    output=nodes.new('ShaderNodeOutputWorld');world.node_tree.links.new(env.outputs['Color'],bg.inputs['Color']);world.node_tree.links.new(bg.outputs['Background'],output.inputs['Surface'])
    bpy.ops.object.camera_add();cam=bpy.context.object;scene.camera=cam
    cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=36;cam.data.sensor_height=36
    cam.data.clip_start=.01;cam.data.clip_end=100
    return scene,cam,verts

def set_intrinsics(cam,K):
    cam.data.lens=float(K[0,0])*36/512
    assert K[0,0]==K[1,1]
    cam.data.shift_x=float(256-K[0,2])/512
    cam.data.shift_y=float(K[1,2]-256)/512

def validate_projection(scene,cam,T,K):
    import bpy
    from mathutils import Matrix,Vector
    from bpy_extras.object_utils import world_to_camera_view
    cam.matrix_world=Matrix(T.tolist());bpy.context.view_layer.update()
    local=np.array([[0,0,-2],[.2,.3,-2],[-.4,.1,-3]])
    world=local@T[:3,:3].T+T[:3,3]
    analytic,_=project(world,T,K)
    measured=np.array([[p.x*512,(1-p.y)*512] for p in [world_to_camera_view(scene,cam,Vector(v)) for v in world]])
    error=float(np.max(np.abs(measured-analytic)))
    assert error<.002,(error,measured,analytic)
    return error

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--profile',choices=(*PROFILES,'both'),default='both')
    parser.add_argument('--split',choices=['candidates','test40','both'],default='both')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--samples',type=int,default=100)
    args=parser.parse_args();out=args.output
    import bpy
    from mathutils import Matrix
    canonical=read_json(out/'canonical_transform.json')
    scene,cam,verts=setup_scene(canonical,args.samples)
    for profile in PROFILES if args.profile=='both' else [args.profile]:
        camera=read_json(out/profile/'camera_config.json');K=np.asarray(camera['K']);set_intrinsics(cam,K)
        first=np.array(read_json(out/'candidate_poses.json')['frames'][0]['transform_matrix'])
        error=validate_projection(scene,cam,first,K)
        config={'camera':camera,'blender_version':bpy.app.version_string,'samples':args.samples,'seed':42,'engine':'CYCLES CUDA','denoising':True,'background':'transparent RGBA archived; RGB alpha-composited over white in display space','lighting':'gray_hdri.exr strength 1.0; no other lights','hdri_sha256':sha256(REPO/'data/assets/hdri/gray_hdri.exr'),'materials':'Blender OBJ/MTL import; textures/material nodes retained','color_transform':'Standard / None / exposure 0 / gamma 1','lens_mm':cam.data.lens,'sensor_width_mm':36,'sensor_height_mm':36,'sensor_fit':'HORIZONTAL','shift_x':cam.data.shift_x,'shift_y':cam.data.shift_y,'clip_start':cam.data.clip_start,'clip_end':cam.data.clip_end,'projection_max_error_px':error,'canonical_transform':canonical}
        config_path=out/profile/'render_config.json'
        if config_path.exists():
            assert read_json(config_path)==json.loads(json.dumps(config)), 'Render configuration changed: use a fresh output directory'
        write_json(config_path,config)
        for split in ['candidates','test40'] if args.split=='both' else [args.split]:
            frames=read_json(out/('candidate_poses.json' if split=='candidates' else 'test_poses.json'))['frames']
            folder=out/profile/split
            for name in ['rgba','masks']: (folder/name).mkdir(parents=True,exist_ok=True)
            write_json(folder/'poses.json',frames)
            report_path=folder/'render_validation.json'
            report=read_json(report_path) if report_path.exists() else {}
            for i,f in enumerate(frames[:args.limit] if args.limit is not None else frames):
                name=f'{i:04d}.png';rgb_path=folder/name;mask_path=folder/'masks'/name;rgba_path=folder/'rgba'/name
                if str(i) in report and all(p.exists() for p in [rgb_path,mask_path,rgba_path]):continue
                start=time.monotonic();T=np.array(f['transform_matrix']);cam.matrix_world=Matrix(T.tolist());bpy.context.view_layer.update()
                scene.render.filepath=str(rgba_path);bpy.ops.render.render(write_still=True)
                with Image.open(rgba_path) as im:a=np.array(im.convert('RGBA'))
                alpha=a[:,:,3].astype(np.float32)/255
                assert np.count_nonzero(alpha) > 100, f'Empty foreground {profile}/{split}/{i}'
                rgb=np.rint(a[:,:,:3].astype(float)*alpha[:,:,None]+255*(1-alpha[:,:,None])).clip(0,255).astype('uint8')
                Image.fromarray(rgb).save(rgb_path);Image.fromarray(a[:,:,3]).save(mask_path)
                ys,xs=np.where(alpha>.01)
                uv,depth=project(verts,T,K)
                report[str(i)]={'seconds':time.monotonic()-start,'foreground_fraction':float((alpha>0).mean()),'mask_bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'touches_image_border':bool(xs.min()==0 or ys.min()==0 or xs.max()==511 or ys.max()==511),'min_vertex_depth':float(depth.min()),'rgb_sha256':sha256(rgb_path),'mask_sha256':sha256(mask_path)}
                write_json(report_path,report)
                print(f'BN3D {profile}/{split} {i+1}/{len(frames)} {report[str(i)]["seconds"]:.2f}s',flush=True)
            if split=='test40' and len(report)==40:
                K=np.array(camera['K']);write_json(folder/'transforms.json',{'w':512,'h':512,'camera_model':'OPENCV','fl_x':K[0,0],'fl_y':K[1,1],'cx':K[0,2],'cy':K[1,2],'orientation_override':'none','frames':[dict(file_path=f'{i:04d}.png',transform_matrix=f['transform_matrix'],test_id=i) for i,f in enumerate(frames)]})
            if report:
                ids=sorted(map(int,report))[::max(1,len(report)//20)][:20]
                contact_sheet([folder/f'{i:04d}.png' for i in ids],ids,folder/'contact_sheet.png')

if __name__=='__main__':main()
