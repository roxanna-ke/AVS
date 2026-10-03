"""Validate completed stage 1/2 assets, pose transforms, cached images and selection."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from bn3d_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=p.parse_args();out=args.output
    c=read_json(out/'canonical_transform.json')
    for path,digest in c['source_sha256'].items():assert sha256(path)==digest,f'Source changed: {path}'
    original=np.array([f['camtoworld'] for f in read_json(METADATA)['frames']])
    poses=np.array([f['transform_matrix'] for f in read_json(out/'candidate_poses.json')['frames']])
    np.testing.assert_array_equal(original[:,:3,:3],poses[:,:3,:3])
    np.testing.assert_allclose(poses[:,:3,3],c['scale']*(original[:,:3,3]-c['center_world']),atol=1e-10)
    quat=np.loadtxt(out/'candidate_poses.txt');np.testing.assert_allclose(Rotation.from_quat(quat[:,:4]).as_matrix(),poses[:,:3,:3],atol=2e-6)
    tests=np.array([f['transform_matrix'] for f in read_json(out/'test_poses.json')['frames']])
    assert len(poses)==200 and len(tests)==40
    assert np.linalg.norm(tests[:,None,:3,3]-poses[None,:,:3,3],axis=-1).min()>1e-3
    summary={'source_hashes_unchanged':True,'metadata_rotations_unchanged':True,'normalization_verified':True,'test_cameras_distinct':True,'profiles':{}}
    for profile in PROFILES:
        root=out/profile;config=read_json(root/'render_config.json');camera=read_json(root/'camera_config.json')
        assert config['camera']==camera and config['projection_max_error_px']<.002
        # Independently check every saved candidate/test pose against Blender projection.
        import bpy
        from bn3d_render import set_intrinsics, validate_projection
        if bpy.context.scene.camera is None:
            bpy.ops.object.camera_add()
            bpy.context.scene.camera=bpy.context.object
        scene=bpy.context.scene;cam=scene.camera
        scene.render.resolution_x=scene.render.resolution_y=512
        scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
        cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=cam.data.sensor_height=36
        set_intrinsics(cam,np.asarray(camera['K']))
        max_error=max(validate_projection(scene,cam,T,np.asarray(camera['K'])) for T in np.concatenate([poses,tests]))
        effective_K=[[cam.data.lens*512/36,0,256-cam.data.shift_x*512],[0,cam.data.lens*512/36,256+cam.data.shift_y*512],[0,0,1]]
        np.testing.assert_allclose(effective_K,camera['K'],atol=.002,rtol=0)
        write_json(root/'camera_projection_validation.json',{'requested_K':camera['K'],'effective_K':effective_K,'all_240_camera_projection_max_error_px':max_error,'pixel_convention':camera['pixel_convention']})
        report={'all_240_camera_projection_max_error_px':max_error,'effective_K':effective_K}
        for split,count in [('candidates',200),('test40',40)]:
            validation=read_json(root/split/'render_validation.json');assert len(validation)==count
            for i in range(count):
                for folder,mode in [('', 'RGB'),('masks','L'),('rgba','RGBA')]:
                    path=root/split/folder/f'{i:04d}.png'
                    with Image.open(path) as image:
                        image.load();assert image.size==(512,512) and image.mode==mode
                assert sha256(root/split/f'{i:04d}.png')==validation[str(i)]['rgb_sha256']
                assert sha256(root/split/'masks'/f'{i:04d}.png')==validation[str(i)]['mask_sha256']
            report[split]={'count':count,'border_touching_ids':[int(i) for i,r in validation.items() if r['touches_image_border']],'mean_render_seconds':np.mean([r['seconds'] for r in validation.values()])}
        for subset,count in [('avs20',20),('random20',20),('all200',200)]:
            ids=read_json(root/subset/'indices.json');assert len(ids)==len(set(ids))==count and 0 in ids
            frames=read_json(root/subset/'transforms.json')['frames'];assert len(frames)==count
            for i,f in zip(ids,frames):
                assert f['candidate_id']==i and 0<=i<200
                np.testing.assert_array_equal(f['transform_matrix'],poses[i])
                assert (root/subset/f['file_path']).resolve()==(root/'candidates'/f'{i:04d}.png').resolve()
            subset_camera=read_json(root/subset/'camera_config.json')
            assert subset_camera==dict(camera,canonical_transform='../../canonical_transform.json')
            assert (root/subset/subset_camera['canonical_transform']).resolve()==(out/'canonical_transform.json').resolve()
        selected=read_json(root/'avs20/selected_indices.json');assert selected==read_json(root/'avs20/indices.json')
        assert np.load(root/'avs20/uncertainty_history.npy').shape==(20,48)
        assert np.load(root/'avs20/absolute_uncertainty_history.npy').shape==(20,200)
        report['geometrically_outside_candidate_ids']=[v['candidate_id'] for v in read_json(root/'projection_validation.json') if v['fraction_vertices_outside']>0]
        report['border_note']='Mask border flags include antialiasing/filter footprints; see geometric projection bounds separately.'
        report['avs_indices']=selected;report['random_indices']=read_json(root/'random20/indices.json')
        report['projection_max_error_px']=config['projection_max_error_px']
        report['selection_fallbacks']=read_json(root/'avs20/selection_config.json')['fallback_count']
        summary['profiles'][profile]=report
    assert summary['profiles'][PROFILES[0]]['random_indices']==summary['profiles'][PROFILES[1]]['random_indices']
    summary['random_ids_shared']=True
    write_json(out/'stage12_validation.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
