"""Prepare shared canonical geometry, fixed cameras and deterministic held-out poses."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from bn3d_common import *

def look_at(position):
    back=position/np.linalg.norm(position)
    right=np.cross([0,0,1],back); right/=np.linalg.norm(right)
    up=np.cross(back,right)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([right,up,back]);pose[:3,3]=position
    return pose

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();out=args.output
    meta=read_json(METADATA);assert len(meta['frames'])==200
    raw=np.array([list(map(float,l.split()[1:4])) for l in MESH.read_text().splitlines() if l.startswith('v ')])
    objects,world=import_mesh(MESH)
    wb=bounds(world);center=wb.mean(0);scale=2/np.ptp(world,axis=0).max()
    N=np.eye(4);N[:3,:3]*=scale;N[:3,3]=-scale*center
    canonical=(world-center)*scale;cb=bounds(canonical)
    import bpy
    assets=[MESH,MESH.with_suffix('.mtl'),*sorted(MESH.parent.glob('*.png')),METADATA]
    record={'mesh_path':str(MESH),'metadata_path':str(METADATA),'source_sha256':{str(p):sha256(p) for p in assets},'blender_version':bpy.app.version_string,'import_matrices':{o.name:np.array(o.matrix_world) for o in objects},'raw_obj_aabb':bounds(raw),'imported_world_aabb':wb,'canonical_aabb':cb,'center_world':center,'scale':scale,'world_to_canonical':N,'canonical_to_world':np.linalg.inv(N),'ground_z':cb[0,2],'camera_convention':'OpenGL: +X right, +Y up, -Z forward','metadata_rotations_unchanged':True,'loaded_textures':{i.name:list(i.size) for i in bpy.data.images if i.source=='FILE'}}
    assert all(record['loaded_textures'].get(name)==[5120,5120] for name in ['House_Diff_5k.png','House_Nrm_5K.png','House_Spec_5K.png'])
    matrices=[];frames=[];quats=[];validation=[]
    Kbn=np.asarray(meta['frames'][0]['intrinsics'])
    for i,f in enumerate(meta['frames']):
        np.testing.assert_allclose(f['intrinsics'],Kbn)
        T=np.array(f['camtoworld']);np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=3e-6)
        assert np.linalg.det(T[:3,:3])>0.99999
        T[:3,3]=scale*(T[:3,3]-center)
        assert T[2,3]>=cb[0,2], f'camera {i} below ground'
        assert not np.all((T[:3,3]>=cb[0])&(T[:3,3]<=cb[1])), f'camera {i} inside building AABB'
        assert np.dot(-T[:3,2],-T[:3,3])>0, f'camera {i} faces away'
        matrices.append(T);quats.append(np.r_[Rotation.from_matrix(T[:3,:3]).as_quat(),T[:3,3]])
        frames.append({'candidate_id':i,'source_rgb_path':f['rgb_path'],'source_segmentation_path':f['segmentation_path'],'transform_matrix':T})
    write_json(out/'canonical_transform.json',record)
    triangles=[];offset=0
    for obj in objects:
        obj.data.calc_loop_triangles()
        triangles.extend([[int(v)+offset for v in tri.vertices] for tri in obj.data.loop_triangles])
        offset+=len(obj.data.vertices)
    np.savez_compressed(out/'canonical_geometry.npz',vertices=canonical,triangles=np.array(triangles,dtype=np.int32))
    write_json(out/'candidate_poses.json',{'frames':frames,'coordinate_system':'canonical OpenGL'})
    np.savetxt(out/'candidate_poses.txt',quats)
    camera_centers=np.array(matrices)[:,:3,3]
    write_json(out/'scene_config.json',{'object_aabb':cb,'target_aabb':cb,'camera_aabb':bounds(camera_centers),'candidate_count':200,'candidate_radius_not_forced':True})
    radius=float(np.median(np.linalg.norm(camera_centers,axis=1)))
    tests=[]
    for j,deg in enumerate([10,30,50,70]):
        for k in range(10):
            az=2*np.pi*(k+0.37+0.13*j)/10;el=np.deg2rad(deg)
            direction=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
            r=radius
            for _ in range(100):
                T=look_at(r*direction);uv,depth=project(canonical,T,Kbn)
                if depth.min()>0 and uv.min()>=8 and uv.max()<=504:break
                r*=1.01
            else:raise RuntimeError('Unable to frame test camera')
            assert np.linalg.norm(camera_centers-T[:3,3],axis=1).min()>1e-3
            tests.append({'test_id':len(tests),'transform_matrix':T,'radius':r,'elevation_degrees':deg})
    write_json(out/'test_poses.json',{'generation':'canonical look-at origin; 4 elevation rings x 10 azimuths; deterministic offset; radius starts at candidate median and expands only to ensure 8px margin with BN3D K','frames':tests})
    for profile in PROFILES:
        K=Kbn if profile=='bn3d_faithful' else np.array([[525.,0,256.],[0,525.,256.],[0,0,1]])
        camera={'profile':profile,'K':K,'width':512,'height':512,'camera_convention':'OpenGL','pixel_convention':'image edges 0..512, pixel centers i+0.5; K used literally in Blender and future training rays','canonical_transform':'../canonical_transform.json'}
        write_json(out/profile/'camera_config.json',camera)
        per_view=[]
        for i,T in enumerate(matrices):
            uv,z=project(canonical,T,K)
            per_view.append({'candidate_id':i,'min_depth':z.min(),'projected_vertex_bounds':bounds(uv),'fraction_vertices_outside':np.mean(np.any((uv<0)|(uv>512),axis=1))})
        write_json(out/profile/'projection_validation.json',per_view)
    write_json(out/'prepare_validation.json',{'candidate_count':200,'test_count':40,'all_cameras_above_ground':True,'all_cameras_outside_mesh_aabb':True,'rotations_valid':True,'candidate_radius_range':np.quantile(np.linalg.norm(camera_centers,axis=1),[0,.5,1]),'test_radius_range':np.quantile([t['radius'] for t in tests],[0,.5,1])})
    print('Prepared',out,flush=True)

if __name__=='__main__':main()
