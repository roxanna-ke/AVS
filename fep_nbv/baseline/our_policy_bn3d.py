"""Offline fixed-pool BN3D AVS using original PUN interpolation/all/small rules."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools'),str(REPO/'08-vit-train')]
import numpy as np
import torch
import timm
from PIL import Image
from regress_model import ViTRegressor
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints
from bn3d_common import DEFAULT_OUTPUT,PROFILES,WORKSPACE,read_json,write_json,sha256,contact_sheet,export_subset


def interpolate_uncertainty(relative_history, pose_history, candidates):
    """Same loop/arithmetic as our_policy_single.interpolate_uncertainty.

    The support radius is arbitrary because both direction vectors are normalized.
    Keep the original generator, including its float32 pose representation.
    """
    rows=[]
    for i,relative in enumerate(relative_history):
        support=generate_HEALPix_viewpoints(n_side=2,original_viewpoint=pose_history[i,4:],radius=2.73)
        row=[]
        for pose in candidates:
            cosine=np.dot(support[:,4:]/np.linalg.norm(support[:,4:],axis=1,keepdims=True),pose[4:]/np.linalg.norm(pose[4:]))
            angles=np.arccos(np.clip(cosine,-1,1))
            nearby=np.where(angles<np.radians(30))[0]
            if not len(nearby):raise ValueError('No HEALPix neighbors within 30 degrees')
            weights=np.exp(-angles[nearby]);weights=weights/sum(weights)
            row.append(np.dot(weights,relative[nearby]))
        rows.append(np.array(row))
    result=np.array(rows)
    if not np.isfinite(result).all():raise ValueError('Nonfinite interpolation')
    return result


def filter_small(history,mode='PSNR',threshold=.1):
    normalized=np.zeros_like(history)
    for i,row in enumerate(history):
        if row.max()>row.min():normalized[i]=(row-row.min())/(row.max()-row.min())
        else:normalized[i]=0
    if mode in ['PSNR','SSIM']:bad=np.any(normalized>=1-threshold,axis=0)
    elif mode in ['uncertainty','MSE','LPIPS']:bad=np.any(normalized<=threshold,axis=0)
    else:raise ValueError(mode)
    return ~bad


def rank_all(history,mode='PSNR'):
    score=np.prod(history,axis=0)
    if not np.isfinite(score).all():raise ValueError('Nonfinite history product')
    if mode in ['PSNR','SSIM']:index=int(np.argmin(score))
    elif mode in ['uncertainty','MSE','LPIPS']:index=int(np.argmax(score))
    else:raise ValueError(mode)
    return index,score


def select(predict,poses,count=20):
    selected=[0];relative=[];steps=[];absolute=[]
    while True:
        prediction=np.asarray(predict(selected[-1]))
        assert prediction.shape==(48,) and np.isfinite(prediction).all()
        relative.append(prediction)
        # Save every observation's interpolation onto the full pool for diagnostics.
        absolute.append(interpolate_uncertainty([prediction],poses[selected[-1]:selected[-1]+1],poses)[0])
        if len(selected)==count:break
        ids=np.array([i for i in range(len(poses)) if i not in selected],dtype=int)
        history=interpolate_uncertainty(relative,poses[selected],poses[ids])
        keep=filter_small(history);fallback=not bool(keep.any())
        eligible=ids if fallback else ids[keep]
        # Original PUN recomputes interpolation after filtering.
        filtered=interpolate_uncertainty(relative,poses[selected],poses[eligible])
        choice,score=rank_all(filtered);chosen=int(eligible[choice])
        steps.append({'step':len(selected),'previous_candidate':selected[-1],'remaining_ids':ids.tolist(),'eligible_ids':eligible.tolist(),'filter_fallback':fallback,'aggregate_scores':score.tolist(),'selected_candidate':chosen})
        selected.append(chosen)
        print(f'AVS {len(selected):02d}/20 candidate={chosen:04d} eligible={len(eligible)} fallback={fallback}',flush=True)
    return selected,np.array(relative),np.array(absolute),steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--profile',choices=(*PROFILES,'both'),default='both')
    parser.add_argument('--checkpoint',type=Path,default=WORKSPACE/'models/UPNet/vit_small_patch16_224_PSNR_250425172703/best_vit_regressor.pth')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    assert torch.cuda.is_available(),'UPNet CUDA unavailable'
    model=ViTRegressor('vit_small_patch16_224',output_dim=48,pretrained=False)
    # Full trained weights supersede backbone initialization; no download needed.
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu'),strict=True);model.cuda().eval()
    data_cfg=timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform=timm.data.create_transform(**data_cfg)
    frames=read_json(args.output/'candidate_poses.json')['frames']
    poses=torch.from_numpy(np.loadtxt(args.output/'candidate_poses.txt').astype(np.float32))
    assert len(frames)==len(poses)==200
    random_ids=[0]+random.Random(args.seed).sample(range(1,200),19)
    for profile in PROFILES if args.profile=='both' else [args.profile]:
        root=args.output/profile;camera=read_json(root/'camera_config.json')
        for i in range(200):
            if not (root/'candidates'/f'{i:04d}.png').is_file():raise FileNotFoundError(f'{profile} candidate {i}')
        def predict(i):
            with Image.open(root/'candidates'/f'{i:04d}.png') as im:rgb=np.asarray(im.convert('RGB')).astype(np.float32)/255
            image=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)
            # Match original float BCHW tensor preprocessing, not PIL preprocessing.
            with torch.inference_mode():return model(transform(image).cuda())[0].cpu().numpy()
        selected,relative,absolute,steps=select(predict,poses)
        for name,ids in [('avs20',selected),('random20',random_ids),('all200',list(range(200)))]:
            export_subset(root,name,ids,frames,camera)
        result=root/'avs20'
        write_json(result/'selected_indices.json',selected)
        write_json(result/'selection_order.json',{'initial_candidate':0,'order':selected,'steps':steps})
        np.savetxt(result/'selected_poses.txt',poses[selected].numpy())
        np.save(result/'uncertainty_history.npy',relative)
        np.save(result/'absolute_uncertainty_history.npy',absolute)
        write_json(result/'selection_config.json',{'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256(args.checkpoint),'seed':args.seed,'mode':'PSNR','select_method':'all','aggregation':'numpy.prod','optimization':'argmin','delete_method':'small','threshold':.1,'support':'original generate_HEALPix_viewpoints n_side=2 radius=2.73','interpolation':'original 30deg exp(-angle) normalized weights','preprocessing':data_cfg,'tensor_preprocessing':str(transform),'torch':torch.__version__,'timm':timm.__version__,'fallback_count':sum(s['filter_fallback'] for s in steps),'attention_backend':'math SDP; flash/memory-efficient disabled for determinism','deterministic_replay':'saved predictions replayed below','original_policy_sha256':sha256(REPO/'fep_nbv/baseline/our_policy_single.py'),'direction_generator_sha256':sha256(REPO/'fep_nbv/utils/generate_viewpoints.py')})
        for name,ids in [('avs20',selected),('random20',random_ids)]:
            contact_sheet([root/'candidates'/f'{i:04d}.png' for i in ids],[f'{j+1}: ID {i}' for j,i in enumerate(ids)],root/name/'selected_contact_sheet.png')
        cached=dict(zip(selected,relative))
        replay=select(lambda i:cached[i],poses)[0]
        assert replay==selected and len(set(selected))==20
        write_json(result/'selection_validation.json',{'unique':True,'all_in_pool':True,'initial_zero':True,'deterministic_prediction_replay':True,'random_seed':args.seed,'random_ids':random_ids})
        print(profile,'selected',selected,flush=True)

if __name__=='__main__':main()
