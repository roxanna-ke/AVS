"""Check fixed-pool arithmetic against original functions without importing NVF."""
import ast
import contextlib
import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fep_nbv/baseline'))
import our_policy_bn3d as new
import numpy as np
import torch
from bn3d_common import REPO,DEFAULT_OUTPUT,write_json

def main():
    tree=ast.parse((REPO/'fep_nbv/baseline/our_policy_single.py').read_text())
    names={'interpolate_uncertainty','filter_viewpoints_small','view_select_all'}
    code=compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'original_policy_functions','exec')
    scope={'np':np,'torch':torch,'radius':2.73,'generate_HEALPix_viewpoints':new.generate_HEALPix_viewpoints}
    exec(code,scope)
    poses=torch.from_numpy(np.loadtxt(DEFAULT_OUTPUT/'candidate_poses.txt').astype(np.float32))
    rng=np.random.default_rng(7);history=rng.uniform(10,30,(3,48)).astype(np.float32)
    indices=[0,31,100]
    a=scope['interpolate_uncertainty'](history,poses[indices],poses)
    b=new.interpolate_uncertainty(history,poses[indices],poses)
    np.testing.assert_array_equal(a,b)
    for mode in ['PSNR','SSIM','MSE','LPIPS','uncertainty']:
        with contextlib.redirect_stdout(io.StringIO()):
            filtered=scope['filter_viewpoints_small'](a,poses[indices],poses,mode)
        np.testing.assert_array_equal(filtered,poses[new.filter_small(a,mode)])
        with contextlib.redirect_stdout(io.StringIO()):chosen=scope['view_select_all'](a,poses,mode)
        i,_=new.rank_all(a,mode);np.testing.assert_array_equal(chosen,poses[i])
    write_json(DEFAULT_OUTPUT/'policy_parity_validation.json',{'original_functions':sorted(names),'interpolation_exact_match':True,'filter_and_selection_exact_match_all_five_modes':True,'test_observations':indices,'candidate_count':200})
    print('PASS: interpolation, filter and aggregation match original PUN exactly')
if __name__=='__main__':main()
