"""Stage 1/2 qualitative previews: GT sheets and interactive camera coverage."""
import argparse
import numpy as np
import plotly.graph_objects as go
from bn3d_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=p.parse_args();out=args.output
    geometry=np.load(out/'canonical_geometry.npz');v=geometry['vertices'];tri=geometry['triangles']
    poses=np.array([f['transform_matrix'] for f in read_json(out/'candidate_poses.json')['frames']]);test=np.array([f['transform_matrix'] for f in read_json(out/'test_poses.json')['frames']])
    for profile in PROFILES:
        root=out/profile;avs=read_json(root/'avs20/indices.json');random_ids=read_json(root/'random20/indices.json')
        fig=go.Figure()
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],color='tan',name='GT geometry (untextured)',showlegend=True,opacity=.95))
        for name,T,ids,color,size in [('Candidates (200)',poses,list(range(200)),'lightgray',3),('AVS (20, numbered by selection)',poses[avs],avs,'crimson',6),('Random (20)',poses[random_ids],random_ids,'royalblue',5),('Test (40)',test,list(range(40)),'seagreen',4)]:
            c=T[:,:3,3]
            fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='markers',marker=dict(color=color,size=size),name=name,text=[f'order {j+1}, ID {i}' for j,i in enumerate(ids)],hoverinfo='text+name'))
        c=poses[avs,:3,3];fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='lines',line=dict(color='crimson',width=2),name='Selection order (not a flight path)'))
        fig.update_layout(title=f'{profile}: canonical GT geometry and camera coverage',scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z (up)'),legend=dict(x=0,y=1),margin=dict(l=0,r=0,b=0,t=45))
        fig.write_html(root/'camera_coverage.html',include_plotlyjs=True,full_html=True)
    # Side-by-side matched GT views; these are source geometry renders, not NeRF results.
    ids=[0,40,80,120,160]
    paths=[];labels=[]
    for i in ids:
        for profile in PROFILES:
            paths.append(out/profile/'candidates'/f'{i:04d}.png');labels.append(f'{profile} ID {i}')
    contact_sheet(paths,labels,out/'gt_profile_comparison.png',columns=2,tile=384)
    write_json(out/'visualization_manifest.json',{'gt_comparison':'gt_profile_comparison.png','coverage_html':[f'{p}/camera_coverage.html' for p in PROFILES],'note':'Stage 1/2 only: GT renders and selection/camera coverage; no reconstructed model yet.'})
    print('Saved interactive coverage and GT comparisons')
if __name__=='__main__':main()
