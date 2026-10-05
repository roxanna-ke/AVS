# BN3D building1 through PUN

Submit from the login node with
`bash /mnt/scratch/imos-students/ke/submit_bn3d_pun_native.sh`.
This starts a dedicated GPU pod that runs
`/mnt/scratch/imos-students/ke/run_bn3d_pun_native.sh` with
`/mnt/home/conda-envs/pun`. The output root is
`/mnt/scratch/imos-students/ke/data/bn3d_pun_native/building1`. Each stage
refuses to overwrite a completed output and can be rerun individually.

The source `House.obj` is untouched. `bn3d_pun_native.py prepare` writes a
derived OBJ whose imported Blender vertices equal the previous validated BN3D
canonical coordinates. UVs, material names and texture links are retained.
The only non-lighting Blender adaptation is assigning the same positive
`inst_id` to all eight meshes imported from the BN3D OBJ; PUN's ShapeNet code
labels only one imported mesh, which leaves most BN3D views with an empty mask.

`capture --lighting original` keeps Blender 3.6's startup view transform and
PUN's original world state, including the second renderer's inherited world.
`capture --lighting bright` uses Standard color management, the gray HDRI at
strength 1, and fixed key/fill sun lights in both original PUN renderer classes.
Both branches keep the original spherical candidate sampler, HEALPix
interpolation, 20 online UPNet observations, and `shapenet_eval` poses. The
bright branch changes appearance only, not camera poses or intrinsics.

The second Blender renderer runs in separate batches of 10 images because the
embedded Blender process crashed after a long sequence of `bpycv` renders in
integration testing. Each batch initializes the same PUN world/camera state;
it validates this against the state recorded at AVS completion. Existing
nonempty images are reused on retry, and `finalize` checks every output.

The two renderers intentionally have different intrinsics: first renderer
`fx=525` at 512 px; second renderer uses PUN's 30 degree FOV (`fx≈955`). At
radius 2.732 the latter can crop the house in some views. This difference is
kept for fidelity to PUN; inspect the saved previews before launching the full
run. Original PUN ShapeNetScene uses 10,000 Cycles samples by default and may
take substantial time for its 80 renders per lighting branch. `--samples N`
on `capture` is a debugging override only and must be omitted for the stated
original configuration.

The `runs/<lighting>/<policy>/metrics.json` files include PUN's full-image
batch metric calls and foreground-mask extensions. `summary.csv` has eight
rows. Floating-point NeRF predictions are retained alongside PNG previews.
