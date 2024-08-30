# Helper Scripts to Process Structured3D data

## Contents
This directory contains following helper scripts to process Structured3D(s3d) data so they can be used by F3Loc:
### gather_data_s3d.py
This gathers a part of data from s3d that is useful for F3Loc such as images and camera poses.

WARNING: By default it removes the unused s3d data such as semantic.png, albedo.png etc. Be careful if you wish to keep them.

### downsize_s3d.py
This replaces the images with downsized images of resolution 640x360.

### viewmap_s3d.py
This script serves for 2 purposes:
- generate wall-only floorplan png from s3d annotation_3d.json
- get the camera poses in map (floorplan) coordinate

### get_ray_s3d.py
This generates the ground-truth floorplan depth for all camera poses using the function ray_cast.

### get_desdf_s3d.py
This generates DESDF for floorplans.
NOTE: You don't need DESDF for all scenes. Just test scenes is fine, since DESDFs are only needed for evaluation and the generation takes quite long.

# Usage
To get the s3d data in the form needed by F3Loc and the correct folder structure, please run in order:
- gather_data_s3d.py
- downsize_s3d.py
- viewmap_s3d.py

After this, you will have the following folder structure:
```
├── Structured3D 
│   ├── scene_00000
│       ├── imgs
│           ├── 000.png
│           ├── 001.png
│           ├── ... 
│       ├── annotation_3d.json
│       ├── map.png
│       ├── poses.txt           # s3d camera poses
│       ├── poses_map.txt       # poses in map [x, y, theta, roll, pitch]
│   ├── scene_00001
│   ...
```

If you with to generate the ground-truth floorplan depth or the DESDFs, then run:
- get_ray_s3d.py
- get_desdf_s3d.py

NOTE: Adapt the data path before running the script. Visualization is optional. Look into the script for details.