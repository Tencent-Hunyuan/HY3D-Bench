# HY3D-Full Dataset

## Overview
This is a ready-to-use training dataset for 3D generative DiT models, comprising over 250K curated training samples. The dataset provides watertight meshes, pre-sampled surface point clouds, and pre-rendered multi-view images with corresponding camera parameters and normal maps.

## Data Structure

The dataset is organized into three main directories:

```
full/
├── train/                
│   ├── images/              # Multi-view rendered images (NPZ archives)
│   ├── sample_points/       # Sampled surface point clouds (TAR archives)
│   └── water_tight_meshes/  # Watertight meshes
├── test/  
└── val/            
    
```

---

## 1. Rendered Images
Each object is rendered from multiple viewpoints. The archive includes both RGB images and camera-space normal maps.

### Data Format

- Storage: NPZ archive (one per object)
- Contents: RGB images (000.png, 001.png, ...) and normal maps (000_camnorm.png, 001_camnorm.png, ...)
- Camera parameters: `transforms.json` (stored in the same NPZ)

### Loading Code
```python
import numpy as np
import io
import os
from PIL import Image

# Load npz file
npz_path = "path/to/object.npz"
save_dir = "path/to/output"
data = np.load(npz_path, allow_pickle=True)

# Extract and save images
for i, img_key in enumerate(['000.png', '001.png']):
    if img_key in data:
        img_bytes = data[img_key]
        # Load image bytes into PIL Image
        image = Image.open(io.BytesIO(img_bytes.tobytes()))
        print(f"Image {img_key}: size={image.size}, mode={image.mode}")

        # Save image
        img_save_path = os.path.join(save_dir, f"image_{i:03d}.webp")
        image.save(img_save_path)
```
also see `load_render` function in `baselines/core/data/dit_dataset.py`

## 2. Sampled Points
Surface points sampled from the watertight meshes, including both random surface samples and sharp-edge surface samples.

### Data Format
For each mesh, we sample approximately 250K random surface points and 250K sharp-edge surface points (each with normals, stored as Nx6 arrays).
The point clouds are organized in a chunked layout optimized for training I/O:

Each sample's ~500K points are partitioned into 488 groups of 1,024 points each.
Every chunk aggregates 200 samples. For each of the 488 groups, the corresponding `.npy` files from all 200 samples are packed into a single `.tar` file.
As a result, each chunk contains 488 `.tar` files (one per group), and each `.tar` file holds 200 `.npy` files (one per sample).

```
## for a single sharpedge_surface tar, it contains
chunk_xxxx_sharpedge_surface_yyy.tar
├── chunk_xxxx_uuid1.sharpedge_surface.npy
├── chunk_xxxx_uuid2.sharpedge_surface.npy
...
└── chunk_xxxx_uuid200.sharpedge_surface.npy
## for a single surface tar, it contains
chunk_xxxx_surface_zzz.tar
├── chunk_xxxx_uuid1.random_surface.npy
├── chunk_xxxx_uuid2.random_surface.npy
...
└── chunk_xxxx_uuid200.random_surface.npy
```

To use the dataloader provided in `baselines/core/data/dit_dataset.py`, first run the preprocessing scripts in `baselines/scripts_prepare`

### Converting Code
The following script converts the chunk-level `.tar` files (which interleave data from multiple samples) into per-sample `.tar` or `.npy` files.
```python
from path import Path
import tarfile
from tqdm import tqdm
import json
import shutil
import numpy as np

SOURCE_ROOT = Path('path_to_source_chunk')
tar_files = SOURCE_ROOT.glob('*.tar')
tar_files = [file for file in tar_files if 'surface' in file.name]
tar_files.sort()
TARGET_ROOT = Path('path_to_target_folder')
TARGET_ROOT.makedirs_p()
TARGET_FILE = 'npy' # 'npy' or 'tar'
TMP_ROOT = TARGET_ROOT / 'group_tmp'
TMP_ROOT.makedirs_p()

### step1 load all uids from metadata tar
meta_tar = SOURCE_ROOT.glob('*metadata.tar')
meta_tar = meta_tar[0]
uid_list = []  ## eg. ['ff3bce683a70342989a97de2811edc8fe1c01ef3f5bafd6778b625d97c4d3bb1']
uid_key_list = []  ## eg. ['chunk_1263_ff3bce683a70342989a97de2811edc8fe1c01ef3f5bafd6778b625d97c4d3bb1']
with tarfile.open(meta_tar, 'r') as tf:
    for member in tf.getmembers():
        if member.isfile() and member.name.endswith('.metadata.json'):
            with tf.extractfile(member) as f:
                js = json.load(f)
                uid_list.append(js['uid'])
                uid_key_list.append(js['key'])

### step2 extract npy files from source tar files
bar = tqdm(tar_files, desc='Processing source tar files')
for tar_file in bar:
    bar.set_postfix_str(f'Processing {tar_file.name}')
    with tarfile.open(tar_file, 'r') as tf:
        tmp_root = TMP_ROOT / tar_file.stem
        tmp_root.makedirs_p()
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith('.npy'):                
                npy_file = tmp_root / member.name
                with open(npy_file, 'wb') as f:
                    f.write(tf.extractfile(member).read())
bar.close()

### step3 reconstruct per sample file
all_groups = TMP_ROOT.glob('*')
all_groups = [group for group in all_groups if group.is_dir()]
all_groups.sort()
if TARGET_FILE == 'tar':
    bar = tqdm(uid_key_list, desc='Reconstructing tar files')
    for uid_key in bar:
        bar.set_postfix_str(f'Processing {uid_key}')
        uid_root = TARGET_ROOT / uid_key
        uid_root.makedirs_p()
        for group in all_groups:
            is_sharpedge = 'sharpedge' in group.name
            group_id = group.name.split('_')[-1]
            if is_sharpedge:
                suffix = '.sharpedge_surface.npy'
            else:
                suffix = '.random_surface.npy'
            tmp_file = group / f'{uid_key}{suffix}'
            ## note that we should add group_id here
            tgt_file = uid_root / f'{uid_key}_{group_id}{suffix}'
            if not tmp_file.exists():
                print(f'{tmp_file} not found, skip')
                continue
            shutil.move(tmp_file, tgt_file)
        uid_tar = TARGET_ROOT / f'{uid_key}.tar'
        with tarfile.open(uid_tar, 'w') as tf:
            for member in uid_root.glob('*.npy'):
                tf.add(member, arcname=member.name)
        shutil.rmtree(uid_root)
    shutil.rmtree(TMP_ROOT)
elif TARGET_FILE == 'npy':
    bar = tqdm(uid_key_list, desc='Reconstructing npy files')
    for uid_key in bar:
        bar.set_postfix_str(f'Processing {uid_key}')
        uid_npy_surface = []
        uid_npy_sharpedge = []
        for group in all_groups:
            is_sharpedge = 'sharpedge' in group.name
            if is_sharpedge:
                suffix = '.sharpedge_surface.npy'
            else:
                suffix = '.random_surface.npy'
            tmp_file = group / f'{uid_key}{suffix}'
            if not tmp_file.exists():
                print(f'{tmp_file} not found, skip')
                continue
            tmp_points = np.load(tmp_file)
            if is_sharpedge:
                uid_npy_sharpedge.append(tmp_points)
            else:
                uid_npy_surface.append(tmp_points)
        uid_npy_surface = np.concatenate(uid_npy_surface, axis=0)
        uid_npy_sharpedge = np.concatenate(uid_npy_sharpedge, axis=0)
        np.save(TARGET_ROOT / f'{uid_key}.random_surface.npy', uid_npy_surface)
        np.save(TARGET_ROOT / f'{uid_key}.sharpedge_surface.npy', uid_npy_sharpedge)
    shutil.rmtree(TMP_ROOT)
else:
    raise ValueError(f'Invalid TARGET_FILE: {TARGET_FILE}')

```