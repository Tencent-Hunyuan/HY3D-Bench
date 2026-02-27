# HY3D-Part Dataset

## Overview

This is a large-scale **Part-level 3D Object Dataset** containing **240K** objects across diverse categories. The dataset provides comprehensive data including rendered images, segmentation masks, watertight meshes, and original meshes processing scripts.

## Dataset Scale

- **Number of Objects**: 240,000+
- **Category Coverage**: Diverse object categories (see `part_meta_data.csv`)
- **Annotations**: Part-level segmentation

## Data Structure

The dataset is organized into two main directories:

```
part/
├── images/                # Image data (NPZ archives)
│   ├── Rendered images    # Multi-view rendered images (WebP)
│   ├── Segmentation masks # Part-level masks (NumPy arrays)
│   ├── Camera parameters  # transforms.json
│   └── Original mesh processing  # split_mesh.json
│
└── meshes/                # Mesh data (NPZ archives)
    └── Watertight part and whole meshes    # Watertight processed meshes
    
```

---

## 1. Rendered Images

Multi-view rendered images.

### Data Format

- Storage: NPZ archive (one per object)
- Image format: WebP (keys: `000.webp`, `001.webp`, ...)
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
for i, img_key in enumerate(['000.webp', '001.webp']):
    if img_key in data:
        img_bytes = data[img_key]
        # Load image bytes into PIL Image
        image = Image.open(io.BytesIO(img_bytes.tobytes()))
        print(f"Image {img_key}: size={image.size}, mode={image.mode}")
        
        # Save image
        img_save_path = os.path.join(save_dir, f"image_{i:03d}.webp")
        image.save(img_save_path)
```

---

## 2. Segmentation Masks

Part-level segmentation masks corresponding to rendered images.

### Data Format

- Storage: NPZ archive (same file as rendered images)
- Mask format: NumPy array (keys: `000_mask.npy`, `001_mask.npy`, ...)
- Pixel values: 
  - Positive integers (0, 1, 2, ...): Part IDs
  - -1: Background

### Loading Code

```python
import numpy as np
import cv2

def colorize_mask(mask):
    """Convert mask to colorized image for visualization."""
    unique_values = np.unique(mask)
    
    # Create random color mapping for each part ID
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
    
    # Create color image
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for val in unique_values:
        if val == -1:
            color_mask[mask == val] = [255, 255, 255]  # Background is white
        else:
            color_mask[mask == val] = colors[val % 256]
    
    return color_mask

# Load and process masks
data = np.load(npz_path, allow_pickle=True)
for i, mask_key in enumerate(['000_mask.npy', '001_mask.npy']):
    if mask_key in data:
        mask = data[mask_key]
        
        # Save colorized mask for visualization
        color_mask = colorize_mask(mask)
        cv2.imwrite(f"mask_{i:03d}_color.png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        
        # Save raw mask as npy
        np.save(f"mask_{i:03d}.npy", mask)
```

---

## 3. Watertight Mesh

Watertight processed meshes ensuring geometric closure, suitable for voxelization, SDF computation, and other tasks.

### Data Format

- Format: NPZ (compressed storage)
- Contains part and whole watertight meshes

### Loading Code

```python
import numpy as np
import io
import os
import trimesh

# Load npz file
npz_path = "path/to/mesh.npz"
output_dir = "path/to/output"
os.makedirs(output_dir, exist_ok=True)

data = np.load(npz_path)
print(f"Keys: {list(data.keys())}")

# Extract and save each mesh as PLY file
for key in data.keys():
    # Convert bytes array to trimesh object
    ply_bytes = data[key].tobytes()
    mesh = trimesh.load(io.BytesIO(ply_bytes), file_type='ply')
    print(f"  - {key}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    
    # Save as PLY file
    save_path = os.path.join(output_dir, key)
    mesh.export(save_path)
```

---

## 4. Original Mesh Processing

Scripts for processing original meshes into part-aware meshes.

### Description

This section provides processing scripts to transform raw mesh data into part-level meshes. The script reads `split_mesh.json` from the render NPZ file, which contains part clustering information, then merges mesh components accordingly.

### Processing Code

```python
import os
import json
import numpy as np
import trimesh

def convert_part_data(render_npz_path, texture_mesh_path, output_dir):
    """
    Read split_mesh.json from render NPZ file and merge mesh components by part.
    
    Args:
        render_npz_path: Path to render NPZ file containing split_mesh.json
        texture_mesh_path: Path to original texture mesh (glb)
        output_dir: Output directory for exported part meshes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load split_mesh.json from NPZ
    data = np.load(render_npz_path, allow_pickle=True)
    if 'split_mesh.json' not in data:
        print(f"split_mesh.json not found in {render_npz_path}")
        return
    
    result_dict = json.loads(data['split_mesh.json'].tobytes().decode('utf-8'))
    part_id_to_name = result_dict["part_id_to_name"]
    valid_clusters = result_dict["valid_clusters"]
    
    # Load original texture mesh
    texture_mesh = trimesh.load(texture_mesh_path, process=False)
    
    if not isinstance(texture_mesh, trimesh.Scene):
        print("Warning: texture_mesh is not a Scene, cannot split by parts")
        return
    
    # Merge and export each part cluster
    for cluster_name, cluster_info in valid_clusters.items():
        cluster_meshes = []
        
        for part_id in cluster_info["part_ids"]:
            part_name = part_id_to_name[part_id]
            if part_name not in texture_mesh.geometry:
                continue
            
            # Find all node instances referencing this geometry and apply transforms
            for node_name in texture_mesh.graph.nodes_geometry:
                if texture_mesh.graph[node_name][1] != part_name:
                    continue
                
                try:
                    transform = texture_mesh.graph.get_transform(node_name)
                except Exception:
                    transform = texture_mesh.graph[node_name][0]
                
                geometry = texture_mesh.geometry[part_name]
                if isinstance(geometry, trimesh.Trimesh):
                    mesh_copy = geometry.copy()
                    mesh_copy.apply_transform(transform)
                    cluster_meshes.append(mesh_copy)
        
        if not cluster_meshes:
            continue
        
        # Concatenate and export
        merged = trimesh.util.concatenate(cluster_meshes)
        merged.export(os.path.join(output_dir, f"{cluster_name}.glb"))
        print(f"Exported {cluster_name}: vertices={len(merged.vertices)}, faces={len(merged.faces)}")
```

## License

Please refer to the original license information of each data source (see `*_license_distribution.txt` files).

---

## Citation

If you use this dataset, please cite:

```bibtex
# TODO: Add citation
```
