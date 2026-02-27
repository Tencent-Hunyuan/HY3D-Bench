# HY3D-Synthetic Dataset

## Overview
HY3D-Synthetic is a large-scale AIGC-driven 3D dataset generated with **HY3D-3.0**, containing over **125,000** synthetic 3D assets across diverse e-commerce product categories.


## Data Structure

The dataset is organized into two main directories:

```
synthetic/
├── glb/         # 3D assets in GLB format             
│   ├── chunk1/  # A single class folder
│   ...       
│   └── chunkn/  # A single class folder
└── img/         # the original generated image used to generate 3D assets   
    
```
Each `chunk` folder corresponds to a single product category, identified by a UUID-based folder name. A mapping file (`uuid_mapping_synthetic.json`) is provided to resolve the UUID folder names back to their original category labels.