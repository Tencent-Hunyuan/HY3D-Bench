# Hunyuan3D-2.1-Shape-Small

## Introduction

This is the baseline implementation model for evaluating the effectiveness of full-level data in HY3D-Bench.

The baseline codes are heavily based on the [Hunyuan3D-2.1-Shape Model](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).

## Install Requirements
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

```

## Data Preparation

1. Download the HY3D-Bench full-level dataset from [HuggingFace](https://huggingface.co/tencent/HY3D-Bench)

2. Organize the data with following structure
```yaml
dataset_root
├── full
│   ├── test
│   │   ├──images
│   │   ├──sample_points
│   │   └──water_tight_meshes
│   ├── train
│   └── val
└── part # this is not necessary for baseline model training
    ├── images
    └── water_tight_meshes
```

3. Run scripts 1-2 in `baselines/scripts_prepare` to preprocess the data

4. Update the dataset paths (`dataset_root/full`) in the configuration files

## Model Preparation

1. Download the VAE model weights from [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)

2. Run script 3 in `baselines/scripts_prepare` to calculate the VAE z-scale factor

3. Update the VAE model weights path in the configuration file and fill in the correct z-scale factor

## Training

### Single Node (8 GPUs)
```bash
cd baselines

bash scripts/train_release.sh 8 \
    "/path/to/config" \
    "/path/to/checkpoint" \
    1 \
    0 \
    127.0.0.1
```

### Multi-Node (8 GPUs per node)
```bash
cd baselines

bash scripts/train_release.sh 8 \
    "/path/to/config" \
    "/path/to/checkpoint" \
    <node_num> \
    <node_rank> \
    <master_ip>
```

**Parameters:**
- `8`: Number of GPUs per node
- `/path/to/config`: Path to configuration file
- `/path/to/checkpoint`: Path to save checkpoints
- `<node_num>`: Total number of nodes
- `<node_rank>`: Current node rank (0-indexed)
- `<master_ip>`: IP address of the master node