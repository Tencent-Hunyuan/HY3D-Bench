# Multi-GPU Inference

## Files

- `infe.py`: the main inference script
- `run_multi_gpu_infer.sh`: startup infer.py
- `README.md`
- `README-ch.md`


## Usage

### 1: Using startup scripts（recommend）

```bash
# using 2 GPUs and provide data folder
bash run_multi_gpu_infer.sh 2 --data-root "/path/to/root" 

# using 4 GPUs and provide a data json
bash run_multi_gpu_infer.sh 4 --data-file "/path/to/file.json" 

# using custom parameters
bash run_multi_gpu_inference.sh 2 \
    "/path/to/conf" \
    "/path/to/ckpt" \
    --data-file "/path/to/file.json" \
    --output-dir "/path/to/output" \
    --log-file "/path/to/log"
```

### 2: Using python scripts directly

#### single GPU
```bash
python baselines/scripts_infer/infer.py \
    --config /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint \
    --data_file /path/to/data.json \
    --output_dir ./results \
    --log_file inference.log
```

#### multi GPU
```bash
# using torchrun
torchrun --nproc_per_node=4 baselines/scripts_infer/infer.py \
    --config /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint \
    --data_file /path/to/data.json \
    --output_dir ./results \
    --log_file inference.log
```

## Python Parameters

### Required
- `--config`: path to config（YAML file）, use the training config here is ok
- `--ckpt_path`: path to checkpoint


### Optional
- `--data_root`: path to data folder 
- `--data_file`: path to data file（JSON file）
- `--output_dir`: output dir（default: ./inference_results）
- `--log_file`: log dir（default: inference.log）
- `--num_workers`: dataloader num workers（default: 4）
- `--num_steps`: inference steps（default: 50）

## Json Format

The data file should be in JSON format，with a `file_list` keyword：

```json
{
    "file_list": [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg"
    ]
}
```

## Output Results

- output 3d geometry：`{image_name}.glb`
- all outputs are saved in output_dir
- all logs are in log_file

