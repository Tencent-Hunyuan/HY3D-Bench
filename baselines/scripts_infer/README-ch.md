# 多卡数据并行推理脚本使用说明

## 文件说明

- `infe.py`: 主要推理脚本
- `run_multi_gpu_infer.sh`: 启动脚本
- `README.md`
- `README-ch.md`


## 使用方法

### 方法1: 使用启动脚本（推荐）

```bash
# 使用2张GPU进行推理，并且提供数据文件夹
bash run_multi_gpu_inference.sh 2 --data-root "/path/to/root" 

# 使用4张GPU进行推理，并提供数据json文件
bash run_multi_gpu_inference.sh 4 --data-file "/path/to/file.json" 

# 自定义所有参数
bash run_multi_gpu_inference.sh 2 \
    "/path/to/conf" \
    "/path/to/ckpt" \
    --data-file "/path/to/file.json" \
    --output-dir "/path/to/output" \
    --log-file "/path/to/log"
```

### 方法2: 直接使用Python脚本

#### 单卡推理
```bash
python baselines/scripts_infer/infer.py \
    --config /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint \
    --data_file /path/to/data.json \
    --output_dir ./results \
    --log_file inference.log
```

#### 多卡数据并行推理
```bash
# 使用torchrun启动多卡数据并行推理
torchrun --nproc_per_node=4 baselines/scripts_infer/infer.py \
    --config /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint \
    --data_file /path/to/data.json \
    --output_dir ./results \
    --log_file inference.log
```

## Python参数说明

### 必需参数
- `--config`: 配置文件路径（YAML 文件），可以直接使用训练的配置文件
- `--ckpt_path`: checkpoint路径

### 可选参数
- `--data_root`: 数据文件夹的路径
- `--data_file`: 数据列表的路径（JSON 文件）
- `--output_dir`: 输出路径（默认: ./inference_results）
- `--log_file`: 日志路径（默认: inference.log）
- `--num_workers`: dataloader num workers（d默认: 4）
- `--num_steps`: 推理步数（默认: 50）

## 数据格式

数据文件应该是JSON格式，包含一个`file_list`字段：

```json
{
    "file_list": [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg"
    ]
}
```

## 输出结果

- 每个GPU直接保存GLB格式的3D模型文件：`{image_name}.glb`
- 所有结果保存在指定的输出目录中
- 日志文件包含详细的推理过程信息
- 使用DistributedSampler自动分配数据，确保无重复处理

