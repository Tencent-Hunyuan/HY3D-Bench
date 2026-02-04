#!/bin/bash

# 多卡数据并行推理启动脚本
# 使用方法: 
#   使用DATA_FILE: bash run_multi_gpu_inference.sh [GPU数量] --data-file [文件路径]
#   使用DATA_ROOT: bash run_multi_gpu_inference.sh [GPU数量] --data-root [目录路径]

set -e

# 默认参数
NUM_GPUS=${1:-8}
CONFIG_FILE=${2:-"/apdcephfs_jn/share_302245012/keonkhli/release_3d_v1_1/hy3d_baseline_512token_1_ablation_skipconnection/hy3d_baseline_512token_1_ablation_skipconnection.yaml"}
CKPT_PATH=${3:-"/apdcephfs_jn/share_302245012/keonkhli/release_3d_v1_1/hy3d_baseline_512token_1_ablation_skipconnection/ckpt/model_step00800000_rmsnorm_converted.ckpt"}

# 数据源参数（默认为空）
DATA_FILE=""
DATA_ROOT=""
OUTPUT_DIR=""
LOG_FILE="inference.log"

# 解析数据源参数
shift 3  # 跳过前三个位置参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 验证数据源参数
if [ -z "$DATA_FILE" ] && [ -z "$DATA_ROOT" ]; then
    echo "Error: Either --data-file or --data-root must be provided"
    echo "Usage examples:"
    echo "  bash $0 8 --data-file /path/to/file.json --output-dir /path/to/output"
    echo "  bash $0 8 --data-root /path/to/root --output-dir /path/to/output"
    exit 1
fi

if [ -n "$DATA_FILE" ] && [ -n "$DATA_ROOT" ]; then
    echo "Error: Cannot provide both --data-file and --data-root"
    exit 1
fi

# 设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/apdcephfs_jn/share_302245012/keonkhli/release_3d_v1_1/hy3d_baseline_512token_1_ablation_skipconnection/inference_results/model_step00800000_rmsnorm_converted"
fi

# 检查GPU数量
if [ $NUM_GPUS -gt 8 ]; then
    echo "Warning: Maximum 8 GPUs supported, setting to 8"
    NUM_GPUS=8
fi

# 构建数据源参数
DATA_ARGS=""
if [ -n "$DATA_FILE" ]; then
    DATA_ARGS="--data_file $DATA_FILE"
    echo "Using data file: $DATA_FILE"
elif [ -n "$DATA_ROOT" ]; then
    DATA_ARGS="--data_root $DATA_ROOT"
    echo "Using data root: $DATA_ROOT"
fi

echo "Starting data-parallel inference with $NUM_GPUS GPUs"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CKPT_PATH"
echo "Output dir: $OUTPUT_DIR"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# 启动多卡推理
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single GPU inference..."
    python baselines/scripts_infer/infer.py \
        --config $CONFIG_FILE \
        --ckpt_path $CKPT_PATH \
        $DATA_ARGS \
        --output_dir $OUTPUT_DIR \
        --log_file $LOG_FILE \
        --num_workers 4 \
        --num_steps 50
else
    echo "Running data-parallel inference with $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        baselines/scripts_infer/infer.py \
        --config $CONFIG_FILE \
        --ckpt_path $CKPT_PATH \
        $DATA_ARGS \
        --output_dir $OUTPUT_DIR \
        --log_file $LOG_FILE \
        --num_workers 4 \
        --num_steps 50
fi

echo "Inference completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"