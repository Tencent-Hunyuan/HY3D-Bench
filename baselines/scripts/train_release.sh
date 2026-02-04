export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
NET_TYPE="high" 
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0
fi
export NCCL_DEBUG=WARN
export NCCL_ALGO=ring


num_gpu_per_node=$1
config=$2
output_dir=$3
node_num=$4
node_rank=$5
master_ip=$6

ps -ef|grep main_release.py | awk '{printf $2" "}' | xargs kill -9

echo "--- Script Arguments ---"
echo "node_num: $node_num"
echo "node_rank: $node_rank"
echo "master_ip: $master_ip"
echo "config: $config"
echo "output_dir: $output_dir"
echo "----------------------"

if test -d "$output_dir"; then
    cp $config $output_dir
else
    mkdir -p "$output_dir"
    cp $config $output_dir
fi

NODE_RANK=$node_rank \
HF_HUB_OFFLINE=0 \
MASTER_PORT=16699 \
MASTER_ADDR=$master_ip \
NCCL_SOCKET_IFNAME=bond1 \
NCCL_IB_GID_INDEX=3 \
NCCL_NVLS_ENABLE=0 \
python3 baselines/main_release.py \
    --num_nodes $node_num \
    --num_gpus $num_gpu_per_node \
    --config $config \
    --output_dir $output_dir \
    --deepspeed
