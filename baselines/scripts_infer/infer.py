import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from path import Path
from baselines.core.utils import get_config_from_file, instantiate_from_config
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import json
import argparse
import os
import time
from tqdm import tqdm
import logging
from typing import List, Dict, Any

class InferDataset(data.Dataset):
    def __init__(self, data_file=None, data_root=None):
        assert data_file is not None or data_root is not None, "data_file or data_root must be provided"
        self.data_file = data_file
        self.data_root = data_root
        self.data = self.load_data()
    
    def load_data(self):
        if self.data_file is not None:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            return data['file_list']
        else:
            data_root = Path(self.data_root)
            data_list = list(data_root.glob('*/000.png'))
            data_list.sort()
            return data_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def setup_logging(rank: int, log_file: str = None):
    """设置日志"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

def setup_distributed():
    """初始化分布式环境（仅用于数据并行）"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # 只初始化进程组，不进行模型并行
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_model(model_cfg, ckpt_path, device):
    """加载模型（单卡推理，不进行模型并行）"""
    ckpt_path = Path(ckpt_path)
    model: pl.LightningModule = instantiate_from_config(model_cfg)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / 'pytorch_model.bin'
    
    model.init_from_ckpt(ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model

@torch.no_grad()
def main(args):
    """主函数 - 数据并行推理"""
    # 设置分布式环境（仅用于数据分片）
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # 设置日志
    setup_logging(rank, args.log_file)
    
    logging.info(f"Starting data-parallel inference on rank {rank}/{world_size}, device: {device}")
    
    # 加载配置
    config = get_config_from_file(args.config)
    logging.info(f"Config loaded from {args.config}")

    # 加载模型（每个进程独立加载，不进行模型并行）
    model = load_model(config.model, args.ckpt_path, device)
    # 获取pipeline
    pipeline = model.pipeline

    # 创建数据集
    dataset = InferDataset(data_file=args.data_file, data_root=args.data_root)
    
    # 使用DistributedSampler进行数据分片
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            drop_last=False
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=1,  # 推理时通常使用batch_size=1
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # 单卡模式不使用sampler
        dataloader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # 计算每个进程实际处理的数据量
    if world_size > 1:
        items_per_rank = len(dataset) // world_size
        if rank < len(dataset) % world_size:
            items_per_rank += 1
        logging.info(f"Dataset loaded: {len(dataset)} total items, {items_per_rank} items for rank {rank}")
    else:
        logging.info(f"Dataset loaded: {len(dataset)} items for single GPU")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 执行推理
    start_time = time.time()

    infer_params = {
        'num_inference_steps':  args.num_steps,
        'output_type':          'trimesh',
        'enable_pbar':          False,
        'guidance_scale':       7.5,
        'dual_guidance':        False,
        'generator':            None,
        'box_v':                1.01,
        'octree_resolution':    384,
        'mc_level':             0,
        'mc_algo':              'dmc',
    }

    processed_count = 0
    for image_path in tqdm(dataloader, desc=f"Rank {rank} inference", disable=rank != 0):
        if isinstance(image_path, list):
            image_path = image_path[0]  # 如果DataLoader返回的是列表
        
        image_path_stem = Path(image_path).stem
        try:
            mesh = pipeline(image_path, **infer_params)[0]
            output_file = f'{args.output_dir}/{image_path_stem}.glb'
            mesh.export(output_file)
            processed_count += 1
            tqdm.write(f"Rank {rank}: Processed {image_path_stem} -> {output_file}")
        except Exception as e:
            logging.error(f"Rank {rank}: Error processing {image_path_stem}: {str(e)}")
            continue
        
    end_time = time.time()
    logging.info(f"Rank {rank}: Inference completed in {end_time - start_time:.2f} seconds, "
                 f"processed {processed_count}/{len(dataset)} items")
    cleanup_distributed()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-GPU 3D Model Inference Script')
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    
    # 可选参数
    parser.add_argument('--data_file', type=str, default=None, help='Path to data file (JSON)')
    parser.add_argument('--data_root', type=str, default=None, help='Path to data folder')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--log_file', type=str, default='inference.log', help='Log file path')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of inference steps')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

    