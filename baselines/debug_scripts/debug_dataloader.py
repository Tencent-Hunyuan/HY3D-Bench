import sys
from path import Path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from baselines.core.utils import get_config_from_file, instantiate_from_config
from baselines.core.pipelines import export_to_trimesh
import torch
from PIL import Image
import matplotlib
import numpy as np
import time
import trimesh

import pdb

def save_points(points, batch_idx, name='points', root='debug_files/dataloader'):
    """
    points: [B, N, 7]
    """
    if points.shape[-1] > 3:
        points = points[..., :3]
    root = Path(root)
    root.makedirs_p()
    for v_idx in range(points.shape[0]):
        points_np = points[v_idx].cpu().numpy()
        points_trimesh = trimesh.PointCloud(points_np)
        point_path = root / f'{batch_idx}_{v_idx}_{name}.ply'
        points_trimesh.export(point_path)

def save_images(images, batch_idx, name='images', root='debug_files/dataloader'):
    """
    images: [B, 3, H, W]
    """
    root = Path(root)
    root.makedirs_p()
    for v_idx in range(images.shape[0]):
        image = images[v_idx].permute(1, 2, 0).cpu().numpy()
        image = (image + 1) / 2
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image_path = root / f'{batch_idx}_{v_idx}_{name}.png'
        image.save(image_path)

@torch.no_grad()
def vae_process(batch, vae_model):
    mesh_outs = []
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            latents = vae_model.encode(batch['surface'].cuda(), sample_posterior=True)
            latents = vae_model.decode(latents)
            # latents = vae_model.z_scale_factor * latents
            for b_idx in range(latents.shape[0]):
                mesh_out = vae_model.latents2mesh(
                    latents[[b_idx]],
                    bounds=[-1.01, -1.01, -1.01, 1.01, 1.01, 1.01],
                    mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=256,
                    mc_algo='dmc',
                    enable_pbar=True
                    )[0]
                # mesh_out = vae_model.decode(
                #     latents[[b_idx]],
                #     octree_depth=8,
                #     bounds=[-1.01, -1.01, -1.01, 1.01, 1.01, 1.01],
                #     mc_level=0.0,
                # )[0]
                mesh_outs.append(mesh_out)
    return mesh_outs

def save_vae_mesh(batch, batch_idx, mesh_outs, root='debug_files/dataloader'):
    root = Path(root)
    root.makedirs_p()
    bsz = batch['image'].shape[0]
    for b_idx in range(bsz):
        stem = batch['__key__'][b_idx]
        mesh_out = mesh_outs[b_idx]
        mesh_out = export_to_trimesh(mesh_out)
        mesh_out.export(root / f'{batch_idx}_{b_idx}_vae_mesh.ply')

config_path = 'baselines/configs/debug_dataloader.yaml'
vis_vae_mesh = True

full_config = get_config_from_file(config_path)

data_config = full_config.dataset
data_config.params.num_workers = 0

data = instantiate_from_config(data_config)
train_loader = data.train_dataloader()
train_loader.dataset.debug_mode = True

vae_model = None
if vis_vae_mesh:
    vae_model = instantiate_from_config(full_config.first_stage_config)
    vae_model.eval()
    vae_model.to('cuda')
    vae_model.requires_grad_(False)

step = 0
print('start batch reading...')
save_path = Path('debug_files/dataloader1')
for batch in train_loader:
    print(step)

    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            print(f'{key:<15}', batch[key].shape, batch[key].min(), batch[key].max())
        else:
            print(f'{key:<15}', type(batch[key]), batch[key])
    
    save_points(batch['surface'], step, 'points', save_path)
    save_images(batch['image'], step, 'image', save_path)
    if vis_vae_mesh:
        mesh_outs = vae_process(batch, vae_model)
        save_vae_mesh(batch, step, mesh_outs, save_path)
    step += 1
    pdb.set_trace()
