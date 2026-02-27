"""
VAE Reconstruction Testing Script

This script evaluates VAE reconstruction quality across different token lengths.
For each token length configuration:
  - Encodes surface data into latent space
  - Decodes back to 3D meshes
  - Computes latent distribution statistics (mean, std)
  - Optionally visualizes reconstructed meshes
  - Saves scale factors for normalization

The scale factor (1/std) is critical for normalizing latent codes during training.
"""

from path import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import json
import copy
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from baselines.core.utils import get_config_from_file, instantiate_from_config
from baselines.core.pipelines import export_to_trimesh

# ============================================================================
# Configuration Loading
# ============================================================================
config_path = 'baselines/configs/debug_dataloader.yaml'
full_config = get_config_from_file(config_path)

# Extract VAE architecture configuration
vae_config = full_config.first_stage_config

# Extract dataset configuration and instantiate dataset
dataset_config = full_config.dataset
dataset = instantiate_from_config(dataset_config)

# ============================================================================
# Dataloader Initialization
# ============================================================================
train_loader = dataset.train_dataloader()
# Test settings
test_sample = 10   # Number of samples to process for statistics
vis_sample = 0     # Number of samples to visualize as meshes (0 to disable)

# ============================================================================
# Output Directory Setup
# ============================================================================
vae_recon_save_path = Path('baselines/scripts_prepare/vae_recon_results')
vae_recon_save_path.makedirs_p()
# Save VAE configuration for reproducibility
vae_config_dict = OmegaConf.to_container(vae_config, resolve=True)
with open(vae_recon_save_path / 'vae_config.json', 'w') as f:
    json.dump(vae_config_dict, f, indent=4)

# ============================================================================
# Main Evaluation Loop: Test Different Token Lengths
# ============================================================================
for token_len in [512, 2048, 4096]:
    print(f'token_len: {token_len}')
    # Create token-length-specific VAE configuration
    tmp_config = copy.deepcopy(vae_config)
    tmp_config.params.num_latents = token_len
    tmp_config.params.pc_size = token_len * 20

    # Initialize VAE model with current configuration
    vae_model = instantiate_from_config(vae_config)
    vae_model = vae_model.to('cuda')
    vae_model.eval()
    vae_model.requires_grad_(False)

    # Prepare output directory for current token length
    save_root = vae_recon_save_path / f'token_len_{token_len}'
    save_root.makedirs_p()

    # Storage for latent statistics across all samples
    all_latents = []
    all_mean_pred = []
    all_std_pred = []
    # ========================================================================
    # Process Batches
    # ========================================================================
    with torch.no_grad():
        bar = tqdm(range(test_sample), desc=f'token_len: {token_len}')
        tmp_count = 0
        
        for data_batch in train_loader:
            tmp_count += 1
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Encode surface to latent space
                # Returns: sampled latents z and posterior distribution q(z|x)
                latents, posterior = vae_model.encode(data_batch['surface'].cuda(), 
                                        sample_posterior=True, return_posterior=True)

                # Optional: Visualize reconstructed meshes for first few samples
                if tmp_count <= vis_sample:
                    # Decode latents back to surface representation
                    for b_idx in range(latents.shape[0]):
                        latents = vae_model.decode(latents[[b_idx]])
                        # Convert to explicit mesh via marching cubes
                        mesh_out = vae_model.latents2mesh(
                            latents[[b_idx]],
                            octree_depth=8,
                            bounds=[-1.01, -1.01, -1.01, 1.01, 1.01, 1.01],
                            mc_level=0.0,
                        )[0]

                        # Export mesh for visualization
                        mesh_out = export_to_trimesh(mesh_out)
                        mesh_out.export(save_root / f'{tmp_count}.glb')
                # Collect latent statistics for scale factor computation
                all_latents.append(latents.flatten().cpu())
                all_mean_pred.append(posterior.mean.flatten().cpu())
                all_std_pred.append(posterior.std.flatten().cpu())
            bar.update(1)
            if tmp_count >= test_sample:
                break
    # ========================================================================
    # Compute Latent Distribution Statistics
    # ========================================================================
    all_latents = torch.stack(all_latents, dim=0)
    all_mean_pred = torch.stack(all_mean_pred, dim=0)
    all_std_pred = torch.stack(all_std_pred, dim=0)
    print(all_latents.shape)
    print(all_mean_pred.shape)
    print(all_std_pred.shape)
    # Compute empirical statistics from sampled latents
    mean_latent = all_latents.mean().item()
    std_latent = all_latents.std().item()

    # Compute statistics of the posterior predictions
    mean_pred = all_mean_pred.mean().item()
    mean_pred_std = all_mean_pred.std().item()
    std_pred = all_std_pred.mean().item()
    # Display results
    print(f'\nLatent Distribution Statistics:')
    print(f'  Sampled latents - mean: {mean_latent:.7f}, std: {std_latent:.7f}')
    print(f'  Scale factor (1/std):   {1/std_latent:.16f}')
    print(f'\nPosterior Predictions:')
    print(f'  Mean of predicted means: {mean_pred:.7f} (std: {mean_pred_std:.7f})')
    print(f'  Mean of predicted stds:  {std_pred:.7f}')
    print(f'  Alt scale factor:        {1/std_pred:.16f}')

    # Save scale factor to file
    # We use std from sampled latents as it captures both the predicted std
    # and the variance in predicted means across the dataset
    with open(save_root / 'scale_factor_shape.txt', 'a') as f:
        f.write(
            f'token_len: {token_len}, ' + \
            f'mean_latent: {mean_latent:.7f}, ' + \
            f'std_latent: {std_latent:.7f}, ' + \
            f'scale_factor_shape: {1/std_latent:.16f}\n'
        )


