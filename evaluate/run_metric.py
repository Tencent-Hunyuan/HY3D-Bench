import trimesh
import os
import torch
import json
import fpsample
import sys
import numpy as np
from PIL import Image
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{base_dir}')
from ulip_score import UlipScore
from uni3d_score import Uni3DScore


class MetricMesh(object):
    def __init__(self,
                normalize=True,
                device=torch.device("cuda"),
                pretrained_clip_model_path="./checkpoints/open_clip_pytorch_model.bin",
                ulip_ckpt_path="./checkpoints/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt",
                uni3d_ckpt_path="./checkpoints/model.pt"):
        '''
        initialize ulip and uni3d model
        normalize: whether to normalize the mesh to fit in unit sphere
        device: torch device
        pretrained_clip_model_path: path to pretrained clip model
        ulip_ckpt_path: path to ulip checkpoint
        uni3d_ckpt_path: path to uni3d checkpoint
        '''
        # load ulip and uni3d model
        self.ulip_model = UlipScore(ckpt_path=ulip_ckpt_path)
        print("Loading ulip models successily...")
        self.uni3d_model = Uni3DScore(pretrained_clip_model_path=pretrained_clip_model_path, ckpt_path=uni3d_ckpt_path)
        print("Loading uni3d models successily...")
        self.normalize = normalize
        self.device = device

    def normalizemesh(self, scene_mesh):
        '''
        normalize mesh to fit in unit sphere
        scene_mesh: trimesh mesh
        '''
        bbox = scene_mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max()
        scale = 1 / scale * 2 * 0.98

        scene_mesh.apply_translation(-center)
        scene_mesh.apply_scale(scale)
        return scene_mesh, center, scale

    def compute_ulip(self, pc, images):
        '''
        compute ulip score between point cloud and images
        pc: (1, N, 3) torch tensor
        images: list of PIL images
        '''
        # get ulilp score
        ulip_score = self.ulip_model.sim_img(pc, images).item()
        return ulip_score

    def compute_uni3d(self, pc, images):
        '''
        compute uni3d score between point cloud and images
        pc: (1, N, 3) torch tensor
        images: list of PIL images
        '''
        # get uni3d score
        uni3d_score = self.uni3d_model.sim_img(pc, images).item()
        return uni3d_score

    def compute_metric(self, image_paths, mesh_paths, method_name, save_dir):
        '''
        compute ulip and uni3d score for a list of image and mesh paths
        image_paths: list of image paths
        mesh_paths: list of mesh paths
        method_name: name of the method
        save_dir: directory to save the result
        '''
        result = {}
        
        save_path = os.path.join(save_dir, f"{method_name}_metrics.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ulip_scores = []
        uni3d_scores = []
        for idx, image_path in enumerate(image_paths):
            # get uid
            uid = image_path.split("/")[-2]
            mesh_path = mesh_paths[idx]
            if not os.path.isfile(mesh_path) or not os.path.isfile(image_path):
                continue
            
            # get image
            image = Image.open(image_path).convert("RGB")

            # sample dense point cloud from mesh
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            if self.normalize:
                mesh, _, _ = self.normalizemesh(mesh)
            samples, _ = trimesh.sample.sample_surface(mesh, 300000)

            # farthest point sampling to 8192 points
            idx = fpsample.bucket_fps_kdline_sampling(samples, 8192, h=5)
            samples = samples[idx]
            pc = torch.from_numpy(samples).float().unsqueeze(0)

            # compute ulip and uni3d score
            ulip_score = self.compute_ulip(pc, [image])
            uni3d_score = self.compute_uni3d(pc, [image])

            ulip_scores.append(ulip_score)
            uni3d_scores.append(uni3d_score)

            result[uid] = {
                "ulip_score": ulip_score,
                "uni3d_score": uni3d_score
            }
        
        # compute average score
        result["average"] = {
            "ulip_score": sum(ulip_scores) / len(ulip_scores),
            "uni3d_score": sum(uni3d_scores) / len(uni3d_scores)
        }
        # save result to json file
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

if __name__=="__main__":

    # define output paths and method names
    output_dir = "./output_1024"
    method_names = ['michelangelo_0107', 'trellis_0107', 'ours', 'Hunyuan3D-2.1', 'CraftsMan3D']

    # get all images
    images = sorted(os.listdir(os.path.join(output_dir, "image")))
    print(f"Total {len(images)} images for evaluation.")
    save_dir = "./result/"

    # initialize metric
    metric = MetricMesh(
            normalize=True,
            device=torch.device("cuda"),
            pretrained_clip_model_path="./checkpoints/open_clip_pytorch_model.bin",
            ulip_ckpt_path="./checkpoints/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt",
            uni3d_ckpt_path="./checkpoints/model.pt?download=true",
        )
    for i in range(len(method_names)):
        print(f"Evaluating method: {method_names[i]}")

        # prepare image paths and corresponding mesh paths
        image_paths = []
        mesh_paths = []
        for img in images:
            uid = img.split("/")[-1].split(".")[0]
            image_path = os.path.join(output_dir, "image", uid, "000.png")
            mesh_path = os.path.join(output_dir, method_names[i], f"{uid}/000.glb")
            image_paths.append(image_path)
            mesh_paths.append(mesh_path)

        # compute metric
        metric.compute_metric(image_paths, mesh_paths, method_names[i], save_dir=save_dir)
    print("Evaluation done.")

    