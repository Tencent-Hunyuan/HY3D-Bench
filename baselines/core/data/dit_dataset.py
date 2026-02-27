# -*- coding: utf-8 -*-
import glob
import json
import os
import random
import sys
import tarfile
import time
import traceback
import msgpack
import io
from typing import Optional, Union, List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import functional_datapipe
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import trimesh
from webdataset.gopen import gopen

# from baselines.core.data.transforms import build_transforms
from baselines.core.utils.io import npy_loads, npz_loads, json_loads
from baselines.core.data import tariterators
from baselines.core.data import utils
from baselines.core.data.utils import worker_init_fn

import pdb

print(f'==' * 50)
print(f'DataLoader Version: 2024_1210')
print(f'Features:')
print(f'1. Rotate mesh to the nearest 90 degrees of arbitrary camera pose images;')
print(f'2. Close data augmentation for images.')
print(f'==' * 50)

stage1_dict = {
    "elevation_10_000": {'angle': 0, 'direction': [0, 1, 0]},
    "elevation_10_009": {'angle': 270, 'direction': [0, 1, 0]},
    "elevation_10_018": {'angle': 180, 'direction': [0, 1, 0]},
    "elevation_10_027": {'angle': 90, 'direction': [0, 1, 0]},
    "ortho_elevation_90_000": {'angle': 90, 'direction': [0, 0, -1]},
    "ortho_elevation_-90_000": {'angle': -90, 'direction': [0, 0, -1]}
}

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample

def read_msgpack(path):
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read())
    return data

def rotate_image(image, angle):
    # 获取图像尺寸和旋转中心
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算新边界尺寸
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]

    # 旋转图像，边缘填充
    rotated = cv2.warpAffine(image, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def padding(image, mask, center=True, padding_ratio_range=[1.15, 1.15]):
    h, w = image.shape[:2]
    max_side = max(h, w)
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)
    # resize_side = int(max_side * 1.15)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
    start_w = pad_w // 2
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    return newimg, newmask


def sdf_func(sdf_value, sdf_res):
    if sdf_value >= 1 / sdf_res:
        ret = 1
    elif sdf_value <= -1 / sdf_res:
        ret = 0
    else:
        ret = 0.5 + 0.5 * sdf_value / (1 / sdf_res)
    return ret


def tsdf_func(sdf_value, sdf_res):
    '''
    sdf_res = 128
    '''
    if sdf_value >= 1 / sdf_res:
        ret = 1 / sdf_res
    elif sdf_value <= -1 / sdf_res:
        ret = -1 / sdf_res
    else:
        ret = sdf_value
    return ret * sdf_res


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = (
            utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if self.deterministic:
            seed = utils.make_seed(self.worker_seed(), self.epoch)
        else:
            seed = utils.make_seed(
                self.worker_seed(),
                self.epoch,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.urls) - 1)
            yield dict(url=self.urls[index])


def viz_pc(surface, normal, image_input, name):
    image_input = image_input.cpu().numpy()
    image_input = image_input.transpose(1, 2, 0) * 0.5 + 0.5
    image_input = (image_input * 255).astype(np.uint8)
    cv2.imwrite(name + '.png', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
    surface = surface.cpu().numpy()
    normal = normal.cpu().numpy()
    # surface = surface - [1.1,1.,0]
    surface_mesh = trimesh.Trimesh(surface, vertex_colors=(normal + 1) / 2)
    # trimesh.exchange.export.export_mesh(surface_mesh, name)
    surface_mesh.export(name + '.obj')


def ela_zero_check(path):
    if path in ["000.png","001.png","002.png","003.png","008.png","009.png","010.png","011.png"]:
        return True
    else:
        return False

def fov_check(trans_info,fov_range):
    fov = trans_info['fov']
    proj_type = trans_info['proj_type']
    if proj_type == "ortho":
        return True
    elif proj_type == "persp":
        if fov>fov_range[0] and fov<fov_range[1]:
            return True
        else:
            return False

class AlignedShapeLatentDataset(IterableDataset):

    def __init__(self,
                 datalist: list,
                 meta_info: Dict[str, Dict[str, str]],
                #  tokenizer: SimpleTokenizer,
                 shape_transform=None,
                 image_transform=None,
                 cond_stage_key: str = "all",
                 sampling: bool = True,
                 num_samples: int = 4096,
                 num_near_samples: int = 4096,
                 num_sharpedge_samples: int = 4096,
                 pc_size: int = 2048,
                 pc_sharpedge_size: int = 2048,
                 return_normal: bool = False,
                 surface_sampling: bool = True,
                 label_type: str = "binary",
                 sdf_res: int = 128,
                 image_size: int = 518,
                 view_number: int = 1,
                 buffer_size: int = 3000,
                 yield_size: int = 3000,
                 rotation: bool = False,
                 rotationdict_path: str = None,
                 caption_path: str = None,
                 imgdict_path: str = None,
                 padding: bool = False,
                 padding_ratio_range: list = [1.15, 1.15],
                 blur: bool = False,
                 blur_prob: float = 0.,
                 img_rotation: bool = False,
                 img_rotation_prob: float = 0.,
                 img_rotation_angle: list = [0, 0],
                 contour: bool = False,
                 contour_prob: float = 0.,
                 contour_thickness_range: list = [0, 0],
                 ela_zero_aug: bool = False,
                 ela_zero_aug_prob: float = 0.,
                 fov_range: list = [10,90],
                 radial_distort_aug: bool = False,
                 radial_distort_prob: float = 0.,
                 distort_range: list = [-0.5, 0.5],
                 filter_img_path: str = None,
                 use_high_elev_view: bool = False,
                 high_elev_prob: float = 0.15,
                 use_fill_hole: bool = False,
                 fill_hole_prob: float = 0.,
                 filter_list_path: str = None,
                 load_image_to_memory: bool = False,
                 sharpedge_label: bool = False,
                 ):
        super().__init__()

        self.shardlists = ResampledShards(datalist)
        self.meta_info = meta_info

        self.cond_stage_key = cond_stage_key
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        # self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.shape_transform = shape_transform
        self.num_samples = num_samples
        self.num_near_samples = num_near_samples
        self.num_sharpedge_samples = num_sharpedge_samples
        self.sharpedge_label = sharpedge_label

        self.sampling = sampling
        self.view_number = view_number
        self.return_normal = return_normal
        self.surface_sampling = surface_sampling
        self.yield_size = yield_size
        self.buffer_size = buffer_size
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.blur = blur
        self.blur_prob = blur_prob
        self.img_rotation = img_rotation
        self.img_rotation_prob = img_rotation_prob
        self.img_rotation_angle = img_rotation_angle
        self.ela_zero_aug = ela_zero_aug
        self.ela_zero_aug_prob = ela_zero_aug_prob
        self.contour = contour
        self.contour_prob = contour_prob
        self.contour_thickness_range = contour_thickness_range
        self.filter_list_path = filter_list_path
        self.load_image_to_memory = load_image_to_memory
        self.msgpack = imgdict_path.endswith(".msgpack")

        self.fov_range = fov_range
        self.radial_distort_aug = radial_distort_aug
        self.radial_distort_prob = radial_distort_prob
        self.distort_range = distort_range

        self.use_high_elev_view = use_high_elev_view
        self.high_elev_prob = high_elev_prob

        self.use_fill_hole = use_fill_hole
        self.fill_hole_prob = fill_hole_prob

        self.image_size=image_size

        self.filter_img_path = filter_img_path

        if load_image_to_memory:
            self.image_dict = {}

        print(f'*' * 100)

        print(f'Dataset Infos:')
        print(f'# of Surface Points: {self.pc_size}')
        print(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        if self.cond_stage_key in ["all", "shape", "avae"]:
            print(f'# of Volume Points: {self.num_samples}')
            print(f'# of Near Points: {self.num_near_samples}')
            print(f'# of Sharpedge Points: {self.num_sharpedge_samples}')
        print(f'Using sharp edge label: {self.sharpedge_label}')
        print(f'Using filter dict: {self.filter_list_path}')

        if self.padding:
            print(f'Padding is open!, ratio_range={self.padding_ratio_range}')
        else:
            print(f'No padding!')
        if self.blur:
            assert self.blur_prob > 0, "blur_prob should be set when blur is True"
            print(f'Blur is open! prob={self.blur_prob}')
        else:
            print(f'No blur!')
        if self.contour:
            assert self.contour_prob > 0, "contour_prob should be set when contour is True"
            assert len(self.contour_thickness_range) == 2, "contour_thickness_range should be a list of two values"
            assert isinstance(self.contour_thickness_range[0], int) and isinstance(self.contour_thickness_range[1],
                                                                                   int), "contour_thickness_range should be integers"
            assert self.contour_thickness_range[0] != self.contour_thickness_range[
                1], "contour_thickness_range should be different"
            assert self.contour_thickness_range[0] < self.contour_thickness_range[
                1], "contour_thickness_range should be in ascending order"
            print(f'Contour is open! prob={self.contour_prob}, thickness_range={self.contour_thickness_range}')
        else:
            print(f'No contour!')
        if self.img_rotation:
            assert self.img_rotation_prob > 0, "img_rotation_prob should be set when img_rotation is True"
            assert len(self.img_rotation_angle) == 2, "img_rotation_angle should be a list of two values"
            assert isinstance(self.img_rotation_angle[0], int) and isinstance(self.img_rotation_angle[1],
                                                                              int), "img_rotation_angle should be integers"
            assert self.img_rotation_angle[0] != self.img_rotation_angle[1], "img_rotation_angle should be different"
            assert self.img_rotation_angle[0] < self.img_rotation_angle[
                1], "img_rotation_angle should be in ascending order"
            print(f'Image rotation is open! prob={self.img_rotation_prob}, angle={self.img_rotation_angle}')
        else:
            print(f'No image rotation!')

        print(f"FOV range is ", self.fov_range)
        print(f'*' * 100)

        self.label_type = label_type
        if label_type == "soft":
            print(f"soft label! sdf_res={sdf_res}")
            self.sdf_res = sdf_res
            self.sdf_func = np.vectorize(sdf_func, otypes=[float], excluded=['sdf_res'])

        elif label_type == "tsdf":
            print(f"tsdf label! sdf_res={sdf_res}")
            self.sdf_res = sdf_res
            self.tsdf_func = np.vectorize(tsdf_func, otypes=[float], excluded=['sdf_res'])

        self.rotation = rotation
        if self.rotation:
            self.rotationdata = self.get_rotation_dict(rotationdict_path)

        if self.cond_stage_key in ["all", "image", "avae"]:
            self.imgdict = self.get_img_dict(imgdict_path)

        if self.filter_img_path is not None:
            print(f'Filter extreme views')
            with open(self.filter_img_path,'r') as f:
                self.filter_img_dict = json.load(f)
        else:
            print(f"No filter extreme views")
            self.filter_img_dict = dict() # an empty dict

        self.rng = random.Random(0)
        self.debug_mode = False

        self.caption_dict = None
        if caption_path is not None:
            with open(caption_path, 'r') as f:
                self.caption_dict = json.load(f)

    def get_rotation_dict(self, rotationdict_path):
        print("open rotation successfully!")
        rotationdata = read_json(rotationdict_path)
        return rotationdata

    def get_img_dict(self, imgdict_path):
        if imgdict_path.endswith('.msgpack'):
            imgdict = read_msgpack(imgdict_path)
        elif imgdict_path.endswith('.json'):
            imgdict = read_json(imgdict_path)
        return imgdict

    def pointRotationV2(self, data, rotation_result, scale=False):
        rotation_matrix = np.eye(3)

        if rotation_result != "elevation_10_000":
            # 定义一个旋转矩阵
            rotation_matrix_s1 = trimesh.transformations.rotation_matrix(
                angle=np.deg2rad(stage1_dict[rotation_result]["angle"]),
                direction=stage1_dict[rotation_result]["direction"],
                point=None
            )
            rotation_matrix = rotation_matrix @ rotation_matrix_s1[:3, :3].T
        rotation_matrix_s0 = trimesh.transformations.rotation_matrix(
            angle=np.deg2rad(-90),
            direction=[0, 1, 0],
            point=None
        )
        rotation_matrix = rotation_matrix @ rotation_matrix_s0[:3, :3].T
        # ! Rotation = torch.tensor(rotation_matrix, dtype=torch.float32, device=data.device)
        Rotation = torch.tensor(rotation_matrix, dtype=torch.float32)
        data = torch.mm(data, Rotation)
        if scale == True:
            # scaling = (1 / torch.abs(data).max().item()) * 0.999999
            scaling = (1 / torch.abs(data).max()) * 0.999999
            data *= scaling
        return data

    def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface, sdf_vol, sdf_near, sdf_sharpedge, uid,
                                rotation_result):
        if self.surface_sampling:
            surface_normal = []
            if self.pc_size > 0:
                ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
                random_surface = random_surface[ind]
                if self.sharpedge_label:
                    sharpedge_label = np.zeros((self.pc_size, 1))
                    random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
                surface_normal.append(random_surface)
            if self.pc_sharpedge_size > 0:
                ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
                sharpedge_surface = sharpedge_surface[ind_sharpedge]
                if self.sharpedge_label:
                    sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                    sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
                surface_normal.append(sharpedge_surface)
            surface_normal = np.concatenate(surface_normal, axis=0)

        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size
        normal = surface_normal[:, 3:6]

        if self.cond_stage_key in ["all", "shape", "avae"]:
            vol_points = []
            vol_label = []
            near_points = []
            near_label = []
            sharpedge_points = []
            sharpedge_label = []
            sdf_points = []
            sdf_label = []
            if self.num_samples > 0:
                for sub_sdf_data in sdf_vol:
                    vol_points.append(sub_sdf_data["vol_points"])
                    vol_label.append(sub_sdf_data["vol_label"])
                vol_points = np.concatenate(vol_points, axis=0)
                vol_label = np.concatenate(vol_label, axis=0)
                if self.sampling:
                    ind = rng.choice(vol_points.shape[0], self.num_samples, replace=False)
                    vol_points = vol_points[ind]
                    vol_label = vol_label[ind]
                sdf_points.append(vol_points)
                sdf_label.append(vol_label)

            if self.num_near_samples > 0:
                for sub_sdf_data in sdf_near:
                    near_points.append(sub_sdf_data["random_near_points"])
                    near_label.append(sub_sdf_data["random_near_label"])
                near_points = np.concatenate(near_points, axis=0)
                near_label = np.concatenate(near_label, axis=0)
                if self.sampling:
                    ind = rng.choice(near_points.shape[0], self.num_near_samples, replace=False)
                    near_points = near_points[ind]
                    near_label = near_label[ind]
                sdf_points.append(near_points)
                sdf_label.append(near_label)
            if self.num_sharpedge_samples > 0:
                for sub_sdf_data in sdf_sharpedge:
                    sharpedge_points.append(sub_sdf_data["sharpedge_near_points"])
                    sharpedge_label.append(sub_sdf_data["sharpedge_near_label"])
                sharpedge_points = np.concatenate(sharpedge_points, axis=0)
                sharpedge_label = np.concatenate(sharpedge_label, axis=0)
                if self.sampling:
                    ind = rng.choice(sharpedge_points.shape[0], self.num_sharpedge_samples, replace=False)
                    sharpedge_points = sharpedge_points[ind]
                    sharpedge_label = sharpedge_label[ind]
                sdf_points.append(sharpedge_points)
                sdf_label.append(sharpedge_label)

            sdf_points = np.concatenate(sdf_points, axis=0)
            sdf_label = np.concatenate(sdf_label, axis=0)

            # Convert the distance to label
            if self.label_type == 'binary':
                sdf_label = sdf_label > 0
            elif self.label_type == 'soft':
                sdf_label = self.sdf_func(sdf_label, self.sdf_res)
            elif self.label_type == 'tsdf':
                sdf_label = self.tsdf_func(sdf_label, self.sdf_res)
            else:
                print(f'Unkonw Supervision Strategy')

            # far points
            points = torch.from_numpy(sdf_points)
            labels = torch.from_numpy(sdf_label).float()

            if self.rotation:
                surface = self.pointRotation(surface, uid)
                normal = self.pointRotation(normal, uid)
                points = self.pointRotation(points, uid)

            if self.shape_transform:
                surface, normal, points = self.shape_transform(surface, normal, points.float())
            geo_points = torch.cat([points, labels[:, np.newaxis]], dim=-1)
            assert geo_points.shape[0] == self.num_samples + self.num_near_samples + self.num_sharpedge_samples

        else:
            geo_points = 0.0
            surface = self.pointRotationV2(surface, rotation_result, scale=True)
            normal = self.pointRotationV2(normal, rotation_result)
            normal = F.normalize(normal, p=2, dim=1)

            if self.shape_transform:
                surface, normal = self.shape_transform(surface, normal)

        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)

        return surface, geo_points

    def random_rotation_matrix(self):
        theta_y = np.random.uniform(0, 2 * np.pi)

        # 绕y轴的旋转矩阵
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # 组合这三个旋转矩阵（顺序可能影响结果）
        rotation_matrix = Ry

        return rotation_matrix


    def GetToonImage(self, color_image, normal_image):
        mask_image = color_image[:,:,3:]
        color_image = color_image[:,:,:3]

        normals = (normal_image / 255.0 - 0.5) * 2
        normals = normals / np.linalg.norm(normals, axis=-1)[:,:,None]
        random_rot_matrix = self.random_rotation_matrix()
        normals = np.dot(normals, random_rot_matrix)
        # Create a mask where edges are detected
        mask = np.abs(normals[:,:,0])

        splits = [0.0, 0.4, 1.0]
        intensity = [0.9, 1.1]
        shadow_mask = [(mask<splits[i])*(mask>=splits[i-1]) for i in range(1,len(splits))]

        mask = intensity[0] * shadow_mask[0]
        for j in range(1,len(intensity)):
            mask += intensity[j] * shadow_mask[j]
        result = np.clip(color_image * mask[:,:,None], 0, 255).astype('uint8')
        result = result * (mask_image>0) + 255 * (mask_image==0)
        result = np.concatenate((result, mask_image),axis=-1)
        return result.astype(np.uint8)

    def load_render(self, uid):
        """主函数：加载渲染图像"""
        # 1. 加载基础路径和版本
        imgs_pack_path, render_version = self._get_render_path_and_version(uid)
        
        # 2. 加载图像包
        imgs_pack = np.load(imgs_pack_path)
        
        # 3. 过滤和选择图像
        imgs_trans_info = json.load(io.BytesIO(imgs_pack['transforms.json']))['frames']
        aug_flags = self._get_selection_and_augmentation_flags()
        selected_image_path = self._select_image(
            imgs_pack, imgs_trans_info, uid, render_version, aug_flags
        )
        
        # 4. 计算旋转结果
        image_index = int(selected_image_path.split('.')[0])
        rotation_result = self._compute_rotation_result(imgs_trans_info[image_index])
        
        # 5. 处理图像
        images, masks = self._process_images(
            [selected_image_path], imgs_pack, imgs_trans_info, 
            render_version, aug_flags
        )
        
        return images, masks, rotation_result

    def _get_render_path_and_version(self, uid):
        """获取渲染路径和版本"""
        if self.msgpack:
            imgs_pack_version_path = msgpack.unpackb(self.imgdict['image_paths'][uid])
        else:
            imgs_pack_version_path = self.imgdict['image_paths'][uid]
        
        # 判断是否使用高仰角视图
        # 注意，对于前期渲染失败、后期补充渲染的数据，其高仰角视图合并进了 render_v9 中
        use_high_elev = (
            self.use_high_elev_view 
            and random.random() < self.high_elev_prob
            and "render_v9_complete" in imgs_pack_version_path
        )
        
        render_version = 'render_v9_complete' if use_high_elev else 'render_v9'
        imgs_pack_path = imgs_pack_version_path[render_version]
        
        return imgs_pack_path, render_version
    
    def _get_selection_and_augmentation_flags(self):
        """获取所有数据选择和增强标志"""
        return {
            'ela_zero': self.ela_zero_aug and random.random() < self.ela_zero_aug_prob,
            'radial_distort': self.radial_distort_aug and random.random() < self.radial_distort_prob,
            'fill_hole': self.use_fill_hole and random.random() < self.fill_hole_prob,
            'contour': self.contour and random.random() < self.contour_prob,
            'rotation': self.img_rotation and random.random() < self.img_rotation_prob,
            'blur': self.blur and random.random() < self.blur_prob,
        }
    
    def _select_image(self, imgs_pack, imgs_trans_info, uid, render_version, aug_flags):
        """过滤并选择图像"""
        # 获取所有图像键
        imgs = list(imgs_pack.keys())
        imgs = [
            key for key in imgs 
            if ".png" in key and "basecolor" not in key and "camnorm" not in key
        ]
        imgs.sort(key=lambda x: int(x.split('.')[0]))
        
        # 过滤图像
        valid_imgs = []
        for key in imgs:
            if self._should_skip_image(key, imgs_trans_info, uid, render_version, aug_flags):
                continue
            valid_imgs.append(key)
        
        # 随机选择一张
        selected = self.rng.sample(valid_imgs, 1)
        return selected[0]
    
    def _should_skip_image(self, key, imgs_trans_info, uid, render_version, aug_flags):
        """判断是否应该跳过该图像"""
        idx = int(key.split(".")[0])
        
        # 检查仰角零度增强
        if aug_flags['ela_zero'] and not ela_zero_check(key):
            return True
        
        # 检查FOV范围
        if not fov_check(imgs_trans_info[idx], self.fov_range):
            return True
        
        # 检查过滤字典
        if self._is_in_filter_dict(uid, render_version, key):
            return True
        
        return False
    
    def _is_in_filter_dict(self, uid, render_version, key):
        """检查图像是否在过滤字典中"""
        if uid not in self.filter_img_dict:
            return False
        if render_version not in self.filter_img_dict[uid]:
            return False
        if key not in self.filter_img_dict[uid][render_version]:
            return False
        return True

    def _compute_rotation_result(self, trans_info):
        """根据方位角计算旋转结果"""
        azimuth = (float(trans_info['azi']) + 360) % 360
        
        if azimuth < 45 or azimuth >= 315:
            return 'elevation_10_000'
        elif azimuth < 135:
            return 'elevation_10_009'
        elif azimuth < 225:
            return 'elevation_10_018'
        else:
            return 'elevation_10_027'

    def _process_images(self, image_paths, imgs_pack, imgs_trans_info, 
                    render_version, aug_flags):
        """处理所有选中的图像"""
        if self.cond_stage_key not in ["all", "image", "avae"]:
            return 0.0, 0.0
        
        images = []
        masks = []
        
        for image_path in image_paths:
            image_index = int(image_path.split('.')[0])
            image = Image.open(io.BytesIO(imgs_pack[image_path]))
            image = np.asarray(image)
            
            # 提取mask并处理alpha通道
            image, mask = self._extract_mask_and_composite(image)
            
            # 应用各种增强
            image = self._apply_augmentations(
                image, mask, render_version, aug_flags
            )
            
            # 应用变换
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
            
            images.append(image)
            masks.append(mask)
        
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)[:1, ...]
        assert images.shape[0] == 3
        
        return images, masks

    def _extract_mask_and_composite(self, image):
        """提取mask并合成到白色背景"""
        if image.shape[2] != 4:
            return image, None
        
        mask = image[:, :, 3]
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        foreground = image[:, :, :3]
        background = np.ones_like(foreground) * 255
        composited = foreground * alpha + background * (1 - alpha)
        
        return composited.astype(np.uint8), mask
    
    def _apply_augmentations(self, image, mask, render_version, aug_flags):
        """应用所有数据增强"""
        # 填充孔洞
        if aug_flags['fill_hole']:
            image = self.fill_holes_with_random_color(image)
        
        # 径向畸变
        if aug_flags['radial_distort']:
            image = self._apply_radial_distort(image)
        
        # 轮廓增强
        if aug_flags['contour'] and render_version == "render_v9":
            image = self._apply_contour(image, mask)
        
        # 旋转
        if aug_flags['rotation']:
            image, mask = self._apply_rotation(image, mask)
        
        # Padding
        if self.padding:
            image, mask = self._apply_padding(image, mask)
        
        # 模糊
        if aug_flags['blur']:
            image = self._apply_blur(image)
        
        return image

    def _apply_fill_hole(self, image):
        """精确填充前景内部alpha=0的洞
        
        参数:
            image: 输入图像(RGBA格式)
        返回:
            处理后的图像
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # 提取alpha通道
        alpha = image[:, :, 3]
        # 创建前景mask(alpha>0的区域)
        fg_mask = alpha>100
        fg_mask = fg_mask.astype(np.uint8)
        
        # 快速检测是否存在空洞:计算前景mask中闭合区域的数量
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(fg_mask)
        
        # 如果只有一个连通区域(背景+前景)，说明没有空洞，直接返回
        if num_labels <= 1:
            return image
        
        # 查找前景的外部轮廓和内部空洞
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        # 创建结果图像副本
        result = image.copy()
        
        #pad some random color
        if random.random() < 0.85:
            color = (250, 250, 250)
        else:
            color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        #print("fill color",color,"number of labels",num_labels)
        # 遍历所有轮廓
        for i, cnt in enumerate(contours):
            if i>10: # break the loop, it is possible to have some endless loss
                break
            # 只处理内部空洞(子轮廓)
            if hierarchy[0][i][3] != -1:
                # 创建当前洞的mask
                hole_mask = np.zeros_like(alpha)
                cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
                #print("fill hole")
                # 确保只填充alpha=0的区域
                hole_mask = cv2.bitwise_and(hole_mask, 255 - fg_mask)
                
                if cv2.countNonZero(hole_mask) > 0:
                    # 对空洞mask进行轻微膨胀，消除边缘
                    kernel = np.ones((3,3), np.uint8)
                    hole_mask = cv2.dilate(hole_mask, kernel, iterations=1)
                    # 填充RGB通道
                    result[:, :, :3][hole_mask == 255] = color
                    # 填充Alpha通道
                    result[:, :, 3][hole_mask == 255] = 255
        return result

    def _apply_radial_distort(self, image):
        """应用径向畸变"""
        k1 = random.uniform(self.distort_range[0], self.distort_range[1])
        fov = 30 # use 30, as high fov cause more distortion
        height, width = image.shape[0:2] #assume they are the same, square
        f = width // 2 / (np.tan(fov/2))
        K = np.array([
            [f, 0 , width/2],
            [0, f, height/2],
            [0, 0, 1]])

        distortion_coeffs = np.array([k1, 0, 0, 0, 0])
        image = cv2.undistort(image, K, distortion_coeffs)
        return image

    def _apply_contour(self, image, mask):
        """应用轮廓增强"""
        forground = image.copy()
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
        thickness = random.randint(self.contour_thickness_range[0], self.contour_thickness_range[1])
        cv2.drawContours(forground, contours, -1, color, thickness)
        image[:, :, :3] = forground
        return image

    def _apply_rotation(self, image, mask):
        """应用图像旋转"""
        random_integer = np.random.randint(self.img_rotation_angle[0], self.img_rotation_angle[1])
        image = rotate_image(image, random_integer)
        mask = rotate_image(mask, random_integer)
        return image, mask
    
    def _apply_padding(self, image, mask):
        """应用图像填充"""
        h, w = image.shape[:2]
        binary = mask > 100
        non_zero_coords = np.argwhere(binary)
        x_min, y_min = non_zero_coords.min(axis=0)
        x_max, y_max = non_zero_coords.max(axis=0)
        image, mask = padding(
            image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
            mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
            center=True, padding_ratio_range=self.padding_ratio_range)
        return image, mask
    
    def _apply_blur(self, image):
        """应用图像模糊"""
        h, w = image.shape[:2]
        ratio = random.uniform(0.25, 0.75)
        h_r, w_r = int(self.image_size * ratio), int(self.image_size * ratio) #blur according to final image size
        image = cv2.resize(image, (w_r, h_r))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        return image


    # def load_render(self, uid):
    #     if self.msgpack:
    #         imgs_pack_path = msgpack.unpackb(self.imgdict['image_paths'][uid])
    #     else:
    #         imgs_pack_version_path = self.imgdict['image_paths'][uid]

    #     valid_high_elev = self.use_high_elev_view and random.random() < self.high_elev_prob \
    #         and "render_v9_complete" in imgs_pack_version_path

    #     render_version = "render_v9"
    #     if valid_high_elev:
    #         render_version = 'render_v9_complete'
    #     imgs_pack_path = imgs_pack_version_path[render_version]

    #     imgs_path = []

    #     imgs_pack = np.load(imgs_pack_path)
    #     imgs = list(imgs_pack.keys())
    #     imgs = [key for key in imgs if ".png" in key and "basecolor" not in key and "camnorm" not in key] #filter base color and normal
    #     imgs.sort(key=lambda x:int(x.split('.')[0]))

    #     rotation_result = None
    #     rotation_result_list = ['elevation_10_000', 'elevation_10_009', 'elevation_10_018', 'elevation_10_027']

    #     valid_ela_zero_aug = self.ela_zero_aug \
    #                          and random.random() < self.ela_zero_aug_prob
    #     valid_radial_distort_aug = self.radial_distort_aug \
    #                                and random.random() < self.radial_distort_prob
    #     valid_fill_hole_aug = self.use_fill_hole \
    #                          and random.random() < self.fill_hole_prob

    #     imgs_trans_info = json.load(io.BytesIO(imgs_pack['transforms.json']))['frames']
    #     for key in imgs:
    #         idx = int(key.split(".")[0])
    #         if valid_ela_zero_aug:
    #             if not ela_zero_check(key):
    #                 continue
    #         if not fov_check(imgs_trans_info[idx],self.fov_range):
    #             continue
    #         if uid in self.filter_img_dict:
    #             if render_version in self.filter_img_dict[uid]:
    #                 if key in self.filter_img_dict[uid][render_version]:
    #                     #print(f"filter {key} for {uid}")
    #                     continue
    #         imgs_path.append(key)

    #     # import pdb
    #     # pdb.set_trace()

    #     imgs_choice = self.rng.sample(imgs_path, 1)
    #     image_path = imgs_choice[0]
    #     image_index = int(image_path.split('.')[0]) #make sure the format of the transform do not change later, it is used to get the rotation results

    #     trans_info = imgs_trans_info[image_index]
    #     azi, elev = trans_info['azi'], trans_info['elev']

    #     azimuth_float = (float(azi) + 360) % 360  # ! move negative azimuth to positive
    #     if azimuth_float < 45 or azimuth_float >= 315:
    #         rotation_result = 'elevation_10_000'
    #     elif 45 <= azimuth_float < 135:
    #         rotation_result = 'elevation_10_009'
    #     elif 135 <= azimuth_float < 225:
    #         rotation_result = 'elevation_10_018'
    #     elif 225 <= azimuth_float < 315:
    #         rotation_result = 'elevation_10_027'

    #     images = []
    #     masks = []
    #     if self.cond_stage_key in ["all", "image", "avae"]:
    #         for image_path in imgs_choice:
    #             image_index = int(image_path.split('.')[0])
    #             image = Image.open(io.BytesIO(imgs_pack[image_path]))

    #             image = np.asarray(image)
    #             trans_info = imgs_trans_info[image_index]

    #             if valid_fill_hole_aug:
    #                 image = self.fill_holes_with_random_color(image)

    #             # if valid_radial_distort_aug:
    #             #     k1 = random.uniform(self.distort_range[0], self.distort_range[1])
    #             #     fov = 30 # use 30, as high fov cause more distortion
    #             #     height, width = image.shape[0:2] #assume they are the same, square
    #             #     f = width // 2 / (np.tan(fov/2))
    #             #     K = np.array([
    #             #         [f, 0 , width/2],
    #             #         [0, f, height/2],
    #             #         [0, 0, 1]])

    #             #     distortion_coeffs = np.array([k1, 0, 0, 0, 0])
    #             #     image = cv2.undistort(image, K, distortion_coeffs)
    #             image = self.distort_image(image, valid_radial_distort_aug)

    #             if image.shape[2] == 4:
    #                 mask = image[:, :, 3]
    #                 #print(mask.shape,np.amax(mask))
    #                 alpha = image[:, :, 3:4].astype(np.float32) / 255
    #                 forground = image[:, :, :3]
    #                 background = np.ones_like(forground) * 255
    #                 img_new = forground * alpha + background * (1 - alpha)
    #                 image = img_new.astype(np.uint8)


    #             # ! add contour
    #             if self.contour and random.random() < self.contour_prob and render_version=="render_v9":
    #                 forground = image.copy()
    #                 _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    #                 contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #                 color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
    #                 thickness = random.randint(self.contour_thickness_range[0], self.contour_thickness_range[1])
    #                 cv2.drawContours(forground, contours, -1, color, thickness)
    #                 image[:, :, :3] = forground

    #             # ! rotate image
    #             if self.img_rotation and random.random() < self.img_rotation_prob:
    #                 random_integer = np.random.randint(self.img_rotation_angle[0], self.img_rotation_angle[1])
    #                 image = rotate_image(image, random_integer)
    #                 mask = rotate_image(mask, random_integer)

    #             if self.padding:
    #                 h, w = image.shape[:2]
    #                 binary = mask > 100
    #                 non_zero_coords = np.argwhere(binary)
    #                 x_min, y_min = non_zero_coords.min(axis=0)
    #                 x_max, y_max = non_zero_coords.max(axis=0)
    #                 image, mask = padding(
    #                     image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
    #                     mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
    #                     center=True, padding_ratio_range=self.padding_ratio_range)

    #             # ! image blur
    #             if self.blur and random.random() < self.blur_prob: #blur should be after padding
    #                 h, w = image.shape[:2]
    #                 ratio = random.uniform(0.25, 0.75)
    #                 h_r, w_r = int(self.image_size * ratio), int(self.image_size * ratio) #blur according to final image size
    #                 image = cv2.resize(image, (w_r, h_r))
    #                 image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

    #             if self.image_transform:
    #                 image = self.image_transform(image)
    #                 mask = np.stack((mask, mask, mask), axis=-1)
    #                 mask = self.image_transform(mask)
    #             images.append(image)
    #             masks.append(mask)

    #         images = torch.cat(images, dim=0)
    #         masks = torch.cat(masks, dim=0)[:1, ...]
    #         assert images.shape[0] == 3
    #     else:
    #         images = 0.0
    #         masks = 0.0
    #     return images, masks, rotation_result

    def decode(self, item):

        sample = {}

        for key, value in item.items():

            if key == "__key__":
                sample[key] = value
                continue

            elif key == "__url__":
                sample[key] = ('/').join(value.split('/')[-3:])
                continue

            elif key.endswith(".json"):
                metadata = json_loads(value[0])
                sample["metadata"] = metadata

            elif key.endswith(".png"):
                sample["image"] = [value, item["camera.npy"]]

            elif key.endswith("random_surface.npy"):
                surface = []
                for subvalue in value:
                    subsurface = npy_loads(subvalue)
                    surface.append(subsurface)
                sample["random_surface"] = np.concatenate(surface, axis=0)[:, :6]

            elif key.endswith("sharpedge_surface.npy"):
                surface = []
                for subvalue in value:
                    subsurface = npy_loads(subvalue)
                    surface.append(subsurface)
                sample["sharpedge_surface"] = np.concatenate(surface, axis=0)

            elif key.endswith("sdf_vol.npz"):
                if self.cond_stage_key in ["all", "shape", "avae"]:
                    sdf = []
                    for subvalue in value:
                        sdf.append(npz_loads(subvalue))
                    sample["sdf_vol"] = sdf
                else:
                    sample["sdf_vol"] = None
            elif key.endswith("sdf_near.npz"):
                if self.cond_stage_key in ["all", "shape", "avae"]:
                    sdf = []
                    for subvalue in value:
                        sdf.append(npz_loads(subvalue))
                    sample["sdf_near"] = sdf
                else:
                    sample["sdf_near"] = None
            elif key.endswith("sdf_sharpedge.npz"):
                if self.cond_stage_key in ["all", "shape", "avae"]:
                    sdf = []
                    for subvalue in value:
                        sdf.append(npz_loads(subvalue))
                    sample["sdf_sharpedge"] = sdf
                else:
                    sample["sdf_sharpedge"] = None

        return sample

    def transform(self, sample):

        rng = np.random.default_rng()

        if "metadata" in sample:
            metadata = sample["metadata"]
        else:
            metadata = 0

        # ! Surface points
        if "random_surface" in sample:
            random_surface = sample["random_surface"]
        else:
            random_surface = 0
        if "sharpedge_surface" in sample:
            sharpedge_surface = sample["sharpedge_surface"]
        else:
            sharpedge_surface = 0

        # ! Supervision points
        if "sdf_vol" in sample:
            sdf_vol = sample["sdf_vol"]
        else:
            sdf_vol = 0
        if "sdf_near" in sample:
            sdf_near = sample["sdf_near"]
        else:
            sdf_near = 0
        if "sdf_sharpedge" in sample:
            sdf_sharpedge = sample["sdf_sharpedge"]
        else:
            sdf_sharpedge = 0

        if "image" in sample:
            img_data = sample["image"]
        else:
            img_data = [0, 0]

        # ! load image
        image_input, mask_input, rotation_result = self.load_render(metadata["uid"])

        # ! load surface
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface, sdf_vol, sdf_near,
                                                           sdf_sharpedge, metadata["uid"], rotation_result)

        # os.makedirs('itest-rot-fill-hole', exist_ok=True)
        # viz_pc(surface[:,:3], surface[:, 3:], image_input, f'itest-rot-fill-hole/{rotation_result}-{metadata["uid"]}')
 
        # load text
        if self.caption_dict is not None:
            text = self.caption_dict.get(metadata["uid"], '')
            if metadata['uid'] not in self.caption_dict:
                print('uid {} caption not found'.format(metadata["uid"]))
        else:
            text = ''

        # input_text = text
        # if self.cond_stage_key == "avae":
        #     input_text = self.tokenizer(text)

        sample = {
            "__key__": sample["__key__"],
            "__url__": sample["__url__"],
            "surface": surface,
            "geo_points": geo_points,
            "image": image_input,
            "mask": mask_input,
            "text": text,
            "description": text
        }
        return sample

    def _shuffle(self, initial=3000, rng=None, handler=None):
        """Shuffle the data in the stream.

        This uses a buffer of size `bufsize`. Shuffling at
        startup is less random; this is traded off against
        yielding samples quickly.

        data: iterator
        bufsize: buffer size for shuffling
        returns: iterator
        rng: either random module or random.Random instance

        """
        bufsize = self.buffer_size
        if rng is None:
            rng = random.Random(int((os.getpid() + time.time()) * 1e9))
        initial = min(initial, bufsize)
        buf = []

        if self.cond_stage_key in ["all", "image", "avae"]:
            imgTarNum = self.view_number
        else:
            imgTarNum = 0

        if self.cond_stage_key not in ["all", "shape", "avae"]:
            self.num_samples = 0
            self.num_near_samples = 0
            self.num_sharpedge_samples = 0

        #assert self.pc_size >= 1024 and self.pc_sharpedge_size >= 1024, "Random and Sharpedge surface points should be more than 1024"
        surfaceTarNum = int(np.ceil(self.pc_size / 1024))
        surfacesharpedgeTarNum = int(np.ceil(self.pc_sharpedge_size / 1024))

        sdfVolTarNum = int(np.ceil(self.num_samples / 2048))
        sdfNearTarNum = int(np.ceil(self.num_near_samples / 2048))
        sdfsharpedgeTarNum = int(np.ceil(self.num_sharpedge_samples / 2048))

        data = tariterators.tarfile_samples(self.shardlists,
                                            surfaceTarNum=surfaceTarNum,
                                            surfaceImportanceTarNum=surfacesharpedgeTarNum,
                                            sdfVolTarNum=sdfVolTarNum,
                                            sdfNearTarNum=sdfNearTarNum,
                                            sdfImportanceTarNum=sdfsharpedgeTarNum,
                                            img_json=True,
                                            imgTarNum=imgTarNum)
        for sample in data:
            buf.append(sample)
            if len(buf) < bufsize:
                try:
                    buf.append(next(data))  # skipcq: PYL-R1708
                except StopIteration:
                    pass
            if len(buf) >= initial:
                yield pick(buf, rng)
        while len(buf) > 0:
            yield pick(buf, rng)

    def __iter__(self):
        dataFilter = Data_Filtration(self.filter_list_path)
        total_num = 0
        failed_num = 0
        for data in self._shuffle():
            # pdb.set_trace()
            total_num += 1
            if total_num % 1000 == 0:
                print("失败数据比例：", failed_num / total_num)
            try:
                if dataFilter.whetherDeleter(json_loads(data['metadata.json'][0])['uid']):
                    continue
                if self.debug_mode:
                    print(data["__url__"])
                    print(data["__key__"])
                    print('start decode')
                sample = self.decode(data)
                if self.debug_mode:
                    print('start transform')
                sample = self.transform(sample)
                if self.debug_mode:
                    print('end transform')
            except Exception as err:
                if self.debug_mode:
                    traceback.print_exc()
                    print(f"Unexpected {err=}, {type(err)=}")
                    print(data["__url__"])
                    print(data["__key__"])
                failed_num += 1
                continue
            yield sample


class Data_Filtration():
    def __init__(self, filter_list_path):
        if filter_list_path is None:
            print('!' * 40)
            print("No filter_list_path")
            print('!' * 40)
            self.__skipList = []
        else:
            self.__skipList = read_json(filter_list_path)

    def whetherDeleter(self, uid):
        if uid in self.__skipList:
            return True
        else:
            return False


def getdata_list(meta_info, split_list):
    tar_path_list = []
    for dataset_name, dataset_meta in meta_info.items():
        for split in split_list:
            tar_paths = glob.glob(os.path.join(dataset_meta["tar_folder"], split, "sample_points", "*"))
            tar_paths = [path for path in tar_paths if os.path.isdir(path)]
            tar_path_list.extend(tar_paths)
    return tar_path_list


@rank_zero_only
def getdata_list_zero(meta_info, split_list, filter_list_path):
    print("\n" + "*" * 40)
    tar_path_list = []
    dataFilter = Data_Filtration(filter_list_path)
    total_used = 0
    total_skip = 0
    for dataset_name, dataset_meta in meta_info.items():
        for split in split_list:
            willBeUse = 0
            willBeSkip = 0
            tar_paths = glob.glob(os.path.join(dataset_meta["tar_folder"], split, "sample_points", "*"))
            from tqdm import tqdm
            for tar_path in tqdm(tar_paths):
                tarlist = read_json(os.path.join(tar_path, 'filelist.json'))
                json_url = os.path.join(tar_path, os.path.basename(tarlist["metadata_paths"][0]))
                dataset = gopen(json_url)
                stream = tarfile.open(fileobj=dataset, mode="r|*")
                for tarinfo in stream:
                    data = stream.extractfile(tarinfo).read()
                    metadata = json.loads(data)
                    uid = metadata["uid"]
                    if dataFilter.whetherDeleter(uid) == False:
                        willBeUse += 1
                    else:
                        willBeSkip += 1
            tar_path_list.extend(tar_paths)
            total_used += willBeUse
            total_skip += willBeSkip
            print(
                f"{dataset_name:<35} {split:<10} total_nums: {willBeUse + willBeSkip:<10} using_nums: {willBeUse:<10} skip_nums: {willBeSkip:<10}")
    print("-" * 40)
    print(split, " all_total_nums:", total_used + total_skip, " all_using_nums:", total_used, " all_skip nums:",
          total_skip, )
    print("-" * 40)
    print("*" * 40)
    return tar_path_list


class MultiAlignedShapeLatentModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_info,
        return_normal: bool = False,
        sampling: bool = True,
        num_samples: int = 1024,
        num_near_samples: Optional[int] = None,
        num_sharpedge_samples: Optional[int] = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        surface_sampling: bool = True,
        sharpedge_label: bool = False,
        label_type: str = "binary",
        sdf_res: int = 128,
        image_size: int = 224,
        random_crop: bool = True,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        shape_transform: Optional[dict] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        yield_size: int = 3000,
        buffer_size: int = 1000,
        cond_stage_key: str = "all",
        view_number: int = 0,
        rotation: bool = False,
        rotationdict_path: str = None,
        caption_path: str = None,
        imgdict_path: str = None,
        padding: bool = False,
        padding_ratio_range: list = [1.15, 1.15],
        filter_list_path: str = None,
        load_image_to_memory: bool = False,
        count_data=False,
        contour: bool = False,
        contour_prob: float = 0.,
        contour_thickness_range: list = [0, 0],
        blur: bool = False,
        blur_prob: float = 0.,
        img_rotation: bool = False,
        img_rotation_prob: float = 0.,
        img_rotation_angle: list = [0, 0],
        ela_zero_aug_prob: float = 0.,
        ela_zero_aug: bool = False,
        fov_range: list=[10,90],
        radial_distort_aug: bool = False,
        radial_distort_prob: float = 0.,
        distort_range: list = [-0.5, 0.5],
        filter_img_path: str = None,
        use_high_elev_view: bool = False,
        high_elev_prob: float = 0.,
        use_fill_hole: bool = False,
        fill_hole_prob: float = 0.
    ):

        super().__init__()

        self.meta_info = meta_info
        self.return_normal = return_normal
        self.sampling = sampling
        self.num_samples = num_samples
        self.num_sharpedge_samples = num_sharpedge_samples if num_near_samples is not None else num_samples // 2
        self.num_near_samples = num_near_samples if num_near_samples is not None else num_samples // 2
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.image_size = image_size
        self.random_crop = random_crop
        self.mean = mean
        self.std = std
        self.surface_sampling = surface_sampling
        self.label_type = label_type
        self.sdf_res = sdf_res
        self.cond_stage_key = cond_stage_key
        self.view_number = view_number
        # self.tokenizer = SimpleTokenizer()
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.count_data = count_data
        self.yield_size = yield_size
        self.blur = blur
        self.blur_prob = blur_prob
        self.img_rotation = img_rotation
        self.img_rotation_prob = img_rotation_prob
        self.img_rotation_angle = img_rotation_angle
        self.contour = contour
        self.contour_prob = contour_prob
        self.contour_thickness_range = contour_thickness_range
        self.ela_zero_aug = ela_zero_aug
        self.ela_zero_aug_prob = ela_zero_aug_prob
        self.fov_range=fov_range
        self.filter_img_path = filter_img_path

        self.radial_distort_aug = radial_distort_aug
        self.radial_distort_prob = radial_distort_prob
        self.distort_range = distort_range

        self.use_high_elev_view = use_high_elev_view
        self.high_elev_prob = high_elev_prob

        self.use_fill_hole = use_fill_hole
        self.fill_hole_prob = fill_hole_prob

        if random_crop:
            self.train_image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])
        else:
            self.train_image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean,
                                 std=self.std)
        ])
        # self.shape_transform = build_transforms(shape_transform)
        self.shape_transform = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.buffer_size = buffer_size
        self.rotation = rotation
        self.rotationdict_path = rotationdict_path
        self.caption_path = caption_path
        self.imgdict_path = imgdict_path
        self.filter_list_path = filter_list_path
        self.load_image_to_memory = load_image_to_memory

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        pass

    def read_tar_list(self, split_list, data_count=False):
        tar_path_list_zero = None
        # if data_count:
        #     tar_path_list_zero = getdata_list_zero(self.meta_info, split_list, self.filter_list_path)
        tar_path_list = getdata_list(self.meta_info, split_list)
        if dist.is_initialized():
            dist.barrier()
        if tar_path_list_zero is not None:
            assert tar_path_list == tar_path_list_zero
        return tar_path_list

    def train_dataloader(self):
        tar_urls = self.read_tar_list(["train"], data_count=self.count_data)
        print(f"Found {len(tar_urls)} tar files")
        asl_params = {
            "datalist": tar_urls,
            "meta_info": self.meta_info,
            "cond_stage_key": self.cond_stage_key,
            "return_normal": self.return_normal,
            "sampling": self.sampling,
            "num_samples": self.num_samples,
            "num_near_samples": self.num_near_samples,
            "num_sharpedge_samples": self.num_sharpedge_samples,
            "sharpedge_label": self.sharpedge_label,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "surface_sampling": self.surface_sampling,
            "label_type": self.label_type,
            "sdf_res": self.sdf_res,
            "image_size": self.image_size,
            # "tokenizer": self.tokenizer,
            "image_transform": self.train_image_transform,
            "shape_transform": self.shape_transform,
            "view_number": self.view_number,
            "buffer_size": self.buffer_size,
            "rotation": self.rotation,
            "rotationdict_path": self.rotationdict_path,
            'caption_path': self.caption_path,
            'imgdict_path': self.imgdict_path,
            'padding': self.padding,
            'padding_ratio_range': self.padding_ratio_range,
            'filter_list_path': self.filter_list_path,
            "yield_size": self.yield_size,
            "load_image_to_memory": self.load_image_to_memory,
            "contour": self.contour,
            "contour_prob": self.contour_prob,
            "contour_thickness_range": self.contour_thickness_range,
            "blur": self.blur,
            "blur_prob": self.blur_prob,
            "img_rotation": self.img_rotation,
            "img_rotation_prob": self.img_rotation_prob,
            "img_rotation_angle": self.img_rotation_angle,
            "ela_zero_aug": self.ela_zero_aug,
            "ela_zero_aug_prob": self.ela_zero_aug_prob,
            "fov_range":self.fov_range,
            "radial_distort_aug":self.radial_distort_aug,
            "radial_distort_prob":self.radial_distort_prob,
            "distort_range":self.distort_range,
            "filter_img_path": self.filter_img_path,
            "use_high_elev_view": self.use_high_elev_view,
            "high_elev_prob": self.high_elev_prob,
            "use_fill_hole": self.use_fill_hole,
            "fill_hole_prob": self.fill_hole_prob
        }
        dataset = AlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):

        tar_urls = self.read_tar_list(["train"], data_count=False)
        # print(f"Found {len(tar_urls)} tar files")
        asl_params = {
            "datalist": tar_urls,
            "meta_info": self.meta_info,
            "cond_stage_key": self.cond_stage_key,
            "return_normal": self.return_normal,
            "sampling": self.sampling,
            "num_samples": self.num_samples,
            "num_near_samples": self.num_near_samples,
            "num_sharpedge_samples": self.num_sharpedge_samples,
            "sharpedge_label": self.sharpedge_label,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "surface_sampling": self.surface_sampling,
            "label_type": self.label_type,
            "sdf_res": self.sdf_res,
            "image_size": self.image_size,
            # "tokenizer": self.tokenizer,
            "image_transform": self.train_image_transform,
            # "shape_transform": self.shape_transform,
            "view_number": self.view_number,
            "rotation": self.rotation,
            "rotationdict_path": self.rotationdict_path,
            'caption_path': self.caption_path,
            'imgdict_path': self.imgdict_path,
            'padding': self.padding,
            'padding_ratio_range': self.padding_ratio_range,
            'filter_list_path': self.filter_list_path,
            "yield_size": self.yield_size,
            "load_image_to_memory": self.load_image_to_memory,
            "contour": self.contour,
            "contour_prob": self.contour_prob,
            "contour_thickness_range": self.contour_thickness_range,
            "blur": self.blur,
            "blur_prob": self.blur_prob,
            "img_rotation": self.img_rotation,
            "img_rotation_prob": self.img_rotation_prob,
            "img_rotation_angle": self.img_rotation_angle,
            "ela_zero_aug": self.ela_zero_aug,
            "ela_zero_aug_prob": self.ela_zero_aug_prob,
            "fov_range":self.fov_range,
            "radial_distort_aug":self.radial_distort_aug,
            "radial_distort_prob":self.radial_distort_prob,
            "distort_range":self.distort_range,
            "filter_img_path": self.filter_img_path,
            "use_high_elev_view": self.use_high_elev_view,
            "high_elev_prob": self.high_elev_prob,
            "use_fill_hole": self.use_fill_hole,
            "fill_hole_prob": self.fill_hole_prob,
        }

        dataset = AlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
