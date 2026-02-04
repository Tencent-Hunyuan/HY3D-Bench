import torch
import sys
import munch
import os

# get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"base_dir: {base_dir}")
sys.path.append(f'{base_dir}')
# add the ULIP directory to the path
sys.path.append(f'{base_dir}/ULIP')
sys.path.append(f'{base_dir}/ULIP/models')

from ULIP.models.ULIP_models import ULIP_PointBERT
from collections import OrderedDict
import os
from ULIP.utils.tokenizer import SimpleTokenizer
import timm


dtype = torch.float32

class UlipScore:
    """
    ULIP similarity score calculator
    
    This class wraps the ULIP-PointBERT model, 
    providing functionality to compute similarity scores between point clouds and text/images.
    Uses cosine similarity to measure semantic relevance between different modalities.
    
    Attributes:
        model: ULIP-PointBERT instance
        transforms: image preprocessing transform
        tokenizer: text tokenizer
    """
    def __init__(self, ckpt_path=None):
        # load the checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        # load the state dictionary, remove the 'module.' prefix
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        args = munch.munchify({'evaluate_3d': True})
        # initialize the model
        model = ULIP_PointBERT(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        data_config = timm.data.resolve_model_data_config(model.visual)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.model = model.to(dtype)
        # initialize the tokenizer
        self.tokenizer = SimpleTokenizer()

    @torch.no_grad()
    def __call__(self, pc, texts):
        text_features = []
        # tokenize the text
        texts = self.tokenizer(texts).cuda()
        if len(texts.shape) == 1:
            texts = texts.unsqueeze(0)
        class_embeddings = self.model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings
        
        pc = pc.cuda().to(dtype)
        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features @ text_features.t()
        return logits_per_pc

    @torch.no_grad()
    def sim_img(self, pc, images):
        # preprocess the images
        image_list = [self.transforms(image).unsqueeze(0) for image in images]
        images = torch.cat(image_list, dim=0).cuda().to(dtype)
        
        # encode the images
        class_embeddings = self.model.encode_image(images)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings
        
        pc = pc.cuda().to(dtype)
        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features @ text_features.t()
        return logits_per_pc

