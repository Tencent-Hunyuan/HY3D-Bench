### Clone the Uni3D and ULIP repo:
git clone https://github.com/baaivision/Uni3D.git

git clone https://github.com/salesforce/ULIP.git

### Fixing coding issue:
Replace the ''from torch._six import string_classes'' as ''string_classes = str'' in the ULIP/data/dataset_3d.py

Replace config_addr = './models/pointbert/PointTransformer_8192point.yaml' as the absolute path in the ULIP/models/ULIP_models.py

### Install KNN_CUDA
```
git clone KNN_CUDA https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
python setup.py install
```

### Checkpoints
```
mkdir checkpoints
cd checkpoints

Uni3D:
wget https://huggingface.co/BAAI/Uni3D/resolve/main/modelzoo/uni3d-g/model.pt?download=true

wget https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/resolve/main/open_clip_pytorch_model.bin

Unip:
wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt?download=true

```