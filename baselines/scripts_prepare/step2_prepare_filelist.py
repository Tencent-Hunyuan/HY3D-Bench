from pathlib import Path
import json
from tqdm import tqdm

### generate filelist.json for training and validation
### this json file will be used in data loader, see baselines/core/data/tariterators.py url_opener
data_path = Path("path/to/dataset_root/full")  ## modify to your own data path
splits = ["train", "val", "test"]

for split in splits:
    split_path = data_path / f'{split}/sample_points'
    chunk_paths = list(split_path.glob("chunk_*"))
    chunk_paths.sort()
    bar = tqdm(chunk_paths)
    bar.set_description(f"Processing  split")

    for chunk_path in bar:
        ## glob all tar files in the chunk folder
        tar_list = list(chunk_path.glob("*.tar"))
        tar_list.sort()
        bar.set_postfix({"chunk": f"{chunk_path.name}"})
        metar_tars = []
        surface_tars = []
        edge_tars = []
        ## loop through all tar files in the chunk folder
        for tar in tar_list:
            if "metadata" in tar.name:
                metar_tars.append(str(tar))
            elif "sharpedge" in tar.name:
                edge_tars.append(str(tar))
            elif "surface" in tar.name:
                surface_tars.append(str(tar))
        with open(chunk_path / 'filelist.json', 'w') as f:
            json.dump({
                "metadata_paths": metar_tars,
                "surface_random_paths": surface_tars,
                "surface_sharpedge_paths": edge_tars
            }, f, indent=4
            )
