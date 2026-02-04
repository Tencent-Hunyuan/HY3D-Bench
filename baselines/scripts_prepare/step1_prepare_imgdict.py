from path import Path
import tarfile
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# global lock for shared dictionary and error list
image_dict_lock = Lock()

def process_point_chunk(point_chunk_path, images_root):
    """process a single point cloud chunk, return the result and error info"""
    ## locate the image_save_root
    point_name = point_chunk_path.stem
    image_chunk_path = images_root / f'{point_name}'
    
    local_image_dict = {}
    error_info = None

    meta_data = point_chunk_path / f'{point_name}_metadata.tar'
    try:
        with open(meta_data, 'rb') as f:
            meta_data = tarfile.open(fileobj=f, mode='r')
            for member in meta_data.getmembers():
                if member.isfile():
                    data = meta_data.extractfile(member).read()
                    metadata = json.loads(data)
                    ## get the uid
                    uid = metadata['uid']
                    local_image_dict[uid] = {}
                    ## get the render_v9 and render_v9_complete paths
                    render_v9_path = image_chunk_path / f'{uid}_vc_render.npz'
                    render_v9_complete_path = image_chunk_path / f'{uid}_vc_complete_render.npz'
                    ## check if the files exist
                    ## for some reason, the render_v9_complete is not always available, 
                    ## sometimes, its render_v9_complete is merged into render_v9
                    if render_v9_path.exists():
                        local_image_dict[uid]['render_v9'] = str(render_v9_path)
                    if render_v9_complete_path.exists():
                        local_image_dict[uid]['render_v9_complete'] = str(render_v9_complete_path)
    except Exception as e:
        error_info = point_name
        tqdm.write(f'Error processing {point_name}: {e}')
    
    return local_image_dict, error_info

def main():
    ## replace the path with the actual path
    data_root = Path('path/to/dataset_root')
    image_dict_path = Path('path/to/save/images_dict.json')

    ## locate the subfolders
    train_root = data_root / 'train'
    points_root = train_root / 'sample_points'
    images_root = train_root / 'images'

    ## list all the chunk folders
    points_list = list(points_root.dirs()) ## list of [chunk_xxxx]
    images_list = list(images_root.dirs())

    # sort the list
    points_list.sort()
    images_list.sort()

    image_dict_paths = {}
    error_chunks = []
    
    max_workers = 32
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_point_chunk, point_chunk_path, images_root): point_chunk_path 
            for point_chunk_path in points_list
        }
        
        # Use tqdm to display progress
        with tqdm(total=len(points_list), desc='Processing points chunks') as pbar:
            for future in as_completed(future_to_chunk):
                point_chunk_path = future_to_chunk[future]
                try:
                    local_image_dict, error_info = future.result()
                    # Merge results into main dictionary
                    with image_dict_lock:
                        image_dict_paths.update(local_image_dict)
                        if error_info:
                            error_chunks.append(error_info)
                except Exception as e:
                    point_name = point_chunk_path.stem
                    tqdm.write(f'Unexpected error processing {point_name}: {e}')
                finally:
                    pbar.update(1)
            
    ## A generic key, no practical use, just to maintain consistency
    full_imgdict = {
        'image_names': [
            '000.png', '001.png', '002.png', '003.png', '004.png', '005.png', '006.png', '007.png', 
            '008.png', '009.png', '010.png', '011.png', '012.png', '013.png', '014.png', '015.png', 
            '016.png', '017.png', '018.png', '019.png', '020.png', '021.png', '022.png', '023.png', 
            '024.png', '025.png', '026.png', '027.png', '028.png', '029.png', '030.png', '031.png', 
            '032.png', '033.png', '034.png', '035.png' 
        ],
        'image_paths': image_dict_paths,
    }
    ## save the image dict
    with open(image_dict_path, 'w') as f:
        json.dump(full_imgdict, f, indent=4)
    ## save the error chunks
    with open('error_chunks.json', 'w') as f:
        json.dump(error_chunks, f, indent=4)

    print(f'Saved image dict to {image_dict_path}')

if __name__ == '__main__':
    main()

