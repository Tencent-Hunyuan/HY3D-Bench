# -*- coding: utf-8 -*-

import os
import io
import tarfile
import json
import numpy as np
import numpy.lib.format


def mkdir(path):
    ## make directory if not exists
    os.makedirs(path, exist_ok=True)
    return path


def npy_loads(data):
    ## load npy file
    stream = io.BytesIO(data)
    return np.lib.format.read_array(stream)


def npz_loads(data):
    ## load npz file
    return np.load(io.BytesIO(data))


def json_loads(data):
    ## load json file
    return json.loads(data)


def load_json(filepath):
    ## load json file
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


def write_json(filepath, data):
    ## write json file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def extract_tar(tar_path, tar_cache_folder):
    ## extract tar file
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=tar_cache_folder)

    tar_uids = sorted(os.listdir(tar_cache_folder))
    print(f"extract tar: {tar_path} to {tar_cache_folder}")
    return tar_uids
