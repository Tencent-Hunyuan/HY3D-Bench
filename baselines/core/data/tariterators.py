#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#


"""Low level iteration functions for tar archives."""

import random
import re
import tarfile
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import braceexpand

from webdataset.gopen import gopen
from webdataset import reraise_exception
import json
import os
from . import utils

trace = False
meta_prefix = "__"
meta_suffix = "__"

import pdb

def base_plus_ext(path):
    """Split off all file extensions.

    Returns base, allext.

    Args:
        path: path with extensions

    Returns:
        path with all extensions removed
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Dict[str, Any]) -> bool:
    """Check whether a sample is valid.

    Args:
        sample: a

    Returns:
        boolean indicating whether the sample is valid.
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


# FIXME: UNUSED
def shardlist(urls, *, shuffle=False):
    """Given a list of URLs, yields that list, possibly shuffled."""
    if isinstance(urls, str):
        urls = braceexpand.braceexpand(urls)
    else:
        urls = list(urls)
    if shuffle:
        random.shuffle(urls)
    for url in urls:
        yield dict(url=url)


def url_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    surfaceTarNum: int = 4,                 # 表面随机采样的 tar 文件数量
    surfaceImportanceTarNum: int = 1,       # 表面重要采样（边缘）的 tar 文件数量
    sdfVolTarNum: int = 4,                  # SDF体素采样的 tar 文件数量
    sdfNearTarNum: int = 4,                 # SDF近表面采样的 tar 文件数量
    sdfImportanceTarNum: int = 4,           # SDF近表面重要采样（边缘）的 tar 文件数量
    img_json: int = False,                  # 是否包含图片 json 文件
    imgTarNum: int = 0,                     # 图片 tar 文件数量
    **kw: Dict[str, Any],
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    seed = utils.make_seed(
                utils.pytorch_worker_seed()
            )
    rng = random.Random(seed)
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            # 读取数据目录中的文件列表 JSON
            with open(os.path.join(url, "filelist.json"), 'r', encoding='utf-8') as file:
                tarlist = json.load(file)
            
            # 处理表面随机采样数据
            if surfaceTarNum > 0:
                # 从可用的表面随机采样文件中随机选择指定数量
                surface_selection = rng.sample(tarlist["surface_random_paths"], surfaceTarNum)
                surface_selection = [os.path.join(url, os.path.basename(path)) for path in surface_selection]
                surface_stream = []
                for surface_url in surface_selection:
                    stream = gopen(surface_url, **kw)
                    surface_stream.append(stream)
                sample.update(surface_stream=surface_stream)
            # 处理表面重要性(锐边)采样数据
            if surfaceImportanceTarNum > 0:
                surface_selection = rng.sample(tarlist["surface_sharpedge_paths"], surfaceImportanceTarNum)
                surface_selection = [os.path.join(url, os.path.basename(path)) for path in surface_selection]
                surface_stream = []
                for surface_url in surface_selection:
                    stream = gopen(surface_url, **kw)
                    surface_stream.append(stream)
                sample.update(surface_importance_stream=surface_stream)

            # 处理 SDF 体素采样数据    
            if sdfVolTarNum > 0:
                sdf_selection = rng.sample(tarlist["sdf_vol_paths"], sdfVolTarNum)
                sdf_selection = [os.path.join(url, os.path.basename(path)) for path in sdf_selection]
                sdf_stream = []
                for sdf_url in sdf_selection:
                    stream = gopen(sdf_url, **kw)
                    sdf_stream.append(stream)
                sample.update(sdf_vol_stream=sdf_stream)
            # 处理 SDF 近表面采样数据
            if sdfNearTarNum > 0:
                sdf_selection = rng.sample(tarlist["sdf_near_paths"], sdfNearTarNum)
                sdf_selection = [os.path.join(url, os.path.basename(path)) for path in sdf_selection]
                sdf_stream = []
                for sdf_url in sdf_selection:
                    stream = gopen(sdf_url, **kw)
                    sdf_stream.append(stream)
                sample.update(sdf_near_stream=sdf_stream)
            # 处理 SDF 重要性(锐边)采样数据
            if sdfImportanceTarNum > 0:
                sdf_selection = rng.sample(tarlist["sdf_sharpedge_paths"], sdfImportanceTarNum)
                sdf_selection = [os.path.join(url, os.path.basename(path)) for path in sdf_selection]
                sdf_stream = []
                for sdf_url in sdf_selection:
                    stream = gopen(sdf_url, **kw)
                    sdf_stream.append(stream)
                sample.update(sdf_importance_stream=sdf_stream)
                
            json_url = tarlist["metadata_paths"][0]
            json_url = os.path.join(url, os.path.basename(json_url)) 
            stream = gopen(json_url, **kw)
            sample.update(metadata_stream=[stream])
            yield sample
        except Exception as exn:
            # 将 URL 添加到异常参数中,便于调试
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break

def tar_file_iterator(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: the tar file stream.
        skip_meta: regexp for keys that are skipped entirely. Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
        select: predicate for selecting files. Defaults to None.

    Yields:
        a stream of samples.
    """
    # 以流式模式打开 tar 文件(mode="r|*" 表示自动检测压缩格式)
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            # 只处理常规文件
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)
            ):
                # skipping metadata for now
                continue
            # 使用正则表达式跳过特定的元数据文件
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            # 应用文件重命名函数
            if rename_files:
                fname = rename_files(fname)
            if select_files is not None and not select_files(fname):
                continue
            # 提取文件数据
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def tar_file_expander(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.

    Args:
        data: iterator over opened tar file streams.
        handler: exception handler.
        select_files: select files from tarfiles by name (permits skipping files).

    Yields:
        a stream of samples.
    """
    for source in data:
        try:
            assert isinstance(source, dict)
            assert "surface_stream" in source
            tars = []           # 存储所有数据 tar 文件的迭代器
            imgtars = []        # 图像 tar(未使用)
            sample_flag = None

            # 处理元数据流
            metadata_stream = [tar_file_iterator(
                source["metadata_stream"][0],
                handler=handler,
                select_files=select_files,
                rename_files=rename_files,
            ), source["metadata_stream"][0].name]
            
            # 处理表面随机采样数据流
            if "surface_stream" in source:
                for tar_link in source["surface_stream"]:
                    sample = tar_file_iterator(
                    tar_link,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                    ) 
                    tars.append([sample, tar_link.name])
            # 处理表面重要性(锐边)采样数据流
            if "surface_importance_stream" in source:
                for tar_link in source["surface_importance_stream"]:
                    sample = tar_file_iterator(
                    tar_link,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                    ) 
                    tars.append([sample, tar_link.name])
            # 处理 SDF 体素采样数据流
            if "sdf_vol_stream" in source:
                for tar_link in source["sdf_vol_stream"]:
                    sample = tar_file_iterator(
                    tar_link,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                    ) 
                    tars.append([sample, tar_link.name])
            # 处理 SDF 近表面采样数据流
            if "sdf_near_stream" in source:
                for tar_link in source["sdf_near_stream"]:
                    sample = tar_file_iterator(
                    tar_link,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                    ) 
                    tars.append([sample, tar_link.name])
            # 处理 SDF 重要性(锐边)采样数据流
            if "sdf_importance_stream" in source:
                for tar_link in source["sdf_importance_stream"]:
                    sample = tar_file_iterator(
                    tar_link,
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
                    ) 
                    tars.append([sample, tar_link.name])

            for sample in metadata_stream[0]:
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = metadata_stream[1]
                sample_key = base_plus_ext(sample["fname"])[0]

                yield sample 
                for tar in tars:
                    sample = next(tar[0])
                    assert (
                        isinstance(sample, dict) and "data" in sample and "fname" in sample
                    )
                    sample["__url__"] = tar[1]
                    yield sample

        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.

    Args:
        data: iterator over tarfile contents
        keys: function that takes a file name and returns a key and a suffix.
        lcase: whether to lowercase the suffix.
        suffixes: list of suffixes to keep.
        handler: exception handler.

    Raises:
        ValueError: raised if there are duplicate file names in the tar file.

    Yields:
        iterator over samples.
    """
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            # print(filesample["__url__"])
            # print(fname)
            prefix, suffix = keys(fname)
            if trace:
                print(
                    prefix,
                    suffix,
                    current_sample.keys() if isinstance(current_sample, dict) else None,
                )
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            if current_sample is None or prefix != current_sample["__key__"]:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffix in current_sample:
                current_sample[suffix].append(value)
            else:
                current_sample[suffix] = [value]
        except Exception as exn:
            exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples(
    src: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    surfaceTarNum: int = 4,
    surfaceImportanceTarNum: int = 1,
    sdfVolTarNum: int = 4,
    sdfNearTarNum: int = 4,
    sdfImportanceTarNum: int = 4,
    img_json: int = False,
    imgTarNum: int = 0,
) -> Iterable[Dict[str, Any]]:
    """Given a stream of tar files, yield samples.

    Args:
        src: stream of tar files
        handler: exception handler
        select_files: function that selects files to be included

    Returns:
        stream of samples
    """
    streams = url_opener(src, handler=handler, 
                         surfaceTarNum=surfaceTarNum,
                         surfaceImportanceTarNum=surfaceImportanceTarNum,
                         sdfVolTarNum=sdfVolTarNum,
                         sdfNearTarNum=sdfNearTarNum,
                         sdfImportanceTarNum=sdfImportanceTarNum,
                         img_json=img_json, 
                         imgTarNum = imgTarNum)
    files = tar_file_expander(
        streams, handler=handler, select_files=select_files, rename_files=rename_files
    )
    samples = group_by_keys(files, handler=handler)
    return samples

    