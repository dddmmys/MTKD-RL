#!/usr/bin/env python3

import numpy as np
import torch.distributed as dist
import multi_quantization as quantization
import argparse
import torch
import logging
import h5py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.cifar100 import get_cifar100_dataloaders
from pathlib import Path
from typing import Union
from datetime import datetime
from functools import cached_property
from example import get_single_layer_output_from_inputs, load_teacher_list

Pathlike = Union[str, Path]

def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.
    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL
    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)



class CustomHDF5Writer:
    def __init__(self, storage_path: str, mode: str = "w"):
        p = Path(storage_path)
        self.storage_path = p.with_suffix(".h5") if p.suffix != ".h5" else p
        self.mode = mode
        self.hdf = None

    def __enter__(self):
        self.hdf = h5py.File(self.storage_path, mode=self.mode)
        return self

    def store_array(self, key: str, value: np.ndarray) -> str:
        if key in self.hdf:
            del self.hdf[key]
        self.hdf.create_dataset(key, data=value)
        return key

    def close(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            # PosixPath is not JSON serializable
            if isinstance(v, pathlib.Path) or isinstance(v, torch.device):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)



class CodebookIndexExtractor:

    def __init__(self, params: AttributeDict):
        self.params = params
        params.subsets = ["train"]
        self.init_dirs()
        setup_logger(f"{self.vq_dir}/log-vq_extraction.txt")

    def init_dirs(self):
        self.vq_dir = (self.params.kd_exp_dir / f"vq/{''.join(self.params.teacher_name_list)}_layer{self.params.embedding_layer}_cb{self.params.num_codebooks}/")
        self.vq_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = self.vq_dir / f"splits{self.params.world_size}"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.ori_manifest_dir = Path("./data/fbank/")
        self.dst_manifest_dir = Path(self.vq_dir)
        self.dst_manifest_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # Options about teacher embeddings eatraction.
        parser.add_argument(
            "--embedding-layer",
            type=int,
            help="layer to extract teacher embeddings. 3 -> features, 5 -> logits",
            default=3,
        )
        parser.add_argument(
            "--num-batch-data",
            type=int,
            default=100,
            help="num batches to train quantizer",
        )
        parser.add_argument(
            "--num-batches-extracted",
            type=int,
            default=781,
            help="num utts to extract codebook index",
        )
        parser.add_argument(
            "--num-codebooks-teacher",
            type=int,
            default=8,
            help="""number of codebooks,
            i.e. number of codebook indexes each teacher embedding is compressed.""",
        )
        parser.add_argument(
            "--use-extracted-codebook",
            type=bool,
            default=False,
            help="Whether to use the extracted codebook indexes.",
        )
        parser.add_argument(
            "--embedding-dim",
            type=int,
            default=256,
            help="""parameters used by quantizer""",
        )
        parser.add_argument(
            "--kd-exp-dir",
            type=Path,
            default="QUAD/exp/",
            help="The experiment dir",
        )
        parser.add_argument(
            "--num-codebooks",
            type=int,
            default=8,
            help="Used to construct distillation loss",
        )

    @property
    def embedding_file_path(self):
        embedding_file_id = (f"num_batches_{self.params.num_batch_data}" + f"-layer_{self.params.embedding_layer}" + "-embedding_embeddings.h5")
        embedding_file_path = self.vq_dir / embedding_file_id
        return embedding_file_path

    @property
    def quantizer_train_dl(self):
        train_loader, val_loader = get_cifar100_dataloaders(data_folder=self.params.data, batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle_train=False, use_augmentation=False, drop_last=True)
        return train_loader

    @cached_property
    def quantizer_file_path(self):
        quantizer_file_id = (f"num_batches-{self.params.num_batch_data}" + f"-layer-{self.params.embedding_layer}" + f"-num_codebooks_{self.params.num_codebooks}" + "-quantizer.pt")
        quantizer_file_path = Path(self.vq_dir) / quantizer_file_id
        return quantizer_file_path

    def extract_codebook_indexes(self):
        logging.info("Start to extract codebook indexes.")
        self.extract_codebook_indexes_imp()

    @cached_property
    def quantizer(self):
        assert self.quantizer_file_path.exists()
        quantizer = quantization.Quantizer(
            dim=self.params.embedding_dim,
            num_codebooks=self.params.num_codebooks,
            codebook_size=256,
        )
        quantizer.load_state_dict(torch.load(self.quantizer_file_path))
        quantizer.to(self.params.device)
        return quantizer

    @torch.no_grad()
    def extract_and_save_embedding(self):
        if self.embedding_file_path.exists():
            warn_message = (f"{self.embedding_file_path} already exists." + " Skip extracting embeddings from teacher model" )
            logging.warn(warn_message)
            return
        logging.info("Start to extract embeddings for training the quantizer.")
        teacher_models = load_teacher_list(self.params)
        with CustomHDF5Writer(self.embedding_file_path) as writer:
            for batch_idx, (inputs, targets) in enumerate(self.quantizer_train_dl):
                if batch_idx >= self.params.num_batch_data:
                    break
                features_3_3d, logits_3d = get_single_layer_output_from_inputs(inputs, self.params.device, teacher_models)
                if self.params.embedding_layer == 3:
                    extracted_embedding = features_3_3d.cpu().numpy()
                if self.params.embedding_layer == 5:
                    extracted_embedding = logits_3d.cpu().numpy()
                writer.store_array(key=str(batch_idx), value=extracted_embedding)
                logging.info(f"Processed batch {batch_idx} outputs.")
        logging.info(f"Processed all {self.params.num_batch_data} batches.")

    def train_quantizer(self):
        if self.quantizer_file_path.exists():
            warn_message = (f"{self.quantizer_file_path} already exists." + " Skip trainning quantizer.")
            logging.warn(warn_message)
            return
        assert self.embedding_file_path.exists()
        logging.info("Start to train quantizer.")
        trainer = quantization.QuantizerTrainer(
            dim=self.params.embedding_dim,
            bytes_per_frame=self.params.num_codebooks,
            device=self.params.device,
        )
        train, valid = quantization.read_hdf5_data(self.embedding_file_path)
        B = 512  # Minibatch size, this is very arbitrary,
        # it's close to what we used when we tuned this method.
        def minibatch_generator(data: torch.Tensor, repeat: bool):
            assert 3 * B < data.shape[0]
            cur_offset = 0
            while True if repeat else cur_offset + B <= data.shape[0]:
                start = cur_offset % (data.shape[0] + 1 - B)
                end = start + B
                cur_offset += B
                yield data[start:end, :].to(self.params.device).to(dtype=torch.float)
        for x in minibatch_generator(train, repeat=True):
            trainer.step(x)
            if trainer.done():
                break
        quantizer = trainer.get_quantizer()
        torch.save(quantizer.state_dict(), self.quantizer_file_path)

    @torch.no_grad()
    def extract_codebook_indexes_imp(self):
        manifest_file_path = self.manifest_dir / "train_ci"
        teacher_models = load_teacher_list(self.params)
        with CustomHDF5Writer(manifest_file_path) as writer:
            for batch_idx, (inputs, targets) in enumerate(self.quantizer_train_dl):
                if batch_idx >= self.params.num_batches_extracted:
                    break
                features_3_3d, logits_3d = get_single_layer_output_from_inputs(inputs, self.params.device, teacher_models)
                if self.params.embedding_layer == 3:
                    extracted_embedding = features_3_3d
                if self.params.embedding_layer == 5:
                    extracted_embedding = logits_3d
                print("before quantizer embedding : ", extracted_embedding.shape)
                codebook_indexes = self.quantizer.encode(extracted_embedding)
                print("after quantizer codebook_indexes : ", codebook_indexes.shape)
                # [N, T, C]
                codebook_indexes = codebook_indexes.to("cpu").numpy()
                assert np.min(codebook_indexes) >= 0
                assert np.max(codebook_indexes) < 256
                writer.store_array(key=str(batch_idx), value=codebook_indexes)
                message = f"Processed batch {batch_idx} outputs."
                logging.info(f"{message}.")




