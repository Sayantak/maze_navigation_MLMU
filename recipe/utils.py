"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from functools import partial
from contextlib import contextmanager
from dataclasses import dataclass
import os
import hashlib
from matplotlib import pyplot as plt
import numpy as np

from pymongo import MongoClient


def mongodb_client(mongdb_uri: str = "mongodb://localhost:27017/mongo"):
    """Constructs MongoDB client.

    Args:
        mongdb_uri (str, optional): Defaults to "mongodb://localhost:27017/mongo".

    Returns:
        MongoClient: Client used to connect to MongoDB.
    """
    db_uri = os.environ.get("MONGODB_URI", mongdb_uri)
    return MongoClient(
        host=db_uri,
        socketTimeoutMS=1800000,
        connectTimeoutMS=1800000,
    )

def get_data_root() -> str:
    if os.environ.get("DATA_ROOT", False):
        return os.environ["DATA_ROOT"]
    else:
        raise ValueError("DATA_ROOT not set, set in train_defaults.yaml or as environment variable directly")

@dataclass
class Output:
    loss: torch.Tensor
    logits: torch.Tensor


def check_model_dataset_consistency(model, datamodule):
    if model.model is None:
        return  # for MistralPL, model.model does not exist yet. just assume everything works lol
    modelcfg = model.model.config
    pretrain_name = getattr(modelcfg, "_name_or_path", None)
    if pretrain_name:
        if not hasattr(datamodule.tokenizer, "name_or_path"):
            raise ValueError(
                f"a custom tokenizer {datamodule.tokenizer} cannot be used with a pretrained model"
            )
        assert (
            datamodule.tokenizer.name_or_path == pretrain_name
        ), f"tokenizer from datamodule {datamodule.tokenizer.name_or_path} does not match pretrained model tokenizer {pretrain_name}"
    else:
        if hasattr(datamodule, "vocab_size"):
            msg = f"vocab mismatch {datamodule.vocab_size} vs {modelcfg.vocab_size}"
            assert datamodule.vocab_size <= modelcfg.vocab_size, msg


def commit_hash(output_file, script_file):

    # Calculate the hash of the script
    current_hash = _get_hash(script_file)

    # Write the hash to a file
    stored_hash_file = f"{output_file}.hash"
    with open(stored_hash_file, "w") as file:
        file.write(current_hash)


def should_create_file(output_file, script_file):
    """Check if the output file should be created based on the hash of the script file. If the hash of the script file
    has changed since the last time the output file was created, the function returns True. Otherwise, it returns False.
    """

    # Load the stored hash from a file (if it exists)
    stored_hash_file = f"{output_file}.hash"
    stored_hash = ""
    if os.path.exists(stored_hash_file):
        with open(stored_hash_file, "r") as file:
            stored_hash = file.read().strip()

    # Calculate the current hash of the script
    current_hash = _get_hash(script_file)

    # Compare the current hash with the stored hash
    if current_hash == stored_hash:
        # If the hashes match and the output file exists, no need to create the file
        if os.path.exists(output_file):
            return False
        else:
            # If the hashes match but the output file is missing, create the file
            return True
    else:
        # If the hashes don't match, create the file
        return True


def _get_hash(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
        return hashlib.sha256(file_content).hexdigest()
