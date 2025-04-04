#! /usr/bin/env python3

import os
import json
import torch
import random
import pathlib
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))

from argparse import ArgumentParser
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.load_data_ego_path import LoadDataEgoPath


class EgoPathTrainer():
    def __init__(
            self, 
            ckpt_path: str = "",
            pretrained_ckpt_path: str = "",
            is_pretrained: bool = False
    ):
        
        # Image and label
        self.image = None
        self.label = None

        # Tensors
        self.image_tensor = None
        self.label_tensor = None

        # Metrics
        self.endpoint_loss = 0
        self.gradient_loss = 0
        self.total_loss = 0

        # Check device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        if (is_pretrained):

            