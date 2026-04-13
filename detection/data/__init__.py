import pandas as pd
import os

from .version import Version
from .base import BaseDataset
from .info import DatasetInfo
from . import constants as C


def get_dataset(split="train"):
    meta = pd.read_csv("../data/input_metadata/meta.csv")
    info = DatasetInfo.load("../data/input_metadata/info.yaml")
    return BaseDataset(info, meta)[split]
