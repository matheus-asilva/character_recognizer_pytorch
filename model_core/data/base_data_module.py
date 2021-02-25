from pathlib import Path
from typing import Dict
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from model_core import utils


def load_and_print_info(data_module_class: type) -> None:
    """Load EMNISTLines and print info

    Args:
        data_module_class (type): Class responsible to load data
    """

    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return
    print("Downloading raw dataset from {url} to {filename}...".format(url=metadata["url"], filename=filename))
    utils.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = utils.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


BATCH_SIZE = 128
NUM_WORKERS = 0


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule.
       Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.dims, self.output_dims, self.mapping = [None] * 3
    
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser
    
    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models.
        """
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
    
    def setup(self, stage=None):
        """Split into train, val, test, and set dims.

        Args:
            stage (torch Dataset, optional): Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test. Defaults to None.
        """
        self.data_train, self.data_val, self.data_test = [None] * 3
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)