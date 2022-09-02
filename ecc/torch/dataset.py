import multiprocessing
import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from typeguard import typechecked

from ecc.torch import LABEL_QA1, LABEL_QA2, LABEL_QA3
from ecc.utils import TensorScaler


class CustomTorchDataset(Dataset):
    binary_labels: t.Dict[str, np.ndarray]
    inputs: torch.Tensor
    length: int

    @typechecked
    def __init__(
            self,
            binary_labels: t.Dict[str, np.ndarray],
            inputs: torch.Tensor,
    ) -> None:
        self.binary_labels = binary_labels
        self.inputs = inputs
        self._set_length()

    def __getitem__(self, index: t.Any) -> t.Tuple[torch.Tensor, t.Dict[str, torch.Tensor]]:
        return self.inputs[index], self._get_labels(index=index)

    def _get_labels(self, index: t.Any) -> t.Dict[str, torch.Tensor]:
        dictionary_with_tensors: t.Dict[str, torch.Tensor] = {}
        key: str
        values: np.ndarray
        for key, values in self.binary_labels.items():
            dictionary_with_tensors.update({key: torch.from_numpy(values[index]).reshape((1, -1))})
        return dictionary_with_tensors

    def _set_length(self) -> None:
        length: int = len(self.inputs)
        for ndarray in self.binary_labels.values():
            if length != len(ndarray):
                raise ValueError("Lengths of data differ.")
        self.length = length

    def __len__(self) -> int:
        return self.length


class DatasetPipeline(pl.LightningDataModule):
    inputs: t.Dict[str, torch.Tensor]
    outputs: t.Dict[str, t.Dict[str, np.ndarray]]

    training: DataLoader
    validation: DataLoader
    evaluation: DataLoader

    shuffle: bool
    batch_size: int
    scaling_range: t.Optional[t.Tuple[int, int]]

    scaler: TensorScaler

    @typechecked
    def __init__(
            self,
            inputs: t.Dict[str, torch.Tensor],
            outputs: t.Dict[str, t.Dict[str, np.ndarray]],
            batch_size: int,
            shuffle: bool = True,
            scaling_range: t.Optional[t.Tuple[int, int]] = (-1, 1),
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaling_range = scaling_range
        self.scaler = TensorScaler(scaling_range=scaling_range)
        self.scaler.fit(self.inputs["Training"])
        self._create_dataloader()

    def _create_dataloader(self) -> None:
        self._create_train_dataloader()
        self._create_val_dataloader()
        self._create_eval_dataloader()

    def _get_data_for_all_labels(self, type_: str) -> t.Dict[str, np.ndarray]:
        dictionary: t.Dict[str, np.ndarray] = {}
        for label_type in self.outputs.keys():
            dictionary.update({label_type: self.outputs[label_type][type_]})
        return dictionary

    def _get_dataset(self, type_: str) -> CustomTorchDataset:
        return CustomTorchDataset(
                binary_labels=self._get_data_for_all_labels(type_=type_),
                inputs=self.scaler.transform(self.inputs[type_]),
        )

    def _create_train_dataloader(self) -> None:
        self.training = DataLoader(
                dataset=self._get_dataset(type_="Training"),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=multiprocessing.cpu_count(),
        )

    def _create_val_dataloader(self) -> None:
        self.validation = DataLoader(
                dataset=self._get_dataset(type_="Validation"),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=multiprocessing.cpu_count(),
        )

    def _create_eval_dataloader(self) -> None:
        self.evaluation = DataLoader(
                dataset=self._get_dataset(type_="Testing"),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=multiprocessing.cpu_count(),
        )

    def train_dataloader(self) -> DataLoader:
        return self.training

    def val_dataloader(self) -> DataLoader:
        return self.validation

    def test_dataloader(self) -> DataLoader:
        return self.evaluation

    def predict_dataloader(self) -> DataLoader:
        return self.evaluation

    @property
    def output_units(self) -> t.Dict[str, int]:
        return {
                LABEL_QA1: self.outputs["QA1"]["Training"].shape[-1],
                LABEL_QA2: self.outputs["QA2"]["Training"].shape[-1],
                LABEL_QA3: self.outputs["QA3"]["Training"].shape[-1],
        }
