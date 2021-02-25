from typing import Any, Callable, Sequence, Tuple, Union
import torch


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that simply processes data and targets through optional transforms.
       Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Args:
        data (SequenceOrTensor): commonly these are torch tensors, numpy arrays, or PIL Images
        targets (SequenceOrTensor): commonly these are torch tensors or numpy arrays
        transform (Callable, optional): function that takes a datum and returns the same. Defaults to None.
        target_transform (Callable, optional): function that takes a target and returns the same. Defaults to None.
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise Value("Data and targets must be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        """Return length of the dataset"""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a datum and its target, after processing by transforms.

        Args:
            index (int)

        Returns:
            Tuple[Any, Any]: (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target