from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn

from .cnn import CNN, IMAGE_SIZE

WINDOW_WIDTH = 28
WINDOW_STRIDE = 28


class LineCNNSimple(nn.Module):
    """LeNet based model that takes a line of width that is a multiple of CHAR_WIDTH."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("windo_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]
        self.cnn = CNN(data_config=data_config, args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tuple): (B, C, H, W) input

        Returns:
            torch.Tensor: (B, C, S) logits, where S is the length of the sequence and C is the number of classes
                S can be computed from W and CHAR_WIDTH
                C is self.num_classes
        """
        B, _C, H, W = x.shape
        assert H == IMAGE_SIZE # Make sure we can use our CNN class

        # Compute number of windows
        S = math.floor((W - self.WW) / self.WS + 1)

        # NOTE: type_as properly sets device
        activations = torch.zeros((B, self.num_classes, S)).type_as(x)
        for s in range(S):
            start_w = self.WS * s
            end_w = start_w + self.WW
            window = x[:, :, :, start_w:end_w] # -> (B, C, H, self.WW)
            activations[:, :, s] = self.cnn(window)

        if self.limit_output_length:
            # S might not match ground truth, so let's only take enough activations as are expected
            activations = activations[:, :, : self.output_length]
        return activations

    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window that will slide over the input image."
        )
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser




















