from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
import hashlib
import os

import numpy as np
import cv2
from tqdm import tqdm


def to_categorical(y, num_classes):
    """One-hot encode a tensor

    Args:
        y (np.ndarray): class vector to be converted into a matrix.
        num_classes (int): total number of classes.

    Returns:
        np.ndarray: A binary matrix representation of the input. The classes axis is placed last.
    
    Example:
        >>> print(to_categorical(y=[0,1,2], num_classes=3))
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]], dtype=uint8)
    """
    return np.eye(num_classes, dtype="uint8")[y]


def read_image(image_uri: Union[Path, str], grayscale=False) -> np.array:
    """Read an image from a path or a url

    Args:
        image_uri (Union[Path, str]): Path or url to image to be loaded.
        grayscale (bool, optional): Flag to convert a image into a grayscale. Defaults to False.

    Returns:
        np.array: array of bytes corresponding to the image
    """

    # Read from filename
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(str(image_filename), imread_flag)

    # Read from url
    def read_image_from_url(image_url, imread_flag):
        url_response = urlopen(str(image_url))
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)

    # Check if grayscale is True and tries to read the image    
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, imread_flag)
        else:
            img = read_image_from_url(image_uri, imread_flag)
        assert img is not None
    except Exception as e:
        raise ValueError(f"Could not load image at {image_uri}: {e}")
    return img


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    """Save the image according to the specified format in chosen directory.

    Args:
        image (np.ndarray): Array of image
        filename (Union[Path, str]): Path and filename to save the image
    """
    cv2.imwrite(str(filename), image)


def compute_sha256(filename: Union[Path, str]) -> str:
    """Create a hexadecimal representation of SHA256

    Args:
        filename (Union[Path, str]): Path and filename

    Returns:
        str: Hexadecimal representation of SHA256
    """
    with open(filename, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url: str, filename: str) -> None:
    """Download a file from url to filename, with a progress bar.

    Args:
        url (str): url of file
        filename (str): name of the file to download
    """
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec