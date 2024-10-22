from numba.core.types.npytypes import Array
import numpy as np
from typing import Any, Union, Dict, Optional, List
from numba import jit, prange
from skimage.util import random_noise
from math import ceil

import torch
from torch import from_numpy
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose


class AddPoissonNoise(object):
    def __init__(self, rng, df, scale_range=(1e5, 1e6)):
        self.rng = rng
        self.df = df
        self.scale_range = scale_range

    def __call__(self, x: np.ndarray):
        # Get random number within scale_range
        scale = self.rng.uniform(*self.scale_range)

        # Scale the g2
        x *= scale

        # Normalize the division factor
        df = self.df / min(self.df)

        # Weight the g2 by the division factor
        x = x * df

        # Perform Poisson sampling of g2 weighted by division factor
        x = self.rng.poisson(x)

        # Normalize the g2 by the division factor
        x = (x / df).astype(np.float32)

        # Scale the g2 back to original scale
        x /= scale

        return x.astype(np.float32)


class Normalize(object):
    def __init__(self, submin: bool = True, minmax: tuple = (0, 1)):
        self.submin = submin
        self.minmax = minmax

    def __call__(self, y: torch.Tensor):
        y_min, y_max = self.minmax

        # Optinally subtract the minimum
        y = (y - torch.min(y)) if self.submin else y

        # Normalize to the custom range
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        y = y * (y_max - y_min) + y_min

        return y


class AddChannelDim(object):
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim

    def __call__(self, x):
        return x.unsqueeze(self.dim)


class SlidingWindow(object):
    def __init__(self, window_size: int = 5):
        self._window_size = window_size

    @property
    def window_size(self):
        return self._window_size

    def __call__(self, x: np.ndarray):
        num_windows = ceil(x.size / self.window_size)
        return timeshift_array(x, self.window_size, num_windows)


class ArrayToTensor(object):
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)


class PrepareNeuralODE(object):
    def __init__(
        self, min_steps: int, max_steps: int, rng: np.random.Generator
    ) -> None:
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.rng = rng

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        num_steps = self.rng.integers(self.min_steps, self.max_steps)
        target = batch.get("target")
        total_timesteps = target.size(0)
        # grab a specified number of steps for training
        indices = torch.randperm(total_timesteps)[:num_steps].sort()[0]
        batch["select_time"] = batch.get("ode_time")[indices]
        # get the first time step as well
        batch["t0"] = batch.get("input")[0]
        batch["selected_idx"] = indices
        return batch


def pad_collate_func(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens


def flatten_time_sequence(X: Union[torch.Tensor, np.ndarray]):
    N, T, F = X.shape
    if type(X) == torch.Tensor:
        X = X.numpy()
    data = list()
    for batch_idx in range(N):
        seq = list()
        for t in range(T):
            if t == 0:
                seq.extend(X[batch_idx, t, :])
            else:
                # skip
                seq.append(X[batch_idx, t, -1])
        data.append(seq)
    return np.vstack(data)


def timeshift_array(array: np.ndarray, window_length=4, n_timeshifts=100):
    """
    Uses a neat NumPy trick to vectorize a sliding operation on a 1D array.
    Basically uses a 2D indexer to generate n_shifts number of windows of
    n_elements length, such that the resulting array is a 2D array where
    each successive row is shifted over by one.

    The default values are optimized for a maximum of 30 atoms.

    This is based off this SO answer:
    https://stackoverflow.com/a/42258242

    Parameters
    ----------
    array : np.ndarray
        [description]
    n_elements : int
        Length of each window, by default 5
    n_timeshifts : int, optional
        Number of timeslices, by default 100

    Returns
    -------
    np.ndarray
        NumPy 2D array with rows corresponding to chunks of a sliding
        window through the input array
    """
    shifted = np.zeros((n_timeshifts, window_length), dtype=array.dtype)
    n_actual = array.size - window_length + 1
    indexer = np.arange(window_length).reshape(1, -1) + np.arange(n_actual).reshape(
        -1, 1
    )
    # shifted[:n_actual, :] = array[indexer]
    return array[indexer]


def min_max_scale(y: np.ndarray) -> np.ndarray:
    ymin, ymax = y.min(), y.max()
    return (y - ymin) / (ymax - ymin)


def poisson_gaussian_noise(y: np.ndarray, rng) -> np.ndarray:
    """
    Implements poisson + gaussian noise to the data. The
    first step takes the smooth, continuous data and uses
    for Poisson sampling, which digitizes the data. We then
    add Gaussian noise after typecasting to single precision.

    The amount of Gaussian noise is controlled by a random
    scale variable, and so the combination of both Poisson
    and Gaussian sampling should ensure a variety of spectra
    that covers both shot-limited and integration limited.

    Parameters
    ==========
    y : np.ndarray
        The normalized ground truth, noise free signal 1D array

    Returns
    =======
    x : np.ndarray
        Returns the normalized, Poisson+Gaussian noise 1D array
    """
    # the size kwarg is omitted because it'll default to the same
    # length as the input array
    x = rng.poisson(y, size=y.size).astype(np.float32)
    # for 30% of samples, we want to look at pure Poisson noise
    if rng.uniform() >= 0.95:
        noise_scale = 10 ** rng.uniform(-7.0, 1.0)
        x += rng.normal(loc=0.0, scale=noise_scale, size=x.size)
    return x


def poisson_noise(y: np.ndarray, rng) -> np.ndarray:
    """
    Implements poisson noise to the data. The
    first step takes the smooth, continuous data and uses
    for Poisson sampling, which digitizes the data.

    Parameters
    ==========
    y : np.ndarray
        The normalized ground truth, noise free signal 1D array

    Returns
    =======
    x : np.ndarray
        Returns the normalized, Poisson noise 1D array
    """
    x = rng.poisson(y, size=y.size).astype(np.float32)

    return x


def poisson_sample_log(y: np.ndarray, rng, df: np.ndarray) -> np.ndarray:
    df = df / min(df)
    # x = rng.poisson(y*df, size=len(y))
    x = rng.poisson(y * df)
    x = (x / df).astype(np.float32)

    return x


def input_transform_pipeline(rng, df, scale_range, add_noise=True):
    pipeline = Compose(
        [
            AddPoissonNoise(rng, df, scale_range) if add_noise else None,
            ArrayToTensor(),
            # Normalize(),
            # AddChannelDim(),  # for batching data
        ]
    )
    return pipeline


def target_transform_pipeline():
    """Retrieves the training (Y) data transform pipeline.
    This normalizes the data and transforms NumPy arrays
    into torch tensors.

    Returns
    -------
    Transform pipeline
        A composed pipeline for training input data transformation.
    """
    pipeline = Compose(
        [
            # Normalize(),
            ArrayToTensor(),
            # AddChannelDim(),  # for batching data
        ]
    )
    return pipeline
