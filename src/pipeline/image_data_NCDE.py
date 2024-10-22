from pathlib import Path
from typing import Callable, List, Union, Optional, Dict
from functools import lru_cache

import os
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from src.pipeline.transforms import (
    input_transform_pipeline,
    target_transform_pipeline,
)
from src.utils import paths


class PCFSDataset(Dataset):
    def __init__(
        self,
        filepath: Union[str, Path],
        seed: int = 10236,
        scale_range: Optional[List[int]] = None,
        transpose_image: bool = False,
        time_axis: int = 1,
        nsteps: tuple = (10, 25),
        add_noise: bool = True,
        **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.df = self.data["df"][:]
        self.scale_range = scale_range
        self.add_noise = add_noise
        self.transpose_image = transpose_image
        self.time_axis = time_axis
        self.nsteps = nsteps
        self.use_prior = None

        for k, v in kwargs.items():
            setattr(self, k, v)
        # get the actual time grid
        self.timesteps = [self.data["t"][:], self.data["δ"][:]]

        self.input_transforms = input_transform_pipeline(
            self.rng, self.df, self.scale_range, self.add_noise
        )

        self.target_transforms = target_transform_pipeline()

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def num_params(self) -> int:
        return len([key for key in self.data.keys() if "parameter" in key])

    @lru_cache()
    def __len__(self) -> int:
        """
        Returns the total number of g2s in the dataset.
        Because the array is n-dimensional, the length is
        given as the product of the first four dimensions.

        Returns
        -------
        int
            Number of g2s in the dataset
        """
        g2_shape = self.data["g2s"].shape
        return g2_shape[0]

    @property
    @lru_cache()
    def indices(self) -> np.ndarray:
        return np.arange(len(self))

    @property
    def data(self):
        return h5py.File(self.filepath, "r")

    @property
    @lru_cache()
    def g2s(self) -> np.ndarray:
        """
        Return the g2s stored on disk, however reshaped such that
        we flatten the grid of parameters, and left with a 2D array
        with shape (num_g2s, timesteps).

        Returns
        -------
        np.ndarray
            NumPy 1D array containing photon counts
        """
        return self.data["g2s"]

    @property
    @lru_cache()
    def optical_delays(self) -> np.ndarray:
        return self.data["δ"][:]

    @property
    @lru_cache()
    def nstage(self) -> np.ndarray:
        return self.data["nstage"][:]

    @property
    @lru_cache()
    def lag_precision(self) -> np.ndarray:
        return self.data["lag_precision"][:]

    @property
    @lru_cache()
    def time_bounds(self) -> np.ndarray:
        return self.data["time_bounds"][:]

    @property
    @lru_cache()
    def parameters(self):
        # params = [self.data[f"parameter_w{i}"][:] for i in range(self.num_params)]
        # grid = np.stack(np.meshgrid(*params, indexing="ij")).reshape(self.num_params, -1).T
        # return grid
        return self.data["params"][:]

    @property
    @lru_cache()
    def priors(self) -> np.ndarray:
        try:
            prior = self.data["I_last"]
        except Exception as e:
            print(e)
            prior = None
        return prior

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target_image = self.g2s[index]

        input_image = target_image.copy()
        params = self.parameters[index]

        # get the time stuff
        time = self.timesteps[self.time_axis]
        ode_time = torch.linspace(0.0, 1.0, len(time))

        input_image = self.input_transforms(input_image)
        target_image = self.target_transforms(target_image)

        if self.transpose_image:
            target_image = target_image.T
            input_image = input_image.T

        data = {
            "target": target_image,
            "input": input_image,
            "parameters": params,
            "time": time,
            "ode_time": ode_time,
            "nsteps": self.nsteps,
        }

        return data


class ODEDataModule(pl.LightningDataModule):
    """
    Usual arguemnets of batch_size, seed, and nworkers for the data moduel. The kwargs that are passed to PCFSDataset include:
    - transpose_image: bool, default False
        Whether to transpose the δ and τ axes
    - time_axis: int, default 1
        Which axis to use to grab the time steps of the experiment
    - nsteps: tuple, default (10, 25)
        Number of g2s to use to make t_0 (for now, this does nothing, and this number is actually chosen in the step function of the model)
    - add_noise: bool, default True
        Whether to add noise to the data
    - fixed_scale: float, default None
        Fixed level of Poisson noise
    - scale_range: list, default None
        Range of levels of Poisson noise
    """

    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_type: str = "fixed",
        val_size: float = 0.1,
        **kwargs
    ):
        super().__init__()
        # by default run with the devset
        if not h5_path:
            h5_path = paths.get("raw").joinpath("pcfs_g2_devset.h5")
        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_size = val_size
        self.split_type = split_type
        self.data_kwargs = kwargs

        header = {
            "h5_path": h5_path,
            "batch_size": self.batch_size,
        }
        self.header = {**header, **self.data_kwargs}

        self.check_h5_path()

    def check_h5_path(self):
        """Useful to run on instantiating the module class so that wandb, pytorch etc.
        don't initialize before they realize the data file doesn't exist."""
        if not os.path.exists(self.h5_path):
            raise RuntimeError("Unable to find h5 file path.")

    def setup(self, stage: Union[str, None] = None):
        full_dataset = PCFSDataset(self.h5_path, self.seed, **self.data_kwargs)

        ntotal = int(len(full_dataset))
        ntrain = int(ntotal * (1 - self.val_size))
        nval = ntotal - ntrain

        if self.split_type == "fixed":
            self.train_set = Subset(full_dataset, range(0, ntrain))
            self.val_set = Subset(full_dataset, range(ntrain, ntotal))

        elif self.split_type == "random":
            self.train_set, self.val_set = random_split(
                full_dataset,
                [ntrain, nval],
            )

        # # use 10% of the data set a test set
        # test_size = int(len(full_dataset) * 0.2)
        # self.train_set, self.val_set = random_split(
        #     full_dataset,
        #     [len(full_dataset) - test_size, test_size],
        #     torch.Generator().manual_seed(self.seed),
        # )

    @staticmethod
    def collate_ode_data(
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        batched_data = {}
        for key in keys:
            if "time" not in key:
                data = [b.get(key) for b in batch]
                if isinstance(data[0], torch.Tensor):
                    batched_data[key] = torch.vstack(data)
                else:
                    # batched_data[key] = torch.as_tensor(data)
                    batched_data[key] = torch.tensor(np.array([data]))
        # now generate the random time steps to train with
        timesteps = batch[0].get("time")
        min_steps = batch[0]["nsteps"][0]
        max_steps = batch[0]["nsteps"][1]
        nsteps = np.random.randint(min_steps, max_steps)
        nchannels = batch[0]["nchannels"]
        indices = (
            torch.randperm(len(timesteps) - 2 * nchannels)[:nsteps].sort()[0]
            + nchannels
        )
        batched_data["indices"] = indices
        batched_data["time"] = timesteps
        batched_data["ode_time"] = batch[0].get("ode_time")
        return batched_data

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=self.collate_ode_data,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=self.collate_ode_data,
        )


class NCDEDataset(PCFSDataset):
    def __init__(
        self,
        filepath: Union[str, Path],
        seed: int = 10236,
        transforms: Optional[List[Callable]] = None,
        fixed_scale: Optional[int] = None,
        scale_range: Optional[List[int]] = None,
        transpose_image: bool = False,
        time_axis: int = 1,
        nchannels: int = 1,
        nsteps: tuple = (10, 25),
        add_noise: bool = True,
        **kwargs
    ):
        super().__init__(
            filepath,
            seed,
            transforms,
            fixed_scale,
            scale_range,
            transpose_image,
            time_axis,
            nsteps,
            add_noise,
            **kwargs
        )

        self.kwargs = kwargs
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.df = self.data["df"][:]
        # do the exact same stuff for the noise-free g2s
        # except add noise
        # if transforms:
        #     self.transforms = transforms
        # else:
        #     # otherwise use the default stuff
        #     if add_noise:
        #         self.transforms = [
        #             t.NoisyInputs(self.rng, self.df),
        #             t.ArrayToTensor(),
        #             t.NormalizeBatch(),
        #             t.AddChannelDim(),
        #         ]
        #     else:
        #         self.transforms = [
        #             t.ArrayToTensor(),
        #             t.NormalizeBatch(),
        #             t.AddChannelDim(),
        #         ]

        self.fixed_scale = fixed_scale
        self.scale_range = scale_range
        self.transpose_image = transpose_image
        self.time_axis = time_axis
        self.nchannels = nchannels
        self.nsteps = nsteps

        # get the actual time grid
        self.timesteps = [self.data["t"][:], self.data["δ"][:]]

    @property
    @lru_cache()
    def cubic_spline_coeffs(self) -> np.ndarray:
        """
        Return the g2s stored on disk, however reshaped such that
        we flatten the grid of parameters, and left with a 2D array
        with shape (num_g2s, timesteps).

        Returns
        -------
        np.ndarray
            NumPy 1D array containing photon counts
        """
        return self.data["spline_coeffs"][:]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target_image = self.g2s[index]
        if self.priors is not None:
            prior = self.priors[index]
        else:
            prior = None
        # Use fixed scale, or scale range set before training
        if self.fixed_scale != None:
            scale = self.fixed_scale
        else:
            if self.scale_range != None:
                scale = 10 ** self.rng.uniform(self.scale_range[0], self.scale_range[1])
            else:
                scale = 10 ** self.rng.uniform(0, 3)
        # grab the data
        # target_image *= scale
        input_image = target_image.copy()
        spline_coeffs = self.cubic_spline_coeffs[index]
        params = self.parameters[index]
        # get the time stuff
        time = self.timesteps[self.time_axis]
        ode_time = torch.linspace(0.0, 1.0, len(time))

        data = {
            "target": target_image,
            "input": input_image,
            "spline_coeffs": spline_coeffs,
            "parameters": params,
            "scale": scale,
            "time": time,
            "ode_time": ode_time,
            "nsteps": self.nsteps,
            "nchannels": self.nchannels,
            # "prior": prior,
        }

        # run through transforms
        if self.transforms:
            for transform in self.transforms:
                data = transform(data)
        return data


class NCDEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        nworkers: int = 0,
        **kwargs
    ):
        super().__init__()
        # by default run with the devset
        if not h5_path:
            h5_path = paths.get("raw").joinpath("pcfs_g2_devset.h5")
        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.nworkers = nworkers
        self.data_kwargs = kwargs

    def setup(self, stage: Union[str, None] = None):
        full_dataset = NCDEDataset(self.h5_path, self.seed, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            torch.Generator().manual_seed(self.seed),
        )

    @staticmethod
    def collate_ode_data(
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        batched_data = {}
        for key in keys:
            if "time" not in key:
                data = [b.get(key) for b in batch]
                if isinstance(data[0], torch.Tensor):
                    batched_data[key] = torch.vstack(data)
                else:
                    batched_data[key] = torch.as_tensor(data)
        # now generate the random time steps to train with
        timesteps = batch[0].get("time")
        min_steps = batch[0]["nsteps"][0]
        max_steps = batch[0]["nsteps"][1]
        nsteps = np.random.randint(min_steps, max_steps)
        nchannels = batch[0]["nchannels"]
        indices = (
            torch.randperm(len(timesteps) - 2 * nchannels)[:nsteps].sort()[0]
            + nchannels
        )
        batched_data["indices"] = indices
        batched_data["time"] = timesteps
        batched_data["ode_time"] = batch[0].get("ode_time")
        return batched_data

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworkers,
            collate_fn=self.collate_ode_data,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.nworkers,
            collate_fn=self.collate_ode_data,
        )


if __name__ == "__main__":
    from src.modules.PCFS import load_dot
    from matplotlib import pyplot as plt
    from src.utils import get_system_and_backend

    get_system_and_backend()

    dm = ODEDataModule(
        # "pcfs_g2_2d_n100_20220622.h5",
        "pcfs_g2_2d_n50000_20240623.h5",
        batch_size=10,
        add_noise=True,
        scale_range=(1e3, 1e3),
        num_workers=4,
        pin_memory=True,
        split_type="random",
    )

    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    # g2s = batch.get("input")
    X = batch.get("input")
    Y = batch.get("target")
    # target = batch["target"]

    dot = load_dot("experimental_dot.pickle")

    expt_g2s = dot.g2.copy()

    # Set values higher than 1 to 1
    # expt_g2s[expt_g2s > 1] = 1

    # expt_g2s -= expt_g2s.min()
    # expt_g2s /= expt_g2s.max()


    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(expt_g2s);
    # plt.colorbar()
    ax[1].imshow(X[0]);
    # plt.colorbar()

    plt.plot(X[0, -1, :])
    plt.plot(expt_g2s[-1, :])
    # plt.plot(dot.tau, dot.g2[0, :])

    # plt.xscale("log")

    # plt.plot(input[0, 1, :])
    # plt.plot(input[0, 2, :])
    # plt.plot(input[0, 3, :])
    # plt.plot(input[0, 4, :])
    # plt.ylim([0, 1])
    plt.show()
