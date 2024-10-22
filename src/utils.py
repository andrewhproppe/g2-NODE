from pathlib import Path
import matplotlib
import platform
import numpy as np

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "processed": top.joinpath("data/processed"),
    "models": top.joinpath("models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts"),
    "dots": top.joinpath("dots"),
    "trained_models": top.joinpath("trained_models"),
    "training_losses": top.joinpath("data/training_losses"),
}


def get_system_and_backend():
    if platform.system() == "Linux":
        matplotlib.use("Qt5Agg")

def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":
    test = paths.get("training_losses")