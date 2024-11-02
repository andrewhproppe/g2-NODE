## g2-NODE: Accelerating Quantum Emitter Characterization with Latent Neural Ordinary Differential Equations

Deep neural network models can be used to learn complex dynamics from data and reconstruct sparse or noisy signals, thereby accelerating and augmenting experimental measurements. Evaluating the quantum optical properties of solid-state single-photon emitters is a time-consuming task that typically requires interferometric photon correlation experiments, such as Photon correlation Fourier spectroscopy (PCFS) which measures time-resolved single emitter lineshapes. Here, we demonstrate a latent neural ordinary differential equation model that can forecast a complete and noise-free PCFS experiment from a small subset of noisy correlation functions. By encoding measured photon correlations into an initial value problem, the NODE can be propagated to an arbitrary number of interferometer delay times. We demonstrate this with 10 noisy photon correlation functions that are used to extrapolate an entire de-noised interferograms of up to 200 stage positions, enabling up to a 20-fold speedup in experimental acquisition time from ~3 hours to 10 minutes. Our work presents a new approach to greatly accelerate the experimental characterization of novel quantum emitter materials using deep learning.

![Fig1_v7-01](https://github.com/andrewhproppe/PhaseRetrievalNNs/assets/68742471/0cd6940d-4c24-4835-8a79-6a70863c9132)

## Usage

Please see example scripts for training different models in scripts/model_training. You can customized what encoder and decoder layers to use surrounding the NODE layer (

## Datasets

You can find the datasets used in our NeurIPS paper at: https://zenodo.org/records/13961409
You may also make your own datasets using ```data/make_devset_2d.py```
