## g2-NODE: Accelerating Quantum Emitter Characterization with Latent Neural Ordinary Differential Equations

Deep neural network models can be used to learn complex dynamics from data and reconstruct sparse or noisy signals, thereby accelerating and augmenting experimental measurements. Evaluating the quantum optical properties of solid-state single-photon emitters is a time-consuming task that typically requires interferometric photon correlation experiments, such as Photon correlation Fourier spectroscopy (PCFS) which measures time-resolved single emitter lineshapes. Here, we demonstrate a latent neural ordinary differential equation model that can forecast a complete and noise-free PCFS experiment from a small subset of noisy correlation functions. By encoding measured photon correlations into an initial value problem, the NODE can be propagated to an arbitrary number of interferometer delay times. We demonstrate this with 10 noisy photon correlation functions that are used to extrapolate an entire de-noised interferograms of up to 200 stage positions, enabling up to a 20-fold speedup in experimental acquisition time from ~3 hours to 10 minutes. Our work presents a new approach to greatly accelerate the experimental characterization of novel quantum emitter materials using deep learning.

![fig1_github](https://github.com/user-attachments/assets/6d1c4c56-6a3a-431f-a60b-30b0c6f4d7c0)

## Usage

Please see example scripts for training different models in scripts/model_training. The parent NODE model is ```g2ODE``` in ```src/models/ode/ode.py```. It is structured such that you can choose different submodels for the encoder and decoder layers, and has optional arguments for augmenting the dimensionality of the latent vectors (following Dupont et al. in [this work](https://doi.org/10.48550/arXiv.1904.01681)), and whether or not to use an attention layer or a simple linear layer to combine the encoded inputs into the initial state for the NODE.

We found the most performant model to be ```g2LSTMODE```, with LSTM encoder and decoder layers. To train a version of this model with optimized hyperparameters, use ```scripts/model_training/train_g2LSTMODE.py```, or instantiate the model with these hyperparameters:
```py
from src.models.ode.ode_models import (
    AttentionBlock,
    LSTMEncoder,
    LSTMDecoder,
    MLPStack,
)
from src.models.ode.ode import g2LSTMODE

model = g2LSTMODE(
    input_size=128, # this will change depending on the parameters used to create your dataset. See below
    enc_hidden_size=256,
    enc_depth=3,
    z_size=2**7,
    vf_depth=4,
    vf_hidden_size=256,
    attn_depth=2,
    attn_heads=4,
    norm=False,
    encoder=LSTMEncoder,
    vector_field=MLPStack,
    attention=AttentionBlock,
    decoder=LSTMDecoder,
    time_dim=1,
    nobs=10,
    augment=True,
    augment_size=1,
    atol=1e-2,
    rtol=1e-2,
    dropout=0.0,
    lr=5e-4,
    lr_schedule="RLROP",
    weight_decay=1e-5,
    plot_interval=10,
    fourier_weight=1.0,
    data_info=dm.header,
)
```

## Datasets

You can find the datasets used in our NeurIPS paper at: https://zenodo.org/records/13961409
You may also make your own datasets using ```data/make_devset_2d.py```
