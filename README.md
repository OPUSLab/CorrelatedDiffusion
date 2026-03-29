# Correlated Diffusion with Probabilistic Computers

Code accompanying the paper:

**From Independent to Correlated Diffusion: Generalized Generative Modeling with Probabilistic Computers**

This repository contains the research notebooks used to study independent and correlated diffusion on Ising systems using neural-network denoising and Gibbs-based sampling.

## Overview

Standard discrete diffusion typically uses independent site-wise noise injection. In this work, that independent process is recovered as the special case `J = 0`, while the more general correlated setting restores known Ising couplings and uses Gibbs dynamics during both noising and reverse inference.

The reverse process combines:
- a neural network that predicts per-site clean-state probabilities from a noisy state
- a Gibbs-sampling based candidate generation step under known couplings
- likelihood-based reweighting of candidate reverse states

The main benchmark systems are:
- 2D ferromagnetic Ising model
- 3D Edwards-Anderson spin glass

## Repository contents

```text
notebooks/
    generalizedDiff-2DferroIsing-independentLimit.ipynb
    generalizedDiff-2DferroIsing-correlated.ipynb
    generalizedDiff-3DspinGlass-correlated.ipynb
L50_Results_noising_100000MCS_81beta_s46_nobias/
    Trial_*.mat
3d_spin_glass_dataset_L10/
    L10.mat
    3d_spin_glass_dataset/
        long_run_samples_beta_*.txt
requirements.txt
LICENSE
```

### Notebook descriptions

#### `generalizedDiff-2DferroIsing-independentLimit.ipynb`

Independent diffusion baseline on the 2D ferromagnetic Ising model (`50 x 50` lattice).

#### `generalizedDiff-2DferroIsing-correlated.ipynb`

Correlated diffusion on the 2D ferromagnetic Ising model using interaction-aware Gibbs dynamics.

#### `generalizedDiff-3DspinGlass-correlated.ipynb`

Correlated diffusion on the 3D Edwards-Anderson spin glass (`10 x 10 x 10` lattice).

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training.

## Data

The datasets used in the paper are included directly in this repository at the default paths expected by the notebooks. No separate dataset download is required.

Included dataset paths:

* `L50_Results_noising_100000MCS_81beta_s46_nobias/` — 2D ferromagnetic Ising equilibrium configurations
* `3d_spin_glass_dataset_L10/` — 3D spin-glass state samples and J-coupling matrix (`L10.mat`)

## Reproducing the main experiments

### 1. Independent 2D baseline

```bash
jupyter notebook notebooks/generalizedDiff-2DferroIsing-independentLimit.ipynb
```

### 2. Correlated 2D experiment

```bash
jupyter notebook notebooks/generalizedDiff-2DferroIsing-correlated.ipynb
```

### 3. Correlated 3D spin-glass experiment

```bash
jupyter notebook notebooks/generalizedDiff-3DspinGlass-correlated.ipynb
```

## Main experimental setup

The paper studies:

* a `50 x 50` 2D ferromagnetic Ising system using 10,000 equilibrium configurations
* a `10 x 10 x 10` 3D Edwards-Anderson spin glass using 20,000 equilibrium configurations

The conditional estimator is a 2-hidden-layer MLP with:

* hidden width: 1024
* loss: binary cross-entropy
* optimizer: Adam
* learning rate: `1e-6`
* batch size: 512

## Notes

* This is research code intended to reproduce the experiments and figures in the paper, not a general-purpose library.
* The code is notebook-based and kept close to the original experimental workflow.
* Some helper logic is repeated between notebooks to keep each experiment self-contained.
* Users may need to adjust file paths, checkpoint settings, and output directories for their local environment.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
