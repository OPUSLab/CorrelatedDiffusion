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
    generalizedDiff-3DspinGlass-independentLimit.ipynb
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

#### `generalizedDiff-3DspinGlass-independentLimit.ipynb`

Independent diffusion baseline on the 3D Edwards-Anderson spin glass (`10 x 10 x 10` lattice), the `J = 0` counterpart to the correlated 3D experiment.

#### `generalizedDiff-3DspinGlass-correlated.ipynb`

Correlated diffusion on the 3D Edwards-Anderson spin glass (`10 x 10 x 10` lattice).

### Performance and memory settings

The reverse (inference) process in all four notebooks is GPU-accelerated: the per-spin Gibbs sweep is
replaced by an equivalent group-wise vectorized sweep, and the reverse pass is batched over inference
systems on the device. The forward noising process and the training loop are unchanged. Three settings
in the config cell control this behavior:

| setting | default | meaning |
| --- | --- | --- |
| `use_hw_accel` | `True` | Enables the vectorized sweep and the batched reverse pass. Set to `False` to fall back to the original per-spin, per-system implementation, which is retained in the notebook. |
| `sweep_mode` | `'raster'` | Update order for the grouped sweep. `'raster'` preserves the sequential update order of the original per-spin sweep. `'checkerboard'` (2D only) is a valid but different sweep order, and is opt-in. |
| `rev_system_chunk` | `None` | Processes the inference systems in chunks, so peak memory scales with the chunk size rather than the total number of systems. `None` processes all systems at once and matches the unchunked path exactly. Lower it (for example to `50`) if a large run runs out of GPU memory. |

With these defaults the notebooks run the same algorithm, at the same hyperparameters, as the original
per-spin implementation. Each notebook also includes an optional self-check cell that compares the
accelerated sweep against the original per-spin sweep directly.

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

### 3. Independent 3D spin-glass baseline

```bash
jupyter notebook notebooks/generalizedDiff-3DspinGlass-independentLimit.ipynb
```

### 4. Correlated 3D spin-glass experiment

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

## Outputs

Each notebook ends with a cell that saves the full reverse-diffusion trajectory to a single `.npz` file
in the working directory. It contains one array per diffusion step, `states_t0` through `states_t100`,
each of shape `(num_systems_inference, N)` and stored as `int8` in `{-1, +1}`, alongside the `timesteps`,
`num_systems_inference` and `N` metadata arrays. The filename records the run configuration, for example
`ferro2D_diff100steps_100samples_10chains_corr.npz` for 2D and
`EA3D_diff100steps_100samples_10chains_corr.npz` for 3D.

## Notes

* This is research code intended to reproduce the experiments and figures in the paper, not a general-purpose library.
* The code is notebook-based and kept close to the original experimental workflow.
* Some helper logic is repeated between notebooks to keep each experiment self-contained.
* Users may need to adjust file paths, checkpoint settings, and output directories for their local environment.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
