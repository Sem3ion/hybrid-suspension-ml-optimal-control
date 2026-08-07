# Hybrid Suspension ML Control

Neural-network-based optimal control for a hybrid active/semi-active suspension on a Peugeot 207, developed for the **Applied Machine Learning** course, MSc in Mechatronic Engineering, University of Padova (Unipd, DTG).

**Authors:** Andrea Alessandri, Alex Aliaj
**A.Y.** 2025/2026 — **Instructors:** Prof. Monica Reggiani, Prof. Luigi Salmaso

> The notebook and code comments are written in Italian, as required for the course deliverable.

## Overview

The project replaces classical control laws for a car suspension with controllers learned via neural networks, able to continuously handle the transition between active and semi-active operating modes based on driving conditions and road profile. The system is modeled starting from a quarter-car (1 DOF) formulation; road profiles are generated both from real traces and synthetic ones. An "expert" module provides physical reference targets used for supervised training and for a DAgger-style closed-loop refinement stage. Tested architectures include two-headed CNNs and separate TCN networks, trained with a custom cost function combining comfort terms, physical constraints, and hinge-type penalties.

## Repository structure

- `1. Progetto AML - NOTEBOOK.ipynb` — main deliverable: full notebook (theory, data, modeling, training, evaluation, discussion).
- `fisica.py`, `simulazione.py`, `dati.py`, `strade_sintetiche.py` — physical model, dynamic simulation, dataset construction, synthetic road generation.
- `reti.py`, `perdite.py`, `xi_ottimo.py` — neural network architectures, loss functions, optimal control law.
- `controllo.py`, `contesto.py`, `aggregazione.py` — control logic, contextual scheduling, results aggregation.
- `diagnostica.py`, `grafica.py` — diagnostics, plotting and visualization utilities.
- `config.py`, `hardware.py`, `portabilita.py`, `main.py` — configuration, hardware settings, portability helpers, entry point.
- `img/`, `img_CNN2Teste/`, `img_TCN/`, `img_Ottimo/` — figures used in the notebook.
- `requirements.txt` — Python dependencies.

## Requirements

Python 3.14+ recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

**GPU note.** `requirements.txt` only pins `torch>=2.5.0`, with no platform-specific index. Whether that gives you a GPU-enabled build depends on the OS:

- **macOS (Apple Silicon):** the standard PyPI wheel already includes Metal/MPS GPU support — nothing extra to do. `hardware.py` detects and uses it automatically (`torch.backends.mps.is_available()`).
- **Windows / Linux with an NVIDIA GPU:** the plain PyPI wheel installed by `pip install -r requirements.txt` is **CPU-only** on Windows (Linux gets a CUDA build by default, Windows does not). To get CUDA support, install PyTorch **first**, with the command for your CUDA version from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), e.g.:

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```

  Installing torch first means the subsequent `pip install -r requirements.txt` will see the constraint already satisfied and won't overwrite it with the CPU-only build.
- **No matching GPU / skipped the step above:** the code still runs correctly on CPU — `hardware.py` falls back automatically and only training speed is affected, not correctness.

## Usage

**Locally:** clone the repository and open `1. Progetto AML - NOTEBOOK.ipynb` with Jupyter or VSCode, after installing the requirements above.

**Google Colab:** since the notebook relies on the custom `.py` modules in this repository, clone the repo from within Colab before running any cell that imports them:

```python
!git clone https://github.com/<your-username>/hybrid-suspension-ml-control.git
import sys
sys.path.append("hybrid-suspension-ml-control")
```

Then install the dependencies with `!pip install -r hybrid-suspension-ml-control/requirements.txt`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
