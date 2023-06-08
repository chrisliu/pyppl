# PyPPL
A Python-embedded probabilistic programming language.

# Getting Started
## Installation
Install this package with `setup.py` by running
```bash
pip install .
```
in the root directory of the project (i.e., `cd path/to/pyppl`).

## Development
1. Fetch this project with
```bash 
git clone --recurse-submodules git@github.com:chrisliu/pyppl.git
```

2. Install the necessary dependencies with
```bash
pip install -r requirements.txt
```

3. Run the main development file (`pyppl/__main__.py`) with
```bash
python -m pyppl
```

## Presentation
1. Install the necessary dependencies with
```bash
pip install -r media/presentation/requirements.txt
```

2. Launch the presentation file with Jupyter notebook
```bash
jupyter notebook media/presentation/CS\ 267A\ Invited\ Presentation.ipynb
```

Alternatively, the presentation could be located in `media/presentation` after
launching it Jupyter notebook.
```bash
jupyter notebook
```

> Note: `nbdime` should help with committing Jupyter notebooks.

# Project Objectives
- [x] Base API and probabilistic primitives.
- Sampling inference support.
  - [x] Compiler support for `observe`.
  - [ ] Probabilistic booleans.
  - [ ] Sampling techniques.
    - [x] Rejection sampling.
    - [ ] MCMC.
- Exact inference support.
  - [ ] Compiler pass to construct [BDD](https://en.wikipedia.org/wiki/Binary_decision_diagram).
  - [ ] WMC of BDD.
- Hybrid inference.
  - [ ] Compiler support for hybrid inference techniques.
  - [ ] Context managers for inference techniques.

# Project History
Our final project submission for 
[UCLA CS 267A](https://web.cs.ucla.edu/~guyvdb/teaching/cs267a/2020s/)
(spring 2023) taught by Professor Guy Van den Broeck.

## Contributors
1. Joshua Duquette
2. Christopher Liu
3. Rishi Upadhyay
*(ordered by last name)*
