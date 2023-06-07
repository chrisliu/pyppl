# PyPPL
A Python-embedded probabilistic programming language.

# Getting Started
Fetch this project with
```bash 
git clone --recurse-submodules git@github.com:chrisliu/pyppl.git
```

Install the necessary dependencies with
```bash
pip install -r requirements.txt
```

Run the main development file (`pyppl/__main__.py`) with
```bash
python -m pyppl
```

# Project Objectives
- [x] Base API and probabilistic primitives.
- Sampling inference support.
 - [x] Compiler support for `observe`.
 - [ ] Probabilistic booleans.
 - Sampling techniques.
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
