# Coral

A PyTorch-based neural network library for board game evaluation.

## Overview

Coral provides a flexible framework for building and deploying neural networks to evaluate board game positions. It includes modular components for input conversion, neural network architectures, and output interpretation.

## Features

- **Multiple NN Architectures**: Multi-layer perceptrons, transformers, and custom models
- **Flexible Input/Output Conversion**: Modular converters for different board representations and evaluation formats
- **Point-of-View Evaluation**: Support for evaluating positions from different player perspectives
- **PyTorch Integration**: Built on PyTorch with JIT compilation support for optimized inference

## Installation

```bash
pip install algorhino-coral==0.2.0
```

Release 0.2.0 uses the published `valanga==0.3.0`. Run
`tox --workdir /path/to/private/coral-tox` for complete validation.
`requirements/validation-py314.txt` pins the verified public Valanga wheel and
stable CPU PyTorch artifact by URL and SHA-256 for both local tox and CI.

### Requirements

- Python >= 3.14; validated with standard CPython 3.14.10, GIL enabled, `PYTHON_JIT=0`
- PyTorch >= 2.14.1 (2.14.1+cpu is validated; no release candidate)
- Valanga 0.3.0

## Project Structure

```
src/coral/
├── board_evaluation.py          # Point-of-view and evaluation types
├── chi_nn.py                     # Base neural network class
└── neural_networks/
    ├── factory.py                # NN factory pattern
    ├── nn_state_evaluator.py   # Board evaluation with NNs
    ├── models/                   # NN architectures (MLP, Transformer)
    ├── input_converters/         # Board to tensor conversion
    └── output_converters/        # NN output to evaluation conversion
```

## License

GPL-3.0-only


