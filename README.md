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
pip install git+https://github.com/victorgabillon/coral.git@main
```

### Requirements

- Python >= 3.13
- PyTorch
- valanga (board game library)

## Project Structure

```
src/coral/
├── board_evaluation.py          # Point-of-view and evaluation types
├── chi_nn.py                     # Base neural network class
└── neural_networks/
    ├── factory.py                # NN factory pattern
    ├── nn_content_evaluator.py   # Board evaluation with NNs
    ├── models/                   # NN architectures (MLP, Transformer)
    ├── input_converters/         # Board to tensor conversion
    └── output_converters/        # NN output to evaluation conversion
```

## License

GPL-3.0-only


