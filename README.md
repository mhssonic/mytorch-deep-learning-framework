# MyTorch — From-Scratch Deep Learning Framework

A minimal NumPy-based deep learning library modeled after PyTorch: autograd tensors, layers, activations, losses, and optimizers — implemented without relying on PyTorch/TensorFlow.

Built for a deep learning / neural networks course project (Amirkabir University of Technology).

**Resume-ready summary:** *Built MyTorch, a from-scratch NumPy deep learning framework with autograd tensors, CNN/MLP layers, losses, and optimizers (SGD, Momentum, Adam, RMSProp).*

## Highlights

- **Autograd tensors** with dependency tracking and backward pass
- **Layers:** Linear, Conv2d, MaxPool2d, AvgPool2d, Flatten
- **Activations:** ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Step
- **Losses:** MSE, Cross-Entropy
- **Optimizers:** SGD, Momentum, RMSProp, Adam
- End-to-end experiments in Jupyter (`phase1`–`phase3`), including diabetes dataset modeling

## Tech stack

Python · NumPy · Pandas · Matplotlib · Jupyter · scikit-learn

## Project layout

```
mytorch/
  tensor.py          Autograd Tensor + Dependency
  model.py           Model base
  activation/        ReLU, Sigmoid, Softmax, …
  layer/             Linear, Conv2d, pooling
  loss/              MSE, Cross-Entropy
  optimizer/         SGD, Momentum, Adam, RMSProp
  util/              initializers, data loader, flatten
phase1.ipynb         Phase 1 experiments
phase2.ipynb         Phase 2 experiments
phase3.ipynb         Phase 3 experiments (activations study)
diabetes.csv         Sample tabular dataset
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Open the phase notebooks in Jupyter or VS Code:

```bash
jupyter notebook phase1.ipynb
```

Import the library from the project root:

```python
from mytorch import Tensor, Model
from mytorch.layer import Linear
from mytorch.activation import relu
from mytorch.optimizer import Adam
```

## Topics

- Reverse-mode automatic differentiation
- Multilayer perceptrons & convolutional layers
- Loss functions and gradient-based optimization
- Activation functions and training dynamics

## Course

Deep Learning / Neural Networks coursework — Amirkabir University of Technology
