# mytorch/loss/ce.py
import numpy as np
from mytorch import Tensor, Dependency

def CategoricalCrossEntropy(preds: Tensor, targets: Tensor) -> Tensor:
    eps = 1e-12
    preds_clipped = preds.clip(eps, 1.0 - eps)
    loss = -(targets * preds_clipped.log() + (1 - targets) * (1 - preds_clipped).log())
    return loss.sum() / targets.data.shape[0]