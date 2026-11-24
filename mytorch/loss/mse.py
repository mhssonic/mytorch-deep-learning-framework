from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    diff = preds - actual
    squared = diff ** 2
    return squared.sum() / actual.data.shape[0]
