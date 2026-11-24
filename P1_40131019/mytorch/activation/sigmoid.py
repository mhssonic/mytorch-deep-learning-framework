import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    one = Tensor(np.ones_like(x.data))
    neg_x = -x
    return one * ((one + neg_x.exp()) ** -1)