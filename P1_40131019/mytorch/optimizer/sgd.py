from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            for param_name, param in layer.parameters().items():
                if param.requires_grad and param.grad is not None:
                    # print(self.learning_rate, param_name, param.grad.data)
                    param.data -= self.learning_rate * param.grad.data

                # if param.grad is not None:
                #     param.data = param.data - self.learning_rate * param.grad.data
