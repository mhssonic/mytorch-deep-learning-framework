from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.mode = mode
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = x @ W.T + b
        x: (batch_size, inputs)
        W: (outputs, inputs)  → we store as (outputs, inputs)
        """
        # x @ weight.T → (batch, inputs) @ (outputs, inputs).T = (batch, inputs) @ (inputs, outputs)
        out = x @ self.weight.T

        if self.need_bias and self.bias is not None:
            out = out + self.bias
        return out
    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        weight_data = initializer((self.outputs, self.inputs), mode=self.mode)
        self.weight = Tensor(data=weight_data, requires_grad=True)

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            bias_data = np.zeros((self.outputs,), dtype=np.float64)
            self.bias = Tensor(data=bias_data, requires_grad=True)

    def zero_grad(self):
        if self.weight.requires_grad and self.weight.grad is not None:
            self.weight.zero_grad()
        if self.need_bias and self.bias is not None and self.bias.requires_grad:
            self.bias.zero_grad()

    def parameters(self):
        params = {"weight": self.weight}
        if self.need_bias and self.bias is not None:
            params["bias"] = self.bias
        return params

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
