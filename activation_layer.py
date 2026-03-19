class ActivationLayer:
    def __init__(self, activation, activation_deriv):
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.input = None

    def forward(self, x):
        self.input = x
        return self.activation(x)

    def backward(self, grad_output):
        return grad_output * self.activation_deriv(self.input)