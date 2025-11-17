import numpy as np
from .module import Module

class Linear(Module):
    """
    A fully connected (dense) linear layer: y = xW + b
    """
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # Register learnable parameters
        self._register_parameter("W", (input_features, output_features))
        self._register_parameter("b", (1, output_features))
        
        # Cache for input
        self.x: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass: x @ W + b
        """
        # Cache the input 'x' for the backward pass
        self.x = x
        # W and b were set in _register_parameter
        return np.dot(x, self.W) + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradients for W, b, and the input x.
        
        Args:
            grad: The gradient from the *next* layer (dLoss / dOutput)
        """
        # Gradient for W: (dLoss / dW) = (dLoss / dOutput) * (dOutput / dW)
        # dOutput / dW = x
        self.grad_W[:] = np.dot(self.x.T, grad)
        
        # Gradient for b: (dLoss / db) = (dLoss / dOutput) * (dOutput / db)
        # dOutput / db = 1
        self.grad_b[:] = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient to pass to the *previous* layer: (dLoss / dx)
        # (dLoss / dx) = (dLoss / dOutput) * (dOutput / dx)
        # dOutput / dx = W
        grad_to_pass_back = np.dot(grad, self.W.T)
        
        return grad_to_pass_back


class Sigmoid(Module):
    """
    Applies the Sigmoid activation function element-wise.
    """
    def __init__(self):
        super().__init__()
        # Cache for activation
        self.a: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass: 1 / (1 + exp(-x))
        """
        self.a = 1 / (1 + np.exp(-x))
        return self.a

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient using the chain rule.
        (dLoss / dx) = (dLoss / da) * (da / dx)
        (da / dx) = sigmoid(x) * (1 - sigmoid(x)) = self.a * (1 - self.a)
        """
        # grad is (dLoss / da)
        grad_to_pass_back = grad * (self.a * (1 - self.a))
        return grad_to_pass_back


class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    """
    def __init__(self):
        super().__init__()
        # Cache for input
        self.x: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass: max(0, x)
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient for ReLU.
        (dLoss / dx) = (dLoss / da) * (da / dx)
        (da / dx) = 1 if x > 0, 0 otherwise
        """
        # grad is (dLoss / da)
        grad_to_pass_back = grad.copy()
        grad_to_pass_back[self.x <= 0] = 0
        return grad_to_pass_back
