import numpy as np
from typing import Generator, Tuple

class SGD:
    """
    Implements Stochastic Gradient Descent (SGD).
    """
    def __init__(self, 
                 params: Generator[Tuple[np.ndarray, np.ndarray], None, None], 
                 lr: float):
        """
        Initializes the optimizer.
        
        Args:
            params: A generator of (param, grad) tuples from model.parameters().
            lr: The learning rate.
        """
        self.params = list(params)  # Store the list of (param, grad) tuples
        self.lr = lr

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).
        param = param - learning_rate * gradient
        """
        for param, grad in self.params:
            # Update the parameter in-place
            param -= self.lr * grad

    def zero_grad(self) -> None:
        """
        Resets all gradients to zero.
        """
        for param, grad in self.params:
            grad.fill(0)
