import numpy as np
from typing import List, Tuple, Generator

class Module:
    """
    The base class for all neural network modules (layers, models).
    """
    def __init__(self):
        # Caches for a single forward/backward pass
        self._parameters: List[Tuple[np.ndarray, np.ndarray]] = []
        self._modules: List['Module'] = []

    def __call__(self, *args) -> np.ndarray:
        """
        Makes the module callable (e.g., model(x)).
        This is a wrapper for the forward pass.
        """
        return self.forward(*args)

    def forward(self, *args) -> np.ndarray:
        """
        Defines the computation performed at every call.
        Must be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Defines the backward pass (gradient computation).
        Must be overridden by all subclasses.
        """
        raise NotImplementedError

    def parameters(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Recursively collects all parameters (weights/biases) and their
        gradients from this module and all submodules.
        
        Yields:
            A (parameter, gradient) tuple.
        """
        for param, grad in self._parameters:
            yield param, grad
        
        for module in self._modules:
            yield from module.parameters()

    def zero_grad(self) -> None:
        """
        Recursively resets all gradients to zero for all parameters.
        """
        for param, grad in self.parameters():
            grad.fill(0)

    def _register_module(self, name: str, module: 'Module') -> None:
        """Helper to register a submodule."""
        if not isinstance(module, Module):
            raise TypeError(f"{name} is not a Module")
        self._modules.append(module)
        setattr(self, name, module)
        
    def _register_parameter(self, name: str, shape: Tuple) -> None:
        """Helper to register a learnable parameter (e.g., weights, bias)."""
        # Initialize parameter with small random values
        param = np.random.randn(*shape) * 0.01
        # Initialize gradient with zeros
        grad = np.zeros(shape, dtype=np.float64)
        
        self._parameters.append((param, grad))
        setattr(self, name, param)
        setattr(self, f"grad_{name}", grad)


class Sequential(Module):
    """
    A container that chains modules together in sequence.
    """
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError(f"Layer {i} is not a Module")
            self.layers.append(layer)
            # Register the layer as a submodule
            self._modules.append(layer) 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passes the input sequentially through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Passes the gradient backward through all layers in reverse order.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
