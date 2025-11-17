import numpy as np

class MSELoss:
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    def __init__(self):
        # Cache for predictions and targets
        self.y_hat: np.ndarray = None
        self.y: np.ndarray = None

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the loss value.
        """
        return self.forward(y_hat, y)

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the mean squared error.
        L = (1/N) * sum((y_hat - y)^2)
        """
        self.y_hat = y_hat
        self.y = y
        n = y.size
        loss = np.sum(np.power(y_hat - y, 2)) / n
        return loss

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t. the prediction (y_hat).
        This is the *first* gradient in the backpropagation chain.
        dLoss / dy_hat = (2/N) * (y_hat - y)
        """
        n = self.y.size
        grad_start = (2.0 / n) * (self.y_hat - self.y)
        return grad_start
