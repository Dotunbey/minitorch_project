import numpy as np
class MSELoss:
    """
    Computes the Mean Squared Error (MSE) loss. (For Regression)
    """
    def __init__(self):
        self.y_hat: np.ndarray = None
        self.y: np.ndarray = None

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return self.forward(y_hat, y)

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        self.y_hat = y_hat
        self.y = y
        n = y.size
        loss = np.sum(np.power(y_hat - y, 2)) / n
        return loss

    def backward(self) -> np.ndarray:
        n = self.y.size
        grad_start = (2.0 / n) * (self.y_hat - self.y)
        return grad_start

class CrossEntropyLoss:
    """
    Computes the Cross-Entropy Loss with integrated Softmax. (For Classification)
    
    This class expects *raw logits* (the output of the final Linear layer)
    and *integer labels* (e.g., 0, 1, 2).
    """
    def __init__(self):
        self.probs: np.ndarray = None
        self.y: np.ndarray = None

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> float:
        return self.forward(logits, y)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the stable softmax and then the cross-entropy loss.
        """
        # 1. Store target labels
        self.y = y
        
        # 2. Compute Stable Softmax
        # Subtract max logit for numerical stability
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 3. Compute the Cross-Entropy Loss
        n = logits.shape[0]
        # Get the probabilities corresponding to the true classes
        true_class_probs = self.probs[range(n), y]
        
        # Calculate the mean negative log-likelihood
        loss = -np.mean(np.log(true_class_probs + 1e-9)) # Add epsilon for stability
        return loss

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the CCE+Softmax w.r.t. the logits.
        The gradient is (probs - y_one_hot).
        """
        n = self.probs.shape[0]
        
        # This is the magic gradient: (y_hat - y_true)
        grad = self.probs.copy()
        grad[range(n), self.y] -= 1
        
        # Average the gradient over the batch
        grad = grad / n
        return grad

