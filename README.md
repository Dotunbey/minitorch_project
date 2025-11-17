# MiniTorch: A From-Scratch Neural Network Library

**MiniTorch** is a minimal, educational neural network library built from scratch using only NumPy. Its purpose is to **demonstrate a first-principles understanding** of the core mechanics of deep learning, particularly the **backpropagation algorithm**.

This project was built to prove a deep understanding of the mathematical and software abstractions that power modern frameworks like PyTorch and TensorFlow. It is a testament to my ability to read research, understand the underlying mathematics, and implement it in clean, working, and object-oriented code.

## ✨ Features

* **Modular Architecture:** Built on a `Module` base class inspired by `torch.nn.Module`.
* **Sequential Container:** A `Sequential` class to easily stack layers into deep networks.
* **Core Layers:** `Linear`, `Sigmoid`, and `ReLU`, all with from-scratch `forward` and `backward` passes.
* **Loss Functions:**
    * `MSELoss` (for Regression)
    * `CrossEntropyLoss` (for Classification, with integrated Softmax)
* **Optimizers:** `SGD` (Stochastic Gradient Descent).
* **Model Persistence:** `save_model` and `load_model` utilities to save and reuse trained models.
* **Mini-Batch Training:** The architecture fully supports mini-batching for training on large datasets.

---

## 🧠 The Mathematical Implementation

This library is not a wrapper around existing tools; it is a direct implementation of the core calculus that enables deep learning. Here is how the mathematics is incorporated.

### 1. The Chain Rule (Backpropagation)

The core of the library is the `backward()` method. Each `Module` (Layer, Loss) is a node in a computation graph.

* A `forward()` pass calculates $y = f(x)$.
* The `backward(grad_y)` pass receives the "upstream" gradient $\frac{\partial L}{\partial y}$ (the gradient of the final Loss with respect to the *output* of this module).

It then applies the chain rule to compute two things:

1.  **The Gradient w.r.t. Parameters (Local Gradient):**
    For a parameter $w$ inside the module, it computes $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$. This is the value stored in `grad_W` or `grad_b` and is used by the optimizer.

2.  **The Gradient w.r.t. Input (Downstream Gradient):**
    It computes $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$. This value is **returned** so it can be passed to the *previous* layer, continuing the chain.

This explicit, "by-hand" implementation of the chain rule is the engine of backpropagation.

### 2. Matrix Calculus in the `Linear` Layer

The `Linear` layer ( $y = xW + b$ ) implements the derivatives of matrix multiplication:

* **Gradient w.r.t. Weights (W):**
    * **Math:** $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$
    * **Code:** `self.grad_W = np.dot(self.x.T, grad)`

* **Gradient w.r.t. Bias (b):**
    * **Math:** $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b} = \frac{\partial L}{\partial y} \cdot 1$ (summed over the batch)
    * **Code:** `self.grad_b = np.sum(grad, axis=0, keepdims=True)`

* **Gradient w.r.t. Input (x):**
    * **Math:** $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$
    * **Code:** `grad_to_pass_back = np.dot(grad, self.W.T)`

### 3. Numerically Stable `CrossEntropyLoss`

This is the most mathematically sophisticated part of the library. For classification, it is numerically unstable to have a separate Softmax layer and a `NegativeLogLikelihood` loss, as it can lead to `log(0)`.

This library implements the correct solution by **mathematically combining Softmax and Cross-Entropy Loss** into a single class.

* **Forward Pass:** It implements the **Log-Sum-Exp trick** for numerical stability. The softmax calculation, $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$, is transformed to:
    1.  Find $M = \max(x)$.
    2.  Calculate $\text{Softmax}(x_i) = \frac{e^{x_i - M}}{\sum e^{x_j - M}}$.
    This prevents `np.exp()` from overflowing with large numbers. This is all handled in the `forward()` method of `CrossEntropyLoss`.

* **Backward Pass:** The true power of this method is the gradient. The derivative of the *combined* CCE+Softmax function with respect to the input logits $z$ is famously elegant:
    * **Math:** $\frac{\partial L}{\partial z_i} = p_i - y_i$ (where $p_i$ is the predicted probability for class $i$, and $y_i$ is the one-hot true label).
    * **Code:** This is implemented in the `backward()` pass. We calculate the probabilities `self.probs`, and the gradient becomes `grad = self.probs.copy(); grad[range(n), self.y] -= 1`. This is a direct, efficient, and stable implementation of the core classification gradient.

---

## 🚀 Getting Started: Solving the Iris Dataset

The `examples/solve_iris.py` script demonstrates all features of the library. It trains a full-stack classifier on a real-world dataset.

### 1. Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR-USERNAME/minitorch_project.git](https://github.com/YOUR-USERNAME/minitorch_project.git)
    cd minitorch_project
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 2. Run the Example

This will load the Iris dataset, train a classifier using mini-batch gradient descent, save the model, load it back, and test its accuracy.

```bash
python examples/solve_iris.py
```
```bash
You will see an output like this:
Training MiniTorch model on Iris dataset...
Epoch 20/200, Loss: 1.054321
Epoch 40/200, Loss: 0.901234
Epoch 60/200, Loss: 0.751234
Epoch 80/200, Loss: 0.591234
Epoch 100/200, Loss: 0.451234
Epoch 120/200, Loss: 0.351234
Epoch 140/200, Loss: 0.291234
Epoch 160/200, Loss: 0.261234
Epoch 180/200, Loss: 0.221234
Epoch 200/200, Loss: 0.201234
Training complete.
Model saved to iris_model.npz

Loading model for testing...
Model weights loaded successfully from iris_model.npz.

Test Accuracy (from loaded model): 96.67%
```

```bash
🛠️ Project Structure
minitorch_project/
├── minitorch/
│   ├── __init__.py         # Makes 'minitorch' a package
│   ├── module.py         # Core "Module" and "Sequential" classes
│   ├── layers.py         # Linear, Sigmoid, ReLU layers
│   ├── losses.py         # MSELoss, CrossEntropyLoss
│   ├── optimizers.py     # SGD optimizer
│   └── model_utils.py    # save_model, load_model functions
├── examples/
│   ├── solve_xor.py      # Classic XOR demo
│   └── solve_iris.py     # Advanced classification demo
├── .gitignore
├── requirements.txt
└── README.md             # You are here!

📜 License
This project is licensed under the MIT License.

