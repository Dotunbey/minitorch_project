import numpy as np
from minitorch import Sequential, Linear, Sigmoid, MSELoss, SGD

# 1. Define the Problem (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float64)

# 2. Define the Model using your library
# A 2-layer neural network: 2 inputs -> 3 hidden neurons -> 1 output
model = Sequential(
    Linear(2, 3),
    Sigmoid(),
    Linear(3, 1),
    Sigmoid()
)

# 3. Define the Loss and Optimizer
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 4. The Training Loop
epochs = 10000

print("Training MiniTorch model on XOR problem...")
for i in range(epochs):
    # --- The 5 steps of training ---
    
    # 1. Reset gradients
    optimizer.zero_grad()
    
    # 2. Forward pass
    y_hat = model(X)  # model.forward(X) is called
    
    # 3. Calculate loss
    loss = loss_fn(y_hat, y) # loss_fn.forward(y_hat, y) is called
    
    # 4. Backward pass
    # Get the first gradient from the loss function
    grad = loss_fn.backward() 
    # Propagate the gradient back through the model
    model.backward(grad)
    
    # 5. Update weights
    optimizer.step()
    
    # --- End of steps ---
    
    if (i + 1) % 1000 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss:.6f}")

# 5. Test the trained model
print("\nTraining complete.")
print("Final predictions:")
y_hat = model(X)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Target: {y[i][0]} -> Predicted: {y_hat[i][0]:.4f}")
