import numpy as np
from minitorch import Sequential, Sigmoid, Linear, ReLU, CrossEntropyLoss, SGD
from minitorch import save_model, load_model
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    """Loads and prepares the Iris dataset."""
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int32) # Our loss fn expects integer labels
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def main():
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test = load_data()
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # 2. Define the Model
    # 4 features -> 16 hidden neurons -> 3 output classes
    model = Sequential(
        Linear(n_features, 16),
        ReLU(),
        Linear(16, n_classes)
        # Note: No Softmax layer! The loss function handles it.
    )

    # 3. Define Loss and Optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.5)

    # 4. The Training Loop (with mini-batching)
    epochs = 5000
    batch_size = 16
    
    print("Training MiniTorch model on Iris dataset...")
    for epoch in range(epochs):
        # Shuffle data each epoch
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]
        
        epoch_loss = 0.0
        
        for i in range(0, X_train.shape[0], batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            
            # --- The 5 steps of training ---
            optimizer.zero_grad()
            y_hat_logits = model(X_batch)
            loss = loss_fn(y_hat_logits, y_batch)
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
            
            epoch_loss += loss * X_batch.shape[0]
        
        epoch_loss /= X_train.shape[0]
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    print("Training complete.")

    # 5. Save the trained model
    model_path = "iris_model.npz"
    save_model(model, model_path)

    # 6. Create a new, untrained model and load the weights
    print("\nLoading model for testing...")
    new_model = Sequential(
        Linear(n_features, 16),
        ReLU(),
        Linear(16, n_classes)
    )
    load_model(new_model, model_path)

    # 7. Test the loaded model
    y_pred_logits = new_model(X_test)
    y_pred_classes = np.argmax(y_pred_logits, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nTest Accuracy (from loaded model): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
