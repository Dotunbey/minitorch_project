import numpy as np
from .module import Module

def save_model(model: Module, filepath: str) -> None:
    """
    Saves all model parameters to a compressed .npz file.
    """
    # Create a dictionary of parameters to save
    params_to_save = {}
    for i, (param, grad) in enumerate(model.parameters()):
        # We give each parameter a unique name like "param_0", "param_1"
        params_to_save[f"param_{i}"] = param

    # Save this dictionary to a compressed .npz file
    np.savez(filepath, **params_to_save)
    print(f"Model saved to {filepath}")

def load_model(model: Module, filepath: str) -> Module:
    """
    Loads parameters from a .npz file into an existing model.
    
    Note: The model architecture must be *identical* to the saved one.
    """
    # Load the saved weights from the file
    try:
        loaded_params = np.load(filepath)
    except FileNotFoundError:
        print(f"Error: No model file found at {filepath}")
        return model
        
    # Create a generator for the model's parameters
    params_generator = model.parameters()
    
    # Check if the number of parameters matches
    if len(loaded_params.files) != len(list(model.parameters())):
        raise ValueError("Model architecture mismatch: Number of parameters differs.")

    # Iterate and copy the saved weights into the model
    # This relies on model.parameters() and the .npz file
    # having the parameters in the *exact same order*.
    for i in range(len(loaded_params.files)):
        param, grad = next(params_generator)
        
        # Get the saved weight by its key
        saved_param = loaded_params[f"param_{i}"]
        
        # Check if shapes match
        if param.shape != saved_param.shape:
            raise ValueError(
                f"Model architecture mismatch: Parameter {i} has shape "
                f"{param.shape} but saved param has shape {saved_param.shape}."
            )
        
        # Copy the saved data into the model's parameter array
        np.copyto(param, saved_param)

    print(f"Model weights loaded successfully from {filepath}.")
    return model
