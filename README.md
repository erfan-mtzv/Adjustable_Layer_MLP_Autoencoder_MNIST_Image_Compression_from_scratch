# Adjustable-Layer Multilayer Perceptron Autoencoder for MNIST Image Compression

This repository contains a custom implementation of multilayer perceptron (MLP) autoencoders with adjustable layers and neurons for compressing and reconstructing MNIST handwritten digit images. The project was developed entirely from scratch, without the use of deep learning frameworks, for educational and experimental purposes.

---

## Project Overview

This project focuses on:
1. **Handwritten Digit Image Compression**:
   - Utilizing MLPs and autoencoders to reduce and reconstruct grayscale MNIST images.
2. **Customizable Architectures**:
   - Dynamic MLP architectures with configurable layer and neuron counts to fit different compression tasks.
3. **Performance Evaluation**:
   - Evaluating the model using metrics such as **MSE (Mean Squared Error)**.
4. **Optimization Techniques**:
   - Incorporating momentum-based backpropagation and early stopping mechanisms to enhance training efficiency and model performance.

---

## Key Features

- **Customizable Model Architecture**:
  - Configurable layers, neurons, and activation functions.
  - Example: `layer_dims = [784, 256, 64, 256, 784]` for compression and reconstruction.
- **Custom Backpropagation Implementation**:
  - Includes gradient calculations and weight/bias updates.
  - Supports early stopping based on validation loss to prevent overfitting.
- **Flexible Dataset Handling**:
  - Allows dynamic splitting of the MNIST dataset into training, validation, and test sets.
  - Uses a small subset of MNIST (e.g., 200 training samples) for rapid prototyping.
- **Performance Visualization**:
  - Plots validation loss (MSE) over epochs.
  - Visualizes input images, compressed representations, and reconstructed outputs.

---

## Algorithm Workflow

1. **Model Initialization**:
   - Define layer dimensions, learning rate, and activation functions.
2. **Forward Propagation**:
   - Compute weighted sums and apply activation functions (e.g., sigmoid) to generate outputs.
3. **Backpropagation**:
   - Compute gradients and update weights and biases.
4. **Training**:
   - Iterate through epochs until a stopping condition (e.g., MSE threshold) is met.
5. **Evaluation**:
   - Calculate MSE and visualize reconstructed images.

---

## Code Description

### **Initialization**:
The `MLP` class initializes weights and biases for each layer based on specified dimensions.

Example:
```python
layer_dims = [784, 256, 64, 256, 784]  # Input -> Hidden -> Bottleneck -> Hidden -> Output
model = MLP(alpha=0.01, layer_dims=layer_dims)
```

### **Training**:
The `train` function handles training, with early stopping based on:
- Maximum number of epochs.
- Minimum training MSE threshold.
- Patience for validation MSE improvement.

Example:
```python
model.train(X_train, X_val, max_epoc=100, epsilon=1e-4, patience=5)
```

### **Feedforward**:
The `feedforward` method calculates layer outputs for prediction or backpropagation.

Example:
```python
y_pred, layers_inputs, weighted_sums = model.feedforward(x, model.weights_list, model.biases_list)
```

### **Backpropagation**:
The `backpropagation` method computes gradients and updates weights and biases.

Example:
```python
weights, biases = model.backpropagation(weights, biases, inputs, weighted_sums, y_pred)
```

### **Evaluation and Plotting**:
Evaluate performance and visualize results:
```python
model.PLOT(epoc, MSE_val_list)  # Plot validation loss over epochs
```

---

## Dataset

- **MNIST Dataset (test.csv file)**:
  - Contains 10,000 handwritten digit images, preprocessed into training, validation, and test sets.
  - **Training**: 200 samples.
  - **Validation**: 50 samples.
  - **Test**: 65 samples.

---

## Example Results

### Reconstruction Example:
| ![Original](mnist.png) | ![Reconstructed](pred.png) |

### Loss Curve:
Example of validation MSE decreasing over epochs:
![Loss Curve](diag.png)

---
