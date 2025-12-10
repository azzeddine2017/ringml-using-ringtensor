
# RingML: Deep Learning Library for Ring Language

RingML is a lightweight, modular Deep Learning framework built from scratch in Ring. It leverages the **FastPro** extension for accelerated matrix operations, providing a PyTorch-like API for building and training Neural Networks.

## 🚀 Features

*   **Tensor Engine**: Wraps `FastPro` C-extension for matrix math (MatMul, Transpose, etc.).
*   **Automatic Differentiation**: Implements full Backpropagation (Backward Pass).
*   **Layers**:
    *   `Dense` (Fully Connected).
    *   `Sigmoid`, `ReLU`, `Tanh` (Activations).
*   **Optimization**:
    *   `SGD` (Stochastic Gradient Descent).
    *   `MSELoss` (Mean Squared Error).
*   **API**:
    *   `Sequential` model container for easy stacking of layers.

## 📦 Project Structure

```text
src/
├── core/
│   └── tensor.ring       # The mathematical heart (Matrix ops)
├── layers/
│   ├── layer.ring        # Abstract Base Class
│   ├── dense.ring        # Fully Connected Layer
│   └── activation.ring   # Activation Functions
├── loss/
│   └── mse.ring          # Mean Squared Error
├── model/
│   └── sequential.ring   # Model Container
├── optim/
│   └── sgd.ring          # Optimizer
└── ringml.ring           # Main Loader
examples/
└── xor_train.ring        # Proof-of-concept (XOR Problem)
```
## ⚡ Quick Start

### Solving XOR Problem

```Ring
load "src/ringml.ring"

# 1. Prepare Data
inputs  = new Tensor(4, 2) { aData = [[0,0], [0,1], [1,0], [1,1]] }
targets = new Tensor(4, 1) { aData = [[0],   [1],   [1],   [0]]   }

# 2. Build Model
model = new Sequential
model.add(new Dense(2, 4)) # Input: 2, Hidden: 4
model.add(new Sigmoid)
model.add(new Dense(4, 1)) # Output: 1
model.add(new Sigmoid)

# 3. Setup Training
optimizer = new SGD(0.5)   # Learning Rate
criterion = new MSELoss

# 4. Training Loop
for epoch = 1 to 5000
    # Forward
    preds = model.forward(inputs)
    
    # Backward
    lossGrad = criterion.backward(preds, targets)
    model.backward(lossGrad)
    
    # Update
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next
```
# 5. Predict
model.forward(inputs).print()

**🛠 Dependencies**
*  Ring Language (1.24 or later)
*  FastPro Extension (Must be loaded/dll available)

## 📝 License
Open Source. 

