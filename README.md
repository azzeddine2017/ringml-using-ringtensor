# 🧠 RingML: Deep Learning Library for Ring

RingML is a high-performance, object-oriented Deep Learning framework built for the **Ring programming language**. It is powered by **RingTensor**, a custom C-extension designed specifically to provide fast, double-precision matrix operations and fused optimizer kernels.

The library offers a **PyTorch-like API**, adhering to *Jacob's Law* by providing a familiar and intuitive interface for building Neural Networks.

**Current Version:** 1.0.0 (Stable - Powered by RingTensor)

## 📦 Installation

You can easily install the package via the Ring Package Manager:

```bash
ringpm install ringml-using-ringtensor from Azzeddine2017
```

> **Note:** Ensure the `ring_tensor.dll` (Windows) or `libring_tensor.so` (Linux/macOS) is in your execution path.

## 🛠️ Tech Stack & Architecture

- **Core Language:** Ring Programming Language (v1.24+)
- **Engine:** RingTensor Extension (Custom C-based DLL/SO).
- **Architecture:** Modular OOP (Tensor, Layers, Model, Optim, Data).
- **Math Backend:** Full C Acceleration.

Unlike previous versions that used slow loops for stability, **RingML** now uses the **RingTensor** C-extension for all critical operations (MatMul, Transpose, Element-wise Math, Activations).

- **Fused Kernels:** Optimizers (Adam, SGD) calculate updates inside C in a single pass for maximum speed.
- **Precision:** Full Double Precision (64-bit float) guaranteed across the pipeline.

## 🚀 Key Features

### 1. Core Engine (`src/core/`)
- **Tensor Class:** The mathematical heart of the library.
- **RingTensor Integration:** Direct calls to C functions for `tensor_matmul`, `tensor_add`, `tensor_sigmoid`, etc.
- **Stability:** Implements Numerically Stable Softmax (in C) to prevent NaN issues.

### 2. Neural Building Blocks (`src/layers/`)
- **Dense:** Fully Connected Layer with smart random weight initialization.
- **Activations:** ReLU, Sigmoid, Tanh, Softmax.
- **Regularization:** Dropout layer (with C-accelerated mask generation) to prevent overfitting.

### 3. Model & Optimization (`src/model/`, `src/optim/`)
- **Sequential:** Stack layers linearly. Includes `model.summary()` to visualize architecture.
- **Optimizers:**
  - **SGD** (Stochastic Gradient Descent).
  - **Adam** (Adaptive Moment Estimation) - Implemented via Fused Kernel in C.
- **Loss Functions:** `MSELoss` (Regression), `CrossEntropyLoss` (Classification).
- **Modes:** Support for `train()` and `evaluate()` modes.

### 4. Data Pipeline (`src/data/`)
- **DataSplitter:** Utility to shuffle and split raw data into Training/Testing sets.
- **DataLoader:** Efficient Mini-Batch processing to handle large datasets without memory overflow.
- **Lazy Loading:** Support for custom Datasets.

## ⚡ Quick Start Guide

### 1. Data Preparation
Use `DataSplitter` to handle raw CSV data and `DataLoader` for batching.

```ring
load "ringml.ring" # Or specific path inside the package
load "stdlib.ring"

# 1. Load Data
aRawData = [ [0,0,0], [0,1,1], [1,0,1], [1,1,0] ] # Example XOR data

# 2. Split (80% Train, 20% Test) with Shuffle
splitter = new DataSplitter
sets = splitter.splitData(aRawData, 0.2, true) 
trainData = sets[1]
testData  = sets[2]

# 3. Create Loader (Batch Size = 32)
dataset = new TensorDataset(trainData) 
loader  = new DataLoader(dataset, 32)
```

### 2. Building the Model
Construct a model using Tanh for hidden layers and Dropout for regularization.

```ring
model = new Sequential

# Input: 10 features -> Hidden: 64 neurons
model.add(new Dense(10, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2)) # Drop 20% of neurons during training

# Hidden: 64 -> Output: 3 classes
model.add(new Dense(64, 3)) 
model.add(new Softmax)

# View architecture
model.summary()
```

### 3. Advanced: Freezing Layers (Transfer Learning)
You can freeze specific layers to prevent them from updating during training.

```ring
model = new Sequential
l1 = new Dense(6, 32)
l1.freeze()  # <--- This layer's weights will NOT change
model.add(l1)
model.add(new Tanh)
model.add(new Dense(32, 1))
model.add(new Softmax)

model.summary() 
# You will see "Non-trainable params" count increase.
```

### 4. Training with Adam
The training loop handles Forward pass, Backward pass, and Optimization.

```ring
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 50

# Enable Training Mode (Activates Dropout)
model.train() 

for epoch = 1 to nEpochs
    epochLoss = 0
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b)
        inputs = batch[1] 
        targets = batch[2]
        
        # Forward & Loss
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        
        # Backward & Update
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        for layer in model.getLayers() optimizer.update(layer) next
        
        epochLoss += loss
    next
    see "Epoch " + epoch + " Loss: " + (epochLoss / loader.nBatches) + nl
next
```

### 5. Saving & Loading
Switch to evaluation mode to disable Dropout, then save.

```ring
model.evaluate() 
model.saveWeights("mymodel.rdata")
see "Model Saved." + nl

# --- Loading ---
model2 = new Sequential
# ... define same structure ...
model2.loadWeights("mymodel.rdata")
```

## 🧪 Included Examples

- **xor_train.ring:** Binary Classification (Hello World).
- **Chess_End_Game/:** A complete real-world project classifying chess game results (18 classes) from a CSV dataset.
- **mnist_train.ring:** Computer Vision example for digit recognition.

## 📝 License

Open Source under **MIT License**.

**Author:** Azzeddine Remmal.