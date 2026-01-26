# 🧠 RingML: Deep Learning & Transformers for Ring

**RingML** is a high-performance, object-oriented Deep Learning framework built for the Ring programming language. It is powered by **RingTensor**, a custom C-extension designed to provide fast, double-precision matrix operations, fused optimizer kernels, and **Flash Attention** mechanisms.

The library offers a PyTorch-like API, adhering to **Jacob's Law** by providing a familiar interface. It has evolved from simple MLPs to supporting state-of-the-art **Transformer Architectures (GPT/BERT)**.

**Current Version:** 1.2.1

## 📦 Installation

```bash
ringpm install ringml-using-ringtensor from Azzeddine2017
```
> **Requirement:** Ensure `ring_tensor.dll` (Windows) or `libring_tensor.so` (Linux/macOS) is in your execution path.

## 🛠️ Tech Stack & Architecture

- **Core Language:** Ring (v1.25+)
- **Engine:** RingTensor Extension (C-based Zero-Copy Engine).
- **Execution Modes:**
    - **Eager:** Immediate execution (Standard).
    - **Graph Mode:** JIT-compiled computation graph for zero-latency training.
- **Hardware Acceleration:**
    - **CPU:** Multi-Core OpenMP acceleration.
    - **GPU:** OpenCL support (Intel HD, NVIDIA, AMD) via Hybrid Dispatcher.
- **Tensors:** Supports up to 4 Dimensions (Batch, Heads, Sequence, Dimension).

### Why RingML?
- **Speed:** Critical operations (MatMul, Softmax, Attention, LayerNorm, GELU) are executed in optimized C kernels.
- **Memory Efficiency:** Uses Memory-Resident Managed Pointers. No data marshalling overhead.
- **Fused Kernels:** Optimizers (Adam, SGD) update weights directly in C memory.
- **Production Ready:** Includes Binary Serialization (Save/Load) and Quantization (FP32).

---

## 📊 Legacy Example: Simple MLP (Classic Mode)

For standard classification tasks (like XOR or MNIST), the Sequential API remains the perfect tool:

### 1. Data Preparation
Using `UniversalDataset` to handle CSV loading and splitting.

```ring
load "ringml.ring" 
load "stdlib.ring"

# Define how to process a single row
class MyDataset from UniversalDataset
    func rowToTensor row
        # Convert row list to Tensors (Input, Target)
        oIn = new Tensor(1, 2)
        oIn.setVal(1, 1, number(row[1])) 
        oIn.setVal(1, 2, number(row[2])) 
        
        oOut = new Tensor(1, 1)
        oOut.setVal(1, 1, number(row[3]))
        
        return [oIn, oOut]

# Load and Prepare
data = new MyDataset("data.csv")
data.setHeader(true)        
data.setShuffle(true)       
data.setSplit(0.2)          
data.loadData()             

# Create Loaders
trainLoader = new DataLoader(data.getTrainDataset(), 32)
testLoader  = new DataLoader(data.getTestDataset(), 32)
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

### 3. Training with Adam
The training loop handles Forward pass, Backward pass, and Optimization.

```ring
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 50

# Enable Training Mode (Activates Dropout)
model.train() 

for epoch = 1 to nEpochs
    epochLoss = 0
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b)
        inputs = batch[1] 
        targets = batch[2]
        
        # Forward & Loss
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        
        # Backward & Update
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        # Gradient Clipping (Optional)
        aGrads = model.get_gradients()
        tensor_clip_global_norm(aGrads, 1.0)
        
        model.update_weights(optimizer)
        
        epochLoss += loss
    next
    see "Epoch " + epoch + " Loss: " + (epochLoss / trainLoader.nBatches) + nl
next
```

### 4. Saving & Loading
Switch to evaluation mode to disable Dropout, then save using binary serialization.

```ring
model.evaluate() 
model.saveWeights("mymodel.rdata")
see "Model Saved." + nl

# --- Loading ---
model2 = new Sequential
# ... define same structure ...
model2.loadWeights("mymodel.rdata")
```

---

## 📂 Included Projects

1.  **Project train translate bidir Adam2:** A complete Transformer-based translation engine (English <-> Arabic) and code generator.
2.  **Chess End-Game:** Multi-class classification (18 classes).
3.  **XOR:** The "Hello World" of Neural Networks.
4.  **mnist_train.ring:** Computer Vision example for digit recognition.

## 📝 License

Open Source under **MIT License**.

**Author:** Azzeddine Remmal.