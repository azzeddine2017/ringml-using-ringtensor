# Getting Started with RingML

Welcome to RingML! This guide will help you get started with building deep learning models using the Ring programming language.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Your First Neural Network](#your-first-neural-network)
- [Understanding the Basics](#understanding-the-basics)
- [Next Steps](#next-steps)
- [Common Pitfalls](#common-pitfalls)

## Prerequisites

Before you begin, ensure you have:

- **Ring Programming Language** (v1.25 or higher) - [Download](http://ring-lang.net)
- **Basic understanding** of neural networks (recommended but not required)
- **Text editor** or IDE for Ring development

## Installation

### Step 1: Install RingML Package

```bash
ringpm install ringml-using-ringtensor from Azzeddine2017
```

### Step 2: Verify RingTensor Extension

The RingML library depends on the RingTensor C extension. Ensure the appropriate library file is in your path:

- **Windows**: `ring_tensor.dll`
- **Linux/macOS**: `libring_tensor.so`

### Step 3: Test Installation

Create a file named `test_install.ring`:

```ring
load "ringml.ring"

func main
    see "RingML loaded successfully!" + nl
    
    # Create a simple tensor
    oTensor = new Tensor(2, 3)
    oTensor.setVal(1, 1, 5.0)
    see "Tensor created: " + oTensor.getVal(1, 1) + nl
```

Run it:
```bash
ring test_install.ring
```

If you see the success message, you're ready to go!

## Your First Neural Network

Let's build a simple neural network to solve the XOR problem - the "Hello World" of deep learning.

### Step 1: Create the Data

Create a file named `my_first_nn.ring`:

```ring
load "ringml.ring"
load "stdlib.ring"

func main
    # XOR Dataset
    # Input: [0,0], [0,1], [1,0], [1,1]
    # Output: [0], [1], [1], [0]
    
    aInputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    aTargets = [
        [0],
        [1],
        [1],
        [0]
    ]
    
    # Convert to tensors
    aDataset = []
    for i = 1 to len(aInputs)
        # Create input tensor
        oInput = new Tensor(1, 2)
        oInput.setVal(1, 1, aInputs[i][1])
        oInput.setVal(1, 2, aInputs[i][2])
        
        # Create target tensor
        oTarget = new Tensor(1, 1)
        oTarget.setVal(1, 1, aTargets[i][1])
        
        add(aDataset, [oInput, oTarget])
    next
    
    trainXOR(aDataset)

func trainXOR aDataset
    # Build the model
    oModel = new Sequential
    
    # Layer 1: Input (2) -> Hidden (4)
    oModel.add(new Dense(2, 4))
    oModel.add(new Tanh)
    
    # Layer 2: Hidden (4) -> Output (1)
    oModel.add(new Dense(4, 1))
    oModel.add(new Sigmoid)
    
    # Display model architecture
    see "Model Architecture:" + nl
    oModel.summary()
    see nl
    
    # Training setup
    oCriterion = new MSELoss
    oOptimizer = new SGD(0.1)  # Learning rate = 0.1
    nEpochs = 1000
    
    # Enable training mode
    oModel.train()
    
    # Training loop
    see "Training..." + nl
    for epoch = 1 to nEpochs
        epochLoss = 0
        
        # Train on each sample
        for i = 1 to len(aDataset)
            oInput = aDataset[i][1]
            oTarget = aDataset[i][2]
            
            # Forward pass
            oPred = oModel.forward(oInput)
            
            # Calculate loss
            loss = oCriterion.forward(oPred, oTarget)
            
            # Backward pass
            oGrad = oCriterion.backward(oPred, oTarget)
            oModel.backward(oGrad)
            
            # Update weights
            oModel.update_weights(oOptimizer)
            
            epochLoss += loss
        next
        
        # Print progress every 100 epochs
        if epoch % 100 = 0
            see "Epoch " + epoch + " - Loss: " + (epochLoss / len(aDataset)) + nl
        ok
    next
    
    # Test the trained model
    see nl + "Testing the trained model:" + nl
    oModel.evaluate()  # Switch to evaluation mode
    
    for i = 1 to len(aDataset)
        oInput = aDataset[i][1]
        oTarget = aDataset[i][2]
        oPred = oModel.forward(oInput)
        
        see "Input: [" + oInput.getVal(1,1) + ", " + oInput.getVal(1,2) + "] "
        see "-> Predicted: " + oPred.getVal(1,1) + " "
        see "(Target: " + oTarget.getVal(1,1) + ")" + nl
    next
    
    # Save the model
    oModel.saveWeights("xor_model.rdata")
    see nl + "Model saved to xor_model.rdata" + nl
```

### Step 2: Run the Network

```bash
ring my_first_nn.ring
```

You should see the loss decreasing over epochs, and final predictions close to the targets!

## Understanding the Basics

### 1. Tensors

Tensors are the fundamental data structure in RingML. They represent multi-dimensional arrays.

```ring
# Create a 2D tensor (2 rows, 3 columns)
oTensor = new Tensor(2, 3)

# Set values
oTensor.setVal(1, 1, 5.0)  # Row 1, Column 1 = 5.0

# Get values
value = oTensor.getVal(1, 1)  # Returns 5.0

# Get shape
aShape = oTensor.shape()  # Returns [2, 3]
```

### 2. Layers

Layers are the building blocks of neural networks.

**Dense Layer** (Fully Connected):
```ring
# Dense(input_size, output_size)
oLayer = new Dense(10, 5)  # 10 inputs -> 5 outputs
```

**Activation Functions**:
```ring
oTanh = new Tanh       # Tanh activation
oSigmoid = new Sigmoid # Sigmoid activation
oReLU = new ReLU       # ReLU activation
oGELU = new GELU       # GELU activation
```

**Regularization**:
```ring
oDropout = new Dropout(0.2)  # Drop 20% of neurons during training
```

### 3. Models

The `Sequential` model stacks layers in sequence.

```ring
oModel = new Sequential

# Add layers
oModel.add(new Dense(10, 64))
oModel.add(new Tanh)
oModel.add(new Dropout(0.2))
oModel.add(new Dense(64, 3))
oModel.add(new Softmax)

# View architecture
oModel.summary()
```

### 4. Loss Functions

Loss functions measure how well your model is performing.

```ring
# For regression
oMSE = new MSELoss

# For classification
oCrossEntropy = new CrossEntropyLoss
```

### 5. Optimizers

Optimizers update the model weights based on gradients.

```ring
# Stochastic Gradient Descent
oSGD = new SGD(0.01)  # Learning rate = 0.01

# Adam Optimizer (recommended for most cases)
oAdam = new Adam(0.001)  # Learning rate = 0.001
```

### 6. Training Loop

The typical training loop follows this pattern:

```ring
oModel.train()  # Enable training mode

for epoch = 1 to nEpochs
    for each batch
        # 1. Forward pass
        oPred = oModel.forward(oInput)
        
        # 2. Calculate loss
        loss = oCriterion.forward(oPred, oTarget)
        
        # 3. Backward pass
        oGrad = oCriterion.backward(oPred, oTarget)
        oModel.backward(oGrad)
        
        # 4. Update weights
        oModel.update_weights(oOptimizer)
    next
next

oModel.evaluate()  # Switch to evaluation mode
```

## Next Steps

Now that you've built your first neural network, explore these topics:

### 1. Work with Real Datasets

Learn to use the data loading utilities:
- [Data Loading Guide](API_REFERENCE.md#data-loading)
- Example: `samples/UsingRingML/loader_demo.ring`

### 2. Build Classification Models

Try multi-class classification:
- Example: `samples/UsingRingML/classify_demo.ring`
- Example: `samples/UsingRingML/mnist/mnist_train.ring`

### 3. Advanced Architectures

Explore transformer models:
- Example: `samples/UsingRingML/train_translate_bidir/`
- [Transformer Tutorial](TUTORIALS.md#tutorial-5-transformer-models)

### 4. Save and Load Models

Learn model serialization:
- Example: `samples/UsingRingML/save_load_demo.ring`
- [Serialization Guide](API_REFERENCE.md#serialization)

### 5. Optimize Performance

Improve training speed:
- [Performance Guide](PERFORMANCE.md)
- Use Graph Mode for faster execution
- Enable GPU acceleration

## Common Pitfalls

### 1. Shape Mismatches

**Problem**: Tensor shapes don't match between layers.

```ring
# Wrong: Input is 10, but Dense expects 5
oModel.add(new Dense(5, 3))
oPred = oModel.forward(oInput)  # oInput has 10 features - ERROR!
```

**Solution**: Ensure layer dimensions match:
```ring
oModel.add(new Dense(10, 3))  # First layer matches input size
```

### 2. Forgetting Training/Evaluation Mode

**Problem**: Dropout behaves incorrectly during testing.

```ring
# Wrong: Still in training mode during evaluation
oModel.train()
# ... training ...
oPred = oModel.forward(oTestInput)  # Dropout still active!
```

**Solution**: Switch to evaluation mode:
```ring
oModel.evaluate()  # Disable dropout
oPred = oModel.forward(oTestInput)
```

### 3. Not Normalizing Data

**Problem**: Input data has large values causing training instability.

**Solution**: Normalize inputs to [0, 1] or standardize to mean=0, std=1:
```ring
# Simple normalization
normalized = (value - min) / (max - min)
```

### 4. Learning Rate Too High/Low

**Problem**: 
- Too high: Loss explodes or oscillates
- Too low: Training is extremely slow

**Solution**: Start with common values:
- SGD: 0.01 - 0.1
- Adam: 0.001 - 0.0001

### 5. RingTensor DLL Not Found

**Problem**: Error loading `ring_tensor.dll` or `libring_tensor.so`

**Solution**: 
- Ensure RingTensor is installed: `ringpm install ringtensor`
- Add the library path to your system PATH
- On Windows, copy `ring_tensor.dll` to your project directory

## Getting Help

- **Documentation**: Check the [API Reference](API_REFERENCE.md)
- **Examples**: Browse `samples/UsingRingML/` directory
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: Report bugs on GitHub
- **Contact**: azzeddine.remmal@gmail.com

---

**Congratulations!** You've completed the Getting Started guide. Happy coding with RingML! 🎉
