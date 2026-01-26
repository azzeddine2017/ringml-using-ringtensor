# Troubleshooting Guide

Common issues and solutions when working with RingML.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Training Issues](#training-issues)
- [Model Issues](#model-issues)
- [Performance Issues](#performance-issues)
- [FAQ](#faq)

## Installation Issues

### Error: "Can't load library ring_tensor.dll"

**Symptoms**:
```
Error: Can't load library ring_tensor.dll
```

**Solutions**:

1. **Install RingTensor**:
   ```bash
   ringpm install ringtensor
   ```

2. **Add to PATH** (Windows):
   ```powershell
   $env:PATH += ";C:\ring\extensions"
   ```

3. **Copy DLL to project**:
   ```powershell
   Copy-Item "C:\ring\extensions\ring_tensor.dll" -Destination "."
   ```

4. **Linux/macOS**: Ensure `libring_tensor.so` is in `LD_LIBRARY_PATH`

### Error: "Package not found"

**Symptoms**:
```
Error: Package ringml-using-ringtensor not found
```

**Solutions**:

1. **Check spelling**:
   ```bash
   ringpm install ringml-using-ringtensor from Azzeddine2017
   ```

2. **Update RingPM**:
   ```bash
   ringpm update
   ```

3. **Manual installation**: Clone from GitHub and run `ring setup.ring`

## Runtime Errors

### Error: "Tensor shape mismatch"

**Symptoms**:
```
Error: Matrix dimensions incompatible for multiplication
Expected: (32, 512) × (512, 256)
Got: (32, 512) × (128, 256)
```

**Cause**: Layer input size doesn't match previous layer output.

**Solution**:

```ring
# Wrong
oModel.add(new Dense(512, 256))
oModel.add(new Dense(128, 64))  # Input should be 256, not 128!

# Correct
oModel.add(new Dense(512, 256))
oModel.add(new Dense(256, 64))  # Matches previous output
```

**Debugging tip**: Use `oModel.summary()` to verify architecture.

### Error: "File not found"

**Symptoms**:
```
Error: File not found -> data.csv
```

**Solutions**:

1. **Check path**:
   ```ring
   # Use absolute path
   oData = new MyDataset("C:/full/path/to/data.csv")
   ```

2. **Verify file exists**:
   ```ring
   if !fexists("data.csv")
       see "File does not exist!" + nl
   ok
   ```

3. **Check working directory**:
   ```ring
   see "Current directory: " + currentdir() + nl
   ```

### Error: "Architecture Mismatch" when loading weights

**Symptoms**:
```
Error: Architecture Mismatch
```

**Cause**: Model structure doesn't match saved weights.

**Solution**: Ensure exact same architecture:

```ring
# Saving
oModel = new Sequential
oModel.add(new Dense(10, 64))
oModel.add(new Tanh)
oModel.add(new Dense(64, 3))
oModel.saveWeights("model.rdata")

# Loading - MUST BE IDENTICAL
oModel2 = new Sequential
oModel2.add(new Dense(10, 64))  # Same dimensions
oModel2.add(new Tanh)           # Same activation
oModel2.add(new Dense(64, 3))   # Same output
oModel2.loadWeights("model.rdata")  # Now works
```

## Training Issues

### Loss is NaN or Inf

**Symptoms**:
```
Epoch 1 - Loss: 2.5
Epoch 2 - Loss: NaN
```

**Causes & Solutions**:

1. **Learning rate too high**:
   ```ring
   # Too high
   oOptimizer = new Adam(0.1)  # May cause explosion
   
   # Better
   oOptimizer = new Adam(0.001)  # Recommended
   ```

2. **Missing gradient clipping**:
   ```ring
   # Add clipping
   aGrads = oModel.get_gradients()
   tensor_clip_global_norm(aGrads, 1.0)
   oModel.update_weights(oOptimizer)
   ```

3. **Unnormalized data**:
   ```ring
   # Normalize inputs to [0, 1] or standardize
   normalized = (value - min) / (max - min)
   ```

4. **Division by zero in loss**:
   ```ring
   # Use CrossEntropy for classification, not MSE
   oCriterion = new CrossEntropyLoss  # Correct for classification
   ```

### Loss not decreasing

**Symptoms**:
```
Epoch 1 - Loss: 2.5
Epoch 10 - Loss: 2.5
Epoch 50 - Loss: 2.5
```

**Causes & Solutions**:

1. **Learning rate too low**:
   ```ring
   # Try higher learning rate
   oOptimizer = new Adam(0.01)  # Instead of 0.0001
   ```

2. **Model too simple**:
   ```ring
   # Add more capacity
   oModel.add(new Dense(10, 128))  # Instead of 64
   oModel.add(new ReLU)
   oModel.add(new Dense(128, 64))
   ```

3. **Wrong loss function**:
   ```ring
   # For classification, use CrossEntropy
   oCriterion = new CrossEntropyLoss
   
   # For regression, use MSE
   oCriterion = new MSELoss
   ```

4. **Forgot to call backward**:
   ```ring
   # Must include backward pass
   oPred = oModel.forward(oInput)
   loss = oCriterion.forward(oPred, oTarget)
   
   # Don't forget these!
   oGrad = oCriterion.backward(oPred, oTarget)
   oModel.backward(oGrad)
   oModel.update_weights(oOptimizer)
   ```

### Overfitting (train loss << test loss)

**Symptoms**:
```
Train Loss: 0.1
Test Loss: 2.5
```

**Solutions**:

1. **Add Dropout**:
   ```ring
   oModel.add(new Dense(128, 128))
   oModel.add(new ReLU)
   oModel.add(new Dropout(0.3))  # Drop 30%
   ```

2. **Reduce model size**:
   ```ring
   # Smaller model
   oModel.add(new Dense(128, 64))  # Instead of 128
   ```

3. **Get more data**:
   - Collect more training samples
   - Use data augmentation

4. **Early stopping**:
   ```ring
   # Monitor test loss, stop when it starts increasing
   if testLoss > bestTestLoss
       see "Early stopping at epoch " + epoch + nl
       break
   ok
   ```

## Model Issues

### Dropout active during testing

**Symptoms**: Inconsistent predictions on same input.

**Cause**: Model still in training mode.

**Solution**:
```ring
# Before testing/inference
oModel.evaluate()  # Disable dropout

# Predictions are now deterministic
oPred = oModel.forward(oTestInput)
```

### Model predictions all the same

**Symptoms**: All outputs are identical or very similar.

**Causes & Solutions**:

1. **Dead ReLU**:
   ```ring
   # Try different activation
   oModel.add(new GELU)  # Instead of ReLU
   ```

2. **Weights not initialized**:
   - RingML automatically initializes weights (He initialization)
   - If using custom layers, ensure proper initialization

3. **Learning rate too high**:
   ```ring
   # Reduce learning rate
   oOptimizer = new Adam(0.0001)
   ```

## Performance Issues

### Training very slow

**See**: [Performance Guide](PERFORMANCE.md)

**Quick fixes**:

1. **Use Graph Mode**:
   ```ring
   oInput.setGraphMode(true)
   oGraph = oModel.compile(oInput)
   ```

2. **Increase batch size**:
   ```ring
   oLoader = new DataLoader(dataset, 128)  # Instead of 32
   ```

3. **Use Adam optimizer**:
   ```ring
   oOptimizer = new Adam(0.001)  # Faster than SGD
   ```

### Out of memory

**Symptoms**:
```
Error: Out of memory
```

**Solutions**:

1. **Reduce batch size**:
   ```ring
   oLoader = new DataLoader(dataset, 16)  # Smaller batches
   ```

2. **Clear unused tensors**:
   ```ring
   oTensor = NULL
   callgc()
   ```

3. **Process data in chunks**:
   ```ring
   # Don't load all data at once
   # Use DataLoader instead
   ```

4. **Reduce model size**:
   ```ring
   # Smaller dimensions
   oModel.add(new Dense(512, 256))  # Instead of 1024
   ```

## FAQ

### Q: How do I save/load a model?

**A**: Use `saveWeights()` and `loadWeights()`:

```ring
# Save
oModel.saveWeights("model.rdata")

# Load (must have same architecture)
oModel2 = new Sequential
# ... build same architecture ...
oModel2.loadWeights("model.rdata")
```

### Q: How do I use GPU acceleration?

**A**: GPU is automatic for large operations. Ensure:
- OpenCL drivers installed
- RingTensor compiled with OpenCL support
- Tensors are large enough (> 1024×1024)

### Q: Can I use custom datasets?

**A**: Yes, inherit from `UniversalDataset`:

```ring
class MyDataset from UniversalDataset
    func rowToTensor aRow
        # Convert row to [input, target]
        return [oInput, oTarget]
```

### Q: How do I implement custom layers?

**A**: Inherit from `Layer`:

```ring
class MyLayer from Layer
    func forward oInput
        # Implement forward pass
        return oOutput
    
    func backward oGradOutput
        # Implement backward pass
        return oGradInput
```

### Q: What's the difference between train() and evaluate()?

**A**:
- `train()`: Enables Dropout, BatchNorm in training mode
- `evaluate()`: Disables Dropout, BatchNorm in inference mode

Always call `evaluate()` before testing!

### Q: How do I debug shape mismatches?

**A**: Use `summary()` and print shapes:

```ring
oModel.summary()  # See all layer shapes

# Print tensor shapes
see "Input shape: " + oInput.shape()[1] + " × " + oInput.shape()[2] + nl
```

### Q: Can I use RingML for regression?

**A**: Yes, use MSELoss:

```ring
oModel.add(new Dense(10, 1))  # Single output
# No Softmax for regression!

oCriterion = new MSELoss
```

### Q: How do I implement early stopping?

**A**:

```ring
bestLoss = 999999
patience = 10
counter = 0

for epoch = 1 to nEpochs
    # ... training ...
    
    # Evaluate on test set
    testLoss = evaluateModel(oModel, testLoader)
    
    if testLoss < bestLoss
        bestLoss = testLoss
        counter = 0
        oModel.saveWeights("best_model.rdata")
    else
        counter++
        if counter >= patience
            see "Early stopping!" + nl
            break
        ok
    ok
next
```

### Q: How do I visualize training progress?

**A**: Use the built-in visualizer:

```ring
load "ringml.ring"

# Training loop
aLosses = []
for epoch = 1 to nEpochs
    loss = trainEpoch()
    aLosses + loss
    
    # Plot every 10 epochs
    if epoch % 10 = 0
        plotLosses(aLosses)
    ok
next
```

## Getting Help

If your issue isn't covered here:

1. **Check documentation**:
   - [Getting Started](GETTING_STARTED.md)
   - [API Reference](API_REFERENCE.md)
   - [Architecture](ARCHITECTURE.md)

2. **Search examples**:
   - Browse `samples/UsingRingML/` directory
   - Check test files in `libraries/ringml/tests/`

3. **Report bug**:
   - Open GitHub issue with:
     - Minimal code to reproduce
     - Error message
     - Ring version, OS, RingTensor version

4. **Contact**:
   - Email: azzeddine.remmal@gmail.com

---

**Tip**: Enable verbose logging to see what's happening:

```ring
# Add debug prints
see "Forward pass..." + nl
oPred = oModel.forward(oInput)
see "Prediction shape: " + oPred.shape()[1] + " × " + oPred.shape()[2] + nl
```
