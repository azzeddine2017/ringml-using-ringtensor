# Performance Optimization Guide

This guide covers techniques to optimize RingML model performance for both training and inference.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Execution Modes](#execution-modes)
- [Memory Optimization](#memory-optimization)
- [GPU Acceleration](#gpu-acceleration)
- [Batch Size Tuning](#batch-size-tuning)
- [Data Loading](#data-loading)
- [Model Architecture](#model-architecture)
- [Benchmarking](#benchmarking)

## Quick Wins

### 1. Use Graph Mode for Production

**Impact**: 2-3x speedup for large models

```ring
# Create virtual input
oInput = new Tensor(32, 512)
oInput.setGraphMode(true)

# Compile model
oOutput = oModel.compile(oInput)

# Fast execution
oRealInput = new Tensor(32, 512)
# ... load data ...
oResult = graph_run(oOutput.pGraph, oRealInput.pData)
```

### 2. Use Adam Optimizer

**Impact**: Faster convergence, fewer epochs needed

```ring
# Instead of SGD
oOptimizer = new Adam(0.001)  # Recommended
```

### 3. Increase Batch Size

**Impact**: Better GPU utilization, faster training

```ring
# Increase from 32 to 64 or 128 if memory allows
trainLoader = new DataLoader(dataset, 128)
```

### 4. Enable Gradient Clipping

**Impact**: Prevents gradient explosions, stabilizes training

```ring
aGrads = oModel.get_gradients()
tensor_clip_global_norm(aGrads, 1.0)
```

## Execution Modes

### Eager Mode (Default)

**When to use**:
- Development and debugging
- Dynamic architectures
- Small models

**Characteristics**:
- ✅ Easy to debug
- ✅ Flexible
- ❌ Slower (Ring interpreter overhead)

### Graph Mode (Optimized)

**When to use**:
- Production inference
- Large models (Transformers)
- Fixed architecture

**Characteristics**:
- ✅ 2-3x faster
- ✅ Optimized memory
- ❌ Fixed architecture

**Implementation**:

```ring
# Training in eager mode
oModel.train()
for epoch = 1 to nEpochs
    # ... training loop ...
next

# Switch to graph mode for inference
oModel.evaluate()

oInput = new Tensor(nBatch, nFeatures)
oInput.setGraphMode(true)
oGraphOutput = oModel.compile(oInput)

# Fast inference
for batch in testData
    oRealInput = # ... load batch ...
    oPred = graph_run(oGraphOutput.pGraph, oRealInput.pData)
next
```

## Memory Optimization

### 1. Use Appropriate Data Types

RingML uses double precision (FP64) by default. This is accurate but memory-intensive.

**Memory per parameter**:
- FP64: 8 bytes
- FP32: 4 bytes (future support)

### 2. Clear Unused Tensors

```ring
# After processing
oTensor = NULL
callgc()  # Force garbage collection
```

### 3. Batch Processing

Process data in chunks instead of loading everything:

```ring
# Bad: Load all data at once
aAllData = loadEntireDataset()  # May cause OOM

# Good: Use DataLoader
oLoader = new DataLoader(dataset, 32)
for b = 1 to oLoader.nBatches
    batch = oLoader.getBatch(b)
    # Process batch
next
```

### 4. Model Pruning

Remove unnecessary layers or reduce dimensions:

```ring
# Before: Large model
oModel.add(new Dense(512, 512))
oModel.add(new Dense(512, 512))

# After: Smaller model
oModel.add(new Dense(512, 256))
oModel.add(new Dense(256, 128))
```

## GPU Acceleration

### Setup

RingML supports GPU acceleration via OpenCL (Intel HD, NVIDIA, AMD).

**Requirements**:
- OpenCL drivers installed
- RingTensor compiled with OpenCL support

### Automatic Dispatch

RingTensor automatically uses GPU for large operations:

```ring
# Large matrix multiplication automatically uses GPU
oLarge = new Tensor(2048, 2048)
oResult = oLarge.matmul(oLarge)  # Dispatched to GPU
```

**Threshold**: Operations on tensors > 1024×1024 typically use GPU.

### Verify GPU Usage

Check RingTensor logs:
```
[RingTensor] Using OpenCL device: NVIDIA GeForce RTX 3080
[RingTensor] MatMul dispatched to GPU
```

## Batch Size Tuning

### Finding Optimal Batch Size

**Trade-offs**:
- **Larger batches**: Faster training, better GPU utilization, less noise
- **Smaller batches**: More updates, better generalization, less memory

**Recommended approach**:

```ring
# Start with power of 2
aBatchSizes = [16, 32, 64, 128, 256]

for nBatch in aBatchSizes
    see "Testing batch size: " + nBatch + nl
    
    oLoader = new DataLoader(dataset, nBatch)
    
    t1 = clock()
    # Train for 1 epoch
    for b = 1 to oLoader.nBatches
        # ... training ...
    next
    t2 = clock()
    
    see "Time: " + ((t2-t1)/clockspersecond()) + "s" + nl
next
```

**Guidelines**:
- **Small models (< 1M params)**: 32-64
- **Medium models (1M-10M params)**: 64-128
- **Large models (> 10M params)**: 128-256
- **Transformers**: 32-64 (due to attention memory)

## Data Loading

### 1. Use UniversalDataset

Efficient CSV/JSON loading with automatic splitting:

```ring
oData = new MyDataset("large_file.csv")
oData.setHeader(true)
oData.setShuffle(true)
oData.setSplit(true)
oData.loadData()  # Optimized C-level file reading
```

### 2. Shuffle Data

Improves generalization:

```ring
oData.setShuffle(true)
```

### 3. Pre-process Data

Normalize/standardize before training:

```ring
class MyDataset from UniversalDataset
    func rowToTensor aRow
        oInput = new Tensor(1, nFeatures)
        
        for i = 1 to nFeatures
            # Normalize to [0, 1]
            normalized = (number(aRow[i]) - nMin) / (nMax - nMin)
            oInput.setVal(1, i, normalized)
        next
        
        return [oInput, oTarget]
```

## Model Architecture

### 1. Use Appropriate Activations

**Performance ranking** (fastest to slowest):
1. ReLU (fastest)
2. Tanh
3. GELU
4. Sigmoid (slowest)

**Recommendation**: Use ReLU for hidden layers, GELU for Transformers.

### 2. Dropout Placement

```ring
# Good: After activation
oModel.add(new Dense(128, 128))
oModel.add(new ReLU)
oModel.add(new Dropout(0.2))

# Avoid: Before activation
oModel.add(new Dense(128, 128))
oModel.add(new Dropout(0.2))  # Less effective
oModel.add(new ReLU)
```

### 3. Layer Normalization

Use LayerNorm for Transformers:

```ring
# Stabilizes training
oModel.add(new MultiHeadAttention(512, 8))
oModel.add(new LayerNorm(512))
```

## Benchmarking

### Measure Training Speed

```ring
load "ringml.ring"

func benchmarkTraining
    # Setup
    oModel = createModel()
    oLoader = new DataLoader(dataset, 64)
    
    # Warmup (exclude from timing)
    for i = 1 to 10
        batch = oLoader.getBatch(1)
        oModel.forward(batch[1])
    next
    
    # Actual benchmark
    t1 = clock()
    nIterations = 100
    
    for i = 1 to nIterations
        batch = oLoader.getBatch((i % oLoader.nBatches) + 1)
        
        # Forward
        oPred = oModel.forward(batch[1])
        
        # Backward
        oGrad = oCriterion.backward(oPred, batch[2])
        oModel.backward(oGrad)
        
        # Update
        oModel.update_weights(oOptimizer)
    next
    
    t2 = clock()
    
    nSeconds = (t2 - t1) / clockspersecond()
    nSamplesPerSec = (nIterations * 64) / nSeconds
    
    see "Throughput: " + nSamplesPerSec + " samples/sec" + nl
    see "Time per iteration: " + (nSeconds / nIterations * 1000) + " ms" + nl
```

### Compare Eager vs Graph Mode

```ring
func compareModels
    # Eager mode
    t1 = clock()
    for i = 1 to 1000
        oPred = oModel.forward(oInput)
    next
    t2 = clock()
    nEagerTime = (t2 - t1) / clockspersecond()
    
    # Graph mode
    oInput.setGraphMode(true)
    oGraph = oModel.compile(oInput)
    
    t1 = clock()
    for i = 1 to 1000
        oPred = graph_run(oGraph.pGraph, oInput.pData)
    next
    t2 = clock()
    nGraphTime = (t2 - t1) / clockspersecond()
    
    see "Eager mode: " + nEagerTime + "s" + nl
    see "Graph mode: " + nGraphTime + "s" + nl
    see "Speedup: " + (nEagerTime / nGraphTime) + "x" + nl
```

## Performance Checklist

Before deploying to production:

- [ ] Use Graph Mode for inference
- [ ] Optimize batch size for your hardware
- [ ] Use Adam optimizer
- [ ] Enable gradient clipping
- [ ] Normalize input data
- [ ] Use appropriate activations (ReLU for speed)
- [ ] Profile memory usage
- [ ] Benchmark throughput
- [ ] Test on target hardware
- [ ] Consider model quantization (future)

## Expected Performance

### Reference Hardware: Intel i7-10700K, 32GB RAM

| Model | Batch Size | Mode | Throughput |
|-------|------------|------|------------|
| MLP (784→128→10) | 64 | Eager | 15k samples/sec |
| MLP (784→128→10) | 64 | Graph | 25k samples/sec |
| Transformer (512d, 8h) | 32 | Eager | 120 samples/sec |
| Transformer (512d, 8h) | 32 | Graph | 320 samples/sec |

### With GPU (NVIDIA RTX 3080)

| Model | Batch Size | Throughput |
|-------|------------|------------|
| MLP (784→128→10) | 256 | 80k samples/sec |
| Transformer (512d, 8h) | 64 | 1.2k samples/sec |

## Troubleshooting Performance Issues

### Slow Training

1. **Check batch size**: Increase if memory allows
2. **Use Graph Mode**: For fixed architectures
3. **Profile bottlenecks**: Identify slow operations
4. **Reduce model size**: Fewer parameters = faster training

### High Memory Usage

1. **Reduce batch size**
2. **Clear unused tensors**: `oTensor = NULL; callgc()`
3. **Use smaller model**: Reduce layer dimensions
4. **Process data in chunks**: Don't load all at once

### GPU Not Used

1. **Check OpenCL installation**: Verify drivers
2. **Increase tensor size**: Small tensors use CPU
3. **Check RingTensor build**: Ensure OpenCL support compiled

---

**Next**: See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues.
