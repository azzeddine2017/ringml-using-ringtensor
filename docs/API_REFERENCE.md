# RingML API Reference

Complete API documentation for the RingML deep learning framework.

## Table of Contents

- [Core](#core)
  - [Tensor](#tensor)
- [Models](#models)
  - [Sequential](#sequential)
  - [TransformerBlock](#transformerblock)
- [Layers](#layers)
  - [Dense](#dense)
  - [Embedding](#embedding)
  - [Dropout](#dropout)
  - [LayerNorm](#layernorm)
  - [Softmax](#softmax)
  - [Activation](#activation)
  - [MultiHeadAttention](#multiheadattention)
- [Loss Functions](#loss-functions)
  - [MSELoss](#mseloss)
  - [CrossEntropyLoss](#crossentropyloss)
- [Optimizers](#optimizers)
  - [SGD](#sgd)
  - [Adam](#adam)
- [Data](#data)
  - [Dataset](#dataset)
  - [UniversalDataset](#universaldataset)
  - [DataLoader](#dataloader)
- [Serialization](#serialization)
- [Utilities](#utilities)

---

## Core

### Tensor

The fundamental data structure for all operations.

**File**: `libraries/ringml/src/core/tensor.ring`

#### Constructor

```ring
oTensor = new Tensor(nRows, nCols)
```

**Parameters**:
- `nRows` (Number): Number of rows
- `nCols` (Number): Number of columns

**Example**:
```ring
oTensor = new Tensor(32, 512)  # 32 rows, 512 columns
```

#### Methods

##### `setVal(nRow, nCol, nValue)`

Set a single value in the tensor.

**Parameters**:
- `nRow` (Number): Row index (1-indexed)
- `nCol` (Number): Column index (1-indexed)
- `nValue` (Number): Value to set

**Example**:
```ring
oTensor.setVal(1, 1, 5.0)
```

##### `getVal(nRow, nCol)`

Get a single value from the tensor.

**Returns**: Number

**Example**:
```ring
value = oTensor.getVal(1, 1)  # Returns 5.0
```

##### `matmul(oOther)`

Matrix multiplication.

**Parameters**:
- `oOther` (Tensor): Tensor to multiply with

**Returns**: Tensor

**Example**:
```ring
oResult = oTensor1.matmul(oTensor2)
```

##### `transpose()`

Transpose the tensor.

**Returns**: Tensor

**Example**:
```ring
oTransposed = oTensor.transpose()
```

##### `add(oOther)`

Element-wise addition.

**Parameters**:
- `oOther` (Tensor): Tensor to add

**Returns**: Tensor

##### `addRowVec(oVec)`

Add a row vector to all rows (broadcasting).

**Parameters**:
- `oVec` (Tensor): Row vector (1 × N)

**Returns**: Tensor

##### `zeros()`

Fill tensor with zeros.

##### `fill(nValue)`

Fill tensor with a specific value.

**Parameters**:
- `nValue` (Number): Value to fill

##### `random()`

Fill tensor with random values [0, 1].

##### `shape()`

Get tensor dimensions.

**Returns**: List [nRows, nCols]

**Example**:
```ring
aShape = oTensor.shape()  # [32, 512]
```

##### `toList()`

Convert tensor to Ring list (for serialization).

**Returns**: List

##### `fromList(aData)`

Load tensor from Ring list.

**Parameters**:
- `aData` (List): Data to load

---

## Models

### Sequential

Container for stacking layers sequentially.

**File**: `libraries/ringml/src/model/sequential.ring`

#### Constructor

```ring
oModel = new Sequential
```

#### Methods

##### `add(oLayer)`

Add a layer to the model.

**Parameters**:
- `oLayer` (Layer): Layer instance

**Returns**: self (for chaining)

**Example**:
```ring
oModel = new Sequential
oModel.add(new Dense(10, 64))
oModel.add(new Tanh)
oModel.add(new Dense(64, 3))
```

##### `forward(oInput)`

Forward pass through all layers.

**Parameters**:
- `oInput` (Tensor): Input tensor

**Returns**: Tensor (output)

**Example**:
```ring
oPred = oModel.forward(oInput)
```

##### `backward(oGradOutput)`

Backward pass through all layers.

**Parameters**:
- `oGradOutput` (Tensor): Gradient from loss function

**Returns**: Tensor (input gradient)

**Example**:
```ring
oGrad = oCriterion.backward(oPred, oTarget)
oModel.backward(oGrad)
```

##### `train()`

Set model to training mode (enables Dropout).

**Example**:
```ring
oModel.train()
```

##### `evaluate()`

Set model to evaluation mode (disables Dropout).

**Example**:
```ring
oModel.evaluate()
```

##### `summary()`

Display model architecture with parameter counts.

**Example**:
```ring
oModel.summary()
```

Output:
```
_________________________________________________________________
Layer (Type)                  Output Shape              Param #
=================================================================
dense                         (None, 64)                704
tanh                          (None, 64)                0
dense                         (None, 3)                 195
_________________________________________________________________
Total params:         899
Trainable params:     899
Non-trainable params: 0
```

##### `saveWeights(cFileName)`

Save model weights to file.

**Parameters**:
- `cFileName` (String): File path (.rdata extension recommended)

**Example**:
```ring
oModel.saveWeights("model.rdata")
```

##### `loadWeights(cFileName)`

Load model weights from file.

**Parameters**:
- `cFileName` (String): File path

**Example**:
```ring
oModel.loadWeights("model.rdata")
```

##### `getParams()`

Get all trainable parameters.

**Returns**: List of [weights, gradients] pairs

##### `compile(oInputTemplate)`

Compile model to graph mode.

**Parameters**:
- `oInputTemplate` (Tensor): Template input with graph mode enabled

**Returns**: Tensor (graph output node)

**Example**:
```ring
oInput = new Tensor(32, 512)
oInput.setGraphMode(true)
oOutput = oModel.compile(oInput)
```

### TransformerBlock

Transformer block with self-attention and feed-forward network.

**File**: `libraries/ringml/src/model/transformer_block.ring`

#### Constructor

```ring
oBlock = new TransformerBlock(nDModel, nHeads, nDFF, nDropout)
```

**Parameters**:
- `nDModel` (Number): Model dimension
- `nHeads` (Number): Number of attention heads
- `nDFF` (Number): Feed-forward dimension
- `nDropout` (Number): Dropout rate (0.0 - 1.0)

**Example**:
```ring
oBlock = new TransformerBlock(512, 8, 2048, 0.1)
```

---

## Layers

### Dense

Fully connected (linear) layer.

**File**: `libraries/ringml/src/layers/dense.ring`

#### Constructor

```ring
oLayer = new Dense(nInputSize, nNeurons)
```

**Parameters**:
- `nInputSize` (Number): Input feature dimension
- `nNeurons` (Number): Output dimension

**Example**:
```ring
oLayer = new Dense(784, 128)  # 784 inputs -> 128 outputs
```

#### Methods

##### `forward(oInput)`

**Parameters**:
- `oInput` (Tensor): Input tensor (batch_size × input_size)

**Returns**: Tensor (batch_size × output_size)

##### `backward(oGradOutput)`

**Parameters**:
- `oGradOutput` (Tensor): Gradient from next layer

**Returns**: Tensor (gradient w.r.t. input)

##### `freeze()`

Freeze layer (stop training).

##### `unfreeze()`

Unfreeze layer (resume training).

### Embedding

Token embedding layer (lookup table).

**File**: `libraries/ringml/src/layers/embedding.ring`

#### Constructor

```ring
oLayer = new Embedding(nVocabSize, nEmbedDim)
```

**Parameters**:
- `nVocabSize` (Number): Vocabulary size
- `nEmbedDim` (Number): Embedding dimension

**Example**:
```ring
oEmbed = new Embedding(10000, 512)  # 10k vocab, 512-dim embeddings
```

#### Methods

##### `forward(oInputTensor)`

**Parameters**:
- `oInputTensor` (Tensor): Integer indices (batch_size × seq_len)

**Returns**: Tensor (batch_size × seq_len × embed_dim)

### Dropout

Regularization layer that randomly drops neurons during training.

**File**: `libraries/ringml/src/layers/dropout.ring`

#### Constructor

```ring
oLayer = new Dropout(nRate)
```

**Parameters**:
- `nRate` (Number): Dropout probability (0.0 - 1.0)

**Example**:
```ring
oDropout = new Dropout(0.2)  # Drop 20% of neurons
```

**Note**: Automatically disabled in evaluation mode.

### LayerNorm

Layer normalization.

**File**: `libraries/ringml/src/layers/layernorm.ring`

#### Constructor

```ring
oLayer = new LayerNorm(nDim)
```

**Parameters**:
- `nDim` (Number): Feature dimension to normalize

**Example**:
```ring
oNorm = new LayerNorm(512)
```

### Softmax

Softmax activation (converts logits to probabilities).

**File**: `libraries/ringml/src/layers/softmax.ring`

#### Constructor

```ring
oLayer = new Softmax
```

**Example**:
```ring
oModel.add(new Dense(128, 10))
oModel.add(new Softmax)  # Output probabilities for 10 classes
```

### Activation

Activation functions.

**File**: `libraries/ringml/src/layers/activation.ring`

#### Available Activations

```ring
oTanh = new Tanh        # Hyperbolic tangent
oSigmoid = new Sigmoid  # Sigmoid (0, 1)
oReLU = new ReLU        # Rectified Linear Unit
oGELU = new GELU        # Gaussian Error Linear Unit
```

**Example**:
```ring
oModel.add(new Dense(64, 64))
oModel.add(new ReLU)
```

### MultiHeadAttention

Multi-head self-attention mechanism.

**File**: `libraries/ringml/src/layers/multihead_attention.ring`

#### Constructor

```ring
oLayer = new MultiHeadAttention(nDModel, nHeads)
```

**Parameters**:
- `nDModel` (Number): Model dimension
- `nHeads` (Number): Number of attention heads

**Example**:
```ring
oAttn = new MultiHeadAttention(512, 8)
```

#### Methods

##### `forward(oInput, nBatch, nSeq)`

**Parameters**:
- `oInput` (Tensor): Input tensor
- `nBatch` (Number): Batch size
- `nSeq` (Number): Sequence length

**Returns**: Tensor

---

## Loss Functions

### MSELoss

Mean Squared Error loss (for regression).

**File**: `libraries/ringml/src/loss/mse.ring`

#### Constructor

```ring
oCriterion = new MSELoss
```

#### Methods

##### `forward(oPred, oTarget)`

Calculate loss.

**Parameters**:
- `oPred` (Tensor): Predictions
- `oTarget` (Tensor): Ground truth

**Returns**: Number (scalar loss)

**Example**:
```ring
loss = oCriterion.forward(oPred, oTarget)
```

##### `backward(oPred, oTarget)`

Calculate gradient.

**Returns**: Tensor (gradient w.r.t. predictions)

**Example**:
```ring
oGrad = oCriterion.backward(oPred, oTarget)
```

### CrossEntropyLoss

Cross-entropy loss (for classification).

**File**: `libraries/ringml/src/loss/crossentropy.ring`

#### Constructor

```ring
oCriterion = new CrossEntropyLoss
```

**Example**:
```ring
oCriterion = new CrossEntropyLoss
loss = oCriterion.forward(oPred, oTarget)
oGrad = oCriterion.backward(oPred, oTarget)
```

---

## Optimizers

### SGD

Stochastic Gradient Descent optimizer.

**File**: `libraries/ringml/src/optim/sgd.ring`

#### Constructor

```ring
oOptimizer = new SGD(nLearningRate)
```

**Parameters**:
- `nLearningRate` (Number): Learning rate (e.g., 0.01)

**Example**:
```ring
oOptimizer = new SGD(0.01)
```

#### Methods

##### `updateTensor(oWeights, oGrads)`

Update weights using gradients.

**Parameters**:
- `oWeights` (Tensor): Weight tensor
- `oGrads` (Tensor): Gradient tensor

**Example**:
```ring
oOptimizer.updateTensor(oWeights, oGrads)
```

### Adam

Adam optimizer (adaptive learning rates).

**File**: `libraries/ringml/src/optim/adam.ring`

#### Constructor

```ring
oOptimizer = new Adam(nLearningRate, nWeightDecay)
```

**Parameters**:
- `nLearningRate` (Number): Learning rate (default: 0.001)
- `nWeightDecay` (Number): L2 regularization (default: 0.0)

**Example**:
```ring
oOptimizer = new Adam(0.001, 0.0)
```

**Hyperparameters**:
- `lr` = 0.001 (learning rate)
- `beta1` = 0.9 (momentum)
- `beta2` = 0.999 (RMSprop)
- `epsilon` = 1e-7 (numerical stability)

#### Methods

##### `updateTensor(oWeights, oGrads)`

Update weights using Adam algorithm.

**Example**:
```ring
oOptimizer.updateTensor(oWeights, oGrads)
```

---

## Data

### Dataset

Base class for datasets.

**File**: `libraries/ringml/src/data/dataset.ring`

#### Methods

##### `length()`

Get dataset size.

**Returns**: Number

##### `getData(nIdx)`

Get a single sample.

**Parameters**:
- `nIdx` (Number): Sample index (1-indexed)

**Returns**: List [input_tensor, target_tensor]

### UniversalDataset

Flexible dataset for CSV/JSON/TXT files.

**File**: `libraries/ringml/src/data/universaldataset.ring`

#### Constructor

```ring
oDataset = new MyDataset(cFilePath)
```

**Parameters**:
- `cFilePath` (String): Path to data file

**Example**:
```ring
class MyDataset from UniversalDataset
    func rowToTensor aRow
        # Convert row to [input, target] tensors
        oInput = new Tensor(1, 2)
        oInput.setVal(1, 1, number(aRow[1]))
        oInput.setVal(1, 2, number(aRow[2]))
        
        oTarget = new Tensor(1, 1)
        oTarget.setVal(1, 1, number(aRow[3]))
        
        return [oInput, oTarget]

oDataset = new MyDataset("data.csv")
```

#### Methods

##### `setHeader(bStatus)`

Set whether file has header row.

**Parameters**:
- `bStatus` (Boolean): true if header exists

**Returns**: self

##### `setShuffle(bStatus)`

Enable/disable shuffling.

**Parameters**:
- `bStatus` (Boolean): true to shuffle

**Returns**: self

##### `setSplit(bStatus)`

Enable train/test splitting.

**Parameters**:
- `bStatus` (Boolean): true to split

**Returns**: self

##### `setRatio(nRatio)`

Set test split ratio.

**Parameters**:
- `nRatio` (Number): Test ratio (e.g., 0.2 for 20%)

**Returns**: self

##### `loadData()`

Load and process the data file.

**Returns**: self

**Example**:
```ring
oDataset = new MyDataset("data.csv")
oDataset.setHeader(true)
oDataset.setShuffle(true)
oDataset.setSplit(true)
oDataset.setRatio(0.2)
oDataset.loadData()
```

##### `getTrainDataset()`

Get training dataset.

**Returns**: Dataset

##### `getTestDataset()`

Get test dataset.

**Returns**: Dataset

##### `rowToTensor(aRow)`

**Abstract method** - Must be overridden in subclass.

Convert a raw row to tensors.

**Parameters**:
- `aRow` (List): Raw row data

**Returns**: List [input_tensor, target_tensor]

### DataLoader

Mini-batch data loader.

**File**: `libraries/ringml/src/data/DataLoader.ring`

#### Constructor

```ring
oLoader = new DataLoader(oDataset, nBatchSize)
```

**Parameters**:
- `oDataset` (Dataset): Dataset instance
- `nBatchSize` (Number): Batch size

**Example**:
```ring
trainLoader = new DataLoader(trainDataset, 32)
```

#### Properties

- `nBatches` (Number): Total number of batches

#### Methods

##### `getBatch(nIdx)`

Get a specific batch.

**Parameters**:
- `nIdx` (Number): Batch index (1 to nBatches)

**Returns**: List [inputs_tensor, targets_tensor]

**Example**:
```ring
for b = 1 to trainLoader.nBatches
    batch = trainLoader.getBatch(b)
    inputs = batch[1]
    targets = batch[2]
    # ... training code ...
next
```

---

## Serialization

### Saving Models

```ring
oModel.saveWeights("model.rdata")
```

**Format**: Binary serialization using Ring's `SerializeData()`

### Loading Models

```ring
# Create model with same architecture
oModel = new Sequential
oModel.add(new Dense(10, 64))
oModel.add(new Tanh)
oModel.add(new Dense(64, 3))

# Load weights
oModel.loadWeights("model.rdata")
```

**Important**: Model architecture must match exactly.

---

## Utilities

### Gradient Clipping

```ring
# Get all gradients
aGrads = oModel.get_gradients()

# Clip to max norm
tensor_clip_global_norm(aGrads, 1.0)
```

**Parameters**:
- `aGrads` (List): List of gradient tensors
- `nMaxNorm` (Number): Maximum gradient norm

### Helper Functions

**File**: `libraries/ringml/src/utils/functions.ring`

#### `raise(cMessage)`

Raise an error and exit.

**Parameters**:
- `cMessage` (String): Error message

#### `info(cMessage)`

Print info message (colored).

#### `warning(cMessage)`

Print warning message (colored).

#### `success(cMessage)`

Print success message (colored).

#### `error(cMessage)`

Print error message (colored).

---

## Complete Training Example

```ring
load "ringml.ring"
load "stdlib.ring"

# 1. Prepare Data
class MyDataset from UniversalDataset
    func rowToTensor aRow
        oInput = new Tensor(1, 10)
        for i = 1 to 10
            oInput.setVal(1, i, number(aRow[i]))
        next
        
        oTarget = new Tensor(1, 3)
        oTarget.setVal(1, 1, number(aRow[11]))
        oTarget.setVal(1, 2, number(aRow[12]))
        oTarget.setVal(1, 3, number(aRow[13]))
        
        return [oInput, oTarget]

oData = new MyDataset("data.csv")
oData.setHeader(true).setShuffle(true).setSplit(true).loadData()

trainLoader = new DataLoader(oData.getTrainDataset(), 32)
testLoader = new DataLoader(oData.getTestDataset(), 32)

# 2. Build Model
oModel = new Sequential
oModel.add(new Dense(10, 64))
oModel.add(new ReLU)
oModel.add(new Dropout(0.2))
oModel.add(new Dense(64, 32))
oModel.add(new ReLU)
oModel.add(new Dense(32, 3))
oModel.add(new Softmax)

oModel.summary()

# 3. Training Setup
oCriterion = new CrossEntropyLoss
oOptimizer = new Adam(0.001)
nEpochs = 50

# 4. Training Loop
oModel.train()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b)
        inputs = batch[1]
        targets = batch[2]
        
        # Forward
        preds = oModel.forward(inputs)
        loss = oCriterion.forward(preds, targets)
        
        # Backward
        grad = oCriterion.backward(preds, targets)
        oModel.backward(grad)
        
        # Gradient clipping
        aGrads = oModel.get_gradients()
        tensor_clip_global_norm(aGrads, 1.0)
        
        # Update
        oModel.update_weights(oOptimizer)
        
        epochLoss += loss
    next
    
    see "Epoch " + epoch + " - Loss: " + (epochLoss / trainLoader.nBatches) + nl
next

# 5. Evaluation
oModel.evaluate()

testLoss = 0
for b = 1 to testLoader.nBatches
    batch = testLoader.getBatch(b)
    inputs = batch[1]
    targets = batch[2]
    
    preds = oModel.forward(inputs)
    loss = oCriterion.forward(preds, targets)
    testLoss += loss
next

see "Test Loss: " + (testLoss / testLoader.nBatches) + nl

# 6. Save Model
oModel.saveWeights("final_model.rdata")
```

---

**Next**: See [Tutorials](TUTORIALS.md) for step-by-step guides.
