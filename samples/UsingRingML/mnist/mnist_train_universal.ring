# File: examples/mnist_train_universal.ring
# Description: MNIST Training using UniversalDataset & DataSplitter
# Author: Azzeddine Remmal 

load "ringml.ring"             
load "mnist_dataset.ring"


see "num cores :" + tensor_get_cores()
tensor_set_threads(2)

decimals(8)

see "=== RingML MNIST Digit Recognition (Universal Loader) ===" + nl


# ============================================================
#                    Main Execution
# ============================================================

# A. Initialize & Configure Data
cFile = "data/mnist_test.csv" # Ensure this file exists
data = new MnistDataHandler(cFile)

data.setHeader(true)        # Skip the header row ("label, 1x1...")
data.setShuffle(true)       # Randomize data
data.setSplit(0.2)          # 20% for Testing (Validation)
data.loadData()             # Execute loading pipeline

# B. Create Loaders
batch_size = 64
trainLoader = new DataLoader(data.getTrainDataset(), batch_size)
testLoader  = new DataLoader(data.getTestDataset(), batch_size)

see "Batches per Epoch: " + trainLoader.nBatches + nl

# 3. Build Model (784 -> 64 -> 32 -> 10)
model = new Sequential

# Input Layer
model.add(new Dense(784, 64))   
model.add(new ReLU)
model.add(new Dropout(0.2)) 

# Hidden Layer
model.add(new Dense(64, 32))  
model.add(new ReLU)
model.add(new Dropout(0.2))

# Output Layer
model.add(new Dense(32, 10)) 
//model.add(new Softmax)

model.summary()

# 4. Training Setup
criterion = new CrossEntropyLoss
optimizer = new Adam(0.005,0.0001) # Adam
nEpochs   = 50

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    
    # --- Training Phase ---
    model.train() 
    epochLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next

        # Update Visualizer (Loss only)
        if b % 5 = 0 viz.update(epoch, b, loss, 0) ok
        
        # GC Management
        if b % 50 = 0 callgc() ok
    next
    
    avgLoss = epochLoss / trainLoader.nBatches

    # --- Validation Phase (Calculate Accuracy) ---
    model.evaluate()
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        
        # Accuracy Logic
        nBatchSize = preds.nRows
        for r=1 to nBatchSize
            # ArgMax Pred
            pMax = -1000 pIdx = 0
            for k=1 to 10
                val = preds.getVal(r, k)
                if val > pMax pMax=val pIdx=k ok
            next
            
            # ArgMax Target
            tMax = -1000 tIdx = 0
            for k=1 to 10
                val = targets.getVal(r, k)
                if val > tMax tMax=val tIdx=k ok
            next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    acc = (correct / total) * 100
    
    # Finish Epoch with Accuracy
    viz.finishEpoch(epoch, avgLoss, acc)
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 5. Save
model.saveWeights("model/mnist_universal.rdata")
see "Model Saved." + nl

# ============================================================
#             Define Custom Dataset Logic
# ============================================================
class MnistDataHandler from UniversalDataset
    
    nFeatures = 784 # 28x28 pixels
    nClasses  = 10  # Digits 0-9

    # Override: Define how to convert a single CSV row to Tensors
    func rowToTensor row
        # row structure: [label, pixel1, pixel2, ... pixel784]
        
        # A. Process Target Label (First Column)
        # Convert string to number
        nLabel = number(row[1]) 
        
        # Create Target Tensor (1, 10) - One Hot Encoding
        # Ring lists are 1-based, so digit 0 maps to index 1
        oOut = new Tensor(1, nClasses)
        oOut.setVal(1, nLabel + 1, 1.0)

        # B. Process Input Features (Remaining Columns)
        # Create Input Tensor (1, 784)
        oIn = new Tensor(1, nFeatures)
        
        # Pixels start at index 2
        for i = 1 to nFeatures
            # Normalize 0-255 -> 0.0-1.0
            # row index is i+1 because index 1 is label
            nVal = number(row[i+1]) / 255.0
            oIn.setVal(1, i, nVal)
        next

        return [oIn, oOut]
