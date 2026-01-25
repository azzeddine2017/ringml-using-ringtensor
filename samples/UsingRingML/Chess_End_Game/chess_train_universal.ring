# File: examples/chess_train_universal.ring
# Description: Professional Training using UniversalDataset & DataSplitter
# Author: Code Gear-1

load "ringml.ring"
load "chess_utils.ring"

decimals(5)


# ============================================================
#           Main Execution
# ============================================================
see "=== RingML Universal Loader Demo ===" + nl

# A. Initialize & Configure Data
data = new ChessDataHandler("data/chess.csv")
data.setHeader(true)        # Skip first row
data.setShuffle(true)       # Randomize
data.setSplit(0.2)          # 20% for Testing
data.loadData()             # Run the pipeline

# B. Create Loaders
batch_size = 128
trainLoader = new DataLoader(data.getTrainDataset(), batch_size)
testLoader  = new DataLoader(data.getTestDataset(), batch_size)

see "Batches/Epoch: " + trainLoader.nBatches + nl

# C. Build Model (Optimized for Speed/Accuracy)
model = new Sequential

model.add(new Dense(6, 128))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

model.add(new Dense(128, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

model.add(new Dense(64, 18)) 
model.add(new Softmax)

model.summary()

# D. Setup Training
criterion = new CrossEntropyLoss
optimizer = new Adam(0.0001) 
nEpochs   = 50

# E. Visualization
viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    
    # --- 1. Training Phase ---
    model.train()
    epochLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b)
        inputs = batch[1]
        targets = batch[2]
        
        # Optimization Step
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for l in model.getLayers() optimizer.update(l) next
        
        epochLoss += loss
        
        # Update UI every 5 batches
        if b % 5 = 0 viz.update(epoch, b, loss, 0) ok
    next
    
    avgTrainLoss = epochLoss / trainLoader.nBatches
    
    # --- 2. Validation Phase ---
    model.evaluate()
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        inputs = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        
        # Calculate Accuracy
        nBatchSize = preds.nRows
        for r=1 to nBatchSize
            # ArgMax Pred
            pMax = -1 pIdx=0
            for k=1 to 18
                val = preds.getVal(r, k)
                if val > pMax pMax=val pIdx=k ok
            next
            
            # ArgMax Target
            tMax = -1 tIdx=0
            for k=1 to 18
                val = targets.getVal(r, k)
                if val > tMax tMax=val tIdx=k ok
            next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    acc = (correct / total) * 100
    
    # --- 3. Finish Epoch ---
    viz.finishEpoch(epoch, avgTrainLoss, acc)
    
    callgc()
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# F. Save
model.saveWeights("model/chess_universal.rdata")
see "Model Saved." + nl

# ============================================================
#             Define Custom Dataset Logic
# ============================================================
class ChessDataHandler from UniversalDataset
    
    nFeatures = 6
    nClasses  = 18

    # Override: Define how to convert a single CSV row to Tensors
    func rowToTensor row
        # row is a list ["a", "1", "b", "3", ...]
        
        # A. Process Input Features
        wk_f = getFileIndex(row[1])
        wk_r = number(row[2])
        wr_f = getFileIndex(row[3])
        wr_r = number(row[4])
        bk_f = getFileIndex(row[5])
        bk_r = number(row[6])
        
        # Normalize (1-8 -> 0.125-1.0)
        normData = normalizeBoard([wk_f, wk_r, wr_f, wr_r, bk_f, bk_r])
        
        # Create Input Tensor (1, 6)
        oIn = new Tensor(1, nFeatures)
        for i=1 to nFeatures 
            oIn.setVal(1, i, normData[i]) 
        next

        # B. Process Target Label
        cLabel = row[7]
        nLabelIdx = getLabelIndex(cLabel)
        
        # Create Target Tensor (1, 18) - One Hot
        oOut = new Tensor(1, nClasses)
        oOut.setVal(1, nLabelIdx, 1.0) 

        return [oIn, oOut]

