# File: examples/fast_viz_demo.ring
# Description: Quick Demo with GUARANTEED convergence to see visuals
# Author: Azzeddine Remmal


load "../../libraries/ringml/src/ringml.ring"

decimals(10)

see "=== RingML Visual Demo (Guaranteed Learning) ===" + nl

# 1. Generate Very Clear Synthetic Data (3 Distinct Clusters)
see "Generating distinct data..." + nl
aRawsData = []

# Class 0: Near (0, 0)
for i = 1 to 200
    # FIX: Use add() instead of +. (+) merges lists, add() appends the row.
    add(aRawsData, [[random(10)/100.0, random(10)/100.0], [1.0, 0.0, 0.0]])
next

# Class 1: Near (1, 1)
for i = 1 to 200
    add(aRawsData, [[0.9 + random(10)/100.0, 0.9 + random(10)/100.0], [0.0, 1.0, 0.0]])
next

# Class 2: Near (0, 1)
for i = 1 to 200
    add(aRawsData, [[random(10)/100.0, 0.9 + random(10)/100.0], [0.0, 0.0, 1.0]])
next

# 2. Split Data
splitter = new DataSplitter
# FIX: Method name usually 'split' in standard library, ensure naming matches your file
sets = splitter.splitData(aRawsData, 0.2, true) 
aTrain = sets[1]
aTest  = sets[2]

# 3. Create Datasets using Helper
trainDataset = MakeTensorDataset(aTrain)
testDataset  = MakeTensorDataset(aTest)

batch_size = 16 
trainLoader = new DataLoader(trainDataset, batch_size)
testLoader  = new DataLoader(testDataset, batch_size)

# 4. Build Model
model = new Sequential
model.add(new Dense(2, 64))
model.add(new ReLU)
model.add(new Dense(64, 32))
model.add(new ReLU)
model.add(new Dense(32, 16))
model.add(new ReLU)
model.add(new Dense(16, 3))
model.add(new Softmax)

model.summary()

# 5. Training
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 20

# Load Visualizer

viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

for epoch = 1 to nEpochs
    model.train()
    epochLoss = 0

    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b)
        inputs = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for l in model.getLayers() optimizer.update(l) next
        
        # Animation Delay
        sleep(0.01) 
        viz.update(epoch, b, loss, 0)
    next
    
    avgLoss = epochLoss / trainLoader.nBatches
    
    # Validation
    model.evaluate()
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        preds = model.forward(batch[1])
        targets = batch[2]
        
        Rows = preds.nRows
        for r=1 to Rows
            pMax = -1 pIdx=0
            tMax = -1 tIdx=0
            for k=1 to 3
                # FIX: Use getVal() because we are using C-Pointers now
                # Direct access [] will not work with RingTensor
                valPred = preds.getVal(r, k)
                if valPred > pMax 
                    pMax = valPred 
                    pIdx = k 
                ok
                
                valTarget = targets.getVal(r, k)
                if valTarget > tMax 
                    tMax = valTarget 
                    tIdx = k 
                ok
            next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    acc = (correct/total) * 100
    viz.finishEpoch(epoch, avgLoss, acc)
next

see nl + "Done! Look at those colors! 🎨" + nl

model.saveWeights("Visual_test_model.rdata")

# --- Helper to Convert Lists to Tensors (Updated for C-Pointers) ---
func MakeTensorDataset aDataList
    nRows = len(aDataList)
    if nRows = 0 return NULL ok
    
    # Analyze dimensions from first row
    nInCols  = len(aDataList[1][1])
    nOutCols = len(aDataList[1][2])
    
    tIn  = new Tensor(nRows, nInCols)
    tOut = new Tensor(nRows, nOutCols)
    
    for r = 1 to nRows
        # Copy Inputs
        rowIn = aDataList[r][1]
        for c = 1 to nInCols
            tIn.setVal(r, c, rowIn[c]) # Use setVal
        next
        
        # Copy Targets
        rowOut = aDataList[r][2]
        for c = 1 to nOutCols
            tOut.setVal(r, c, rowOut[c]) # Use setVal
        next
    next
    
    return new TensorDataset(tIn, tOut)