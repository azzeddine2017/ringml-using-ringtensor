# File: examples/chess_train_split.ring
# Description: Chess Training using built-in DataSplitter
# Author: Azzeddine Remmal

load "ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"


decimals(5)

see "=== RingML Chess Training (Automated Split) ===" + nl

# 1. Load Data
cFile = "data/chess.csv"
if !fexists(cFile) raise("File missing") ok

see "Reading CSV..." + nl
aRawsData = CSV2List( read(cFile) )
if len(aRawsData) > 0 del(aRawsData, 1) ok 

# 2. Use DataSplitter
see "Splitting Data (80% Train, 20% Test)..." + nl

splitter = new DataSplitter
# Ensure the method name matches your DataSplitter class (split vs splitData)
sets = splitter.splitData(aRawsData, 0.2, true) 

aTrainData = sets[1]
aTestData  = sets[2]

# Free memory
aRawsData = [] 
callgc()

see "Training Set: " + len(aTrainData) + " samples." + nl
see "Testing Set:  " + len(aTestData)  + " samples." + nl

# 3. Setup Datasets & Loaders
batch_size = 128 

trainDataset = new ChessDataset(aTrainData)
testDataset  = new ChessDataset(aTestData)

trainLoader  = new DataLoader(trainDataset, batch_size)
testLoader   = new DataLoader(testDataset, batch_size)

# 4. Build Model
nClasses = 18
model = new Sequential

model.add(new Dense(6, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

model.add(new Dense(64, 32))  
model.add(new Tanh)
model.add(new Dropout(0.2))

model.add(new Dense(32, nClasses)) 
//model.add(new Softmax)

model.summary()

# 5. Training Setup
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001, 0.0001) 
nEpochs   = 10

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    
    # --- Training ---
    model.train() 
    trainLoss = 0
    
    for b = 1 to trainLoader.nBatches
        batch = trainLoader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        trainLoss += loss
        
        grad = criterion.backwardTensor()//(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next
        
        # --- UPDATE VISUALIZER ---
        if b % 5 = 0
            viz.update(epoch, b, loss, 0)
        ok
    next
    
    avgTrainLoss = trainLoss / trainLoader.nBatches
    
    # --- Validation ---
    model.evaluate()
    correct = 0
    total = 0
    
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        
        # Calculate Accuracy logic (Fixed for C-Pointers)
        nBatchSize = preds.nRows
        for i = 1 to nBatchSize
            # ArgMax Pred
            pMax = -1000 pIdx = 0
            for k=1 to nClasses 
                # FIX: Use getVal instead of aData[i][k]
                val = preds.getVal(i, k)
                if val > pMax pMax=val pIdx=k ok 
            next
            
            # ArgMax Target
            tMax = -1000 tIdx = 0
            for k=1 to nClasses 
                # FIX: Use getVal instead of aData[i][k]
                val = targets.getVal(i, k)
                if val > tMax tMax=val tIdx=k ok 
            next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    accuracy = (correct / total) * 100
    
    # --- FINISH EPOCH VISUALIZATION ---
    viz.finishEpoch(epoch, avgTrainLoss, accuracy)

    if epoch % 5 = 0 callgc() ok
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

model.saveWeights("model/chess_split_model.rdata")

see "Done." + nl