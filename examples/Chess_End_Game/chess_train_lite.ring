# File: examples/chess_train_lite.ring
# Description: FAST training for Chess (Lite Version for CPU i3)

load "stdlib.ring"
load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"
load "csvlib.ring"

decimals(4)
see "=== RingML Chess Training (Lite CPU Mode) ===" + nl

# 1. Load Data
cFile = "data/chess.csv"
if !fexists(cFile) raise("File missing") ok

see "Reading CSV..." + nl
aRawsData = CSV2List(read(cFile))

# Clean Data
if len(aRawsData) > 0 
    if lower(aRawsData[1][1]) = "whitekingfile" or lower(aRawsData[1][1]) = "label"
        del(aRawsData, 1)
    ok
ok

# --- OPTIMIZATION 1: Subsampling (Take only 5000 random samples) ---
see "Optimizing Dataset Size..." + nl
nLiteSize = 5000
if len(aRawsData) > nLiteSize
    aLiteData = []
    # Shuffle first to get random variety
    splitter = new DataSplitter
    splitter.shuffle(aRawsData)
    
    # Pick first 5000
    for i = 1 to nLiteSize
        aLiteData + aRawsData[i]
    next
    aRawsData = aLiteData
    see "Reduced dataset to " + nLiteSize + " samples for speed." + nl
ok

# 2. Split (80% Train, 20% Test)
splitter = new DataSplitter
sets = splitter.splitData(aRawsData, 0.2, true) 
aTrain = sets[1]
aTest  = sets[2]

# --- OPTIMIZATION 2: Large Batch Size ---
# Less updates = Less overhead
batch_size = 128

trainDataset = new ChessDataset(aTrain)
testDataset  = new ChessDataset(aTest)

trainLoader  = new DataLoader(trainDataset, batch_size)
testLoader   = new DataLoader(testDataset, batch_size)

# 3. Build Model (Smaller Architecture)
nClasses = 18
model = new Sequential

# Input -> Dense(50) -> Tanh -> Dense(30) -> Output
# Smaller layers = Much faster Matrix Multiplication
model.add(new Dense(6, 50))   
model.add(new Tanh)        
model.add(new Dense(50, 30))  
model.add(new Tanh)
model.add(new Dense(30, nClasses)) 
model.add(new Softmax)

model.summary()

# 4. Training
criterion = new CrossEntropyLoss
optimizer = new Adam(0.001) 
nEpochs   = 20 # 20 Epochs should finish quickly now

viz = new TrainingVisualizer(nEpochs, trainLoader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    
    # Training
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
        
        # Update Visualizer
        viz.update(epoch, b, loss, 0)
    next
    
    avgTrainLoss = epochLoss / trainLoader.nBatches
    
    # Validation (Fast Check)
    model.evaluate()
    correct = 0
    total = 0
    
    # Check accuracy on Test Set
    for b = 1 to testLoader.nBatches
        batch = testLoader.getBatch(b)
        preds = model.forward(batch[1])
        targets = batch[2]
        
        nBatchSize = preds.nRows
        for i = 1 to nBatchSize
            # Simple ArgMax
            pMax = -1 pIdx=0
            for k=1 to nClasses if preds.aData[i][k] > pMax pMax=preds.aData[i][k] pIdx=k ok next
            
            tMax = -1 tIdx=0
            for k=1 to nClasses if targets.aData[i][k] > tMax tMax=targets.aData[i][k] tIdx=k ok next
            
            if pIdx = tIdx correct++ ok
            total++
        next
    next
    
    acc = (correct / total) * 100
    
    # Finish Epoch Display
    viz.finishEpoch(epoch, avgTrainLoss, acc)
    
    callgc()
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl
model.saveWeights("model/chess_model_lite.rdata")