# File: examples/chess_train_adam.ring
# Description: Training with Adam Optimizer (Super Fast)

load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"
load "csvlib.ring"
load "stdlib.ring"

decimals(5)
see "=== RingML Chess Training (Adam) ===" + nl

# 1. Load Data
cFile = "data/chess.csv"

see "Reading CSV..." + nl
aRawData = CSV2List( read(cFile) )
if len(aRawData) > 0 del(aRawData, 1) ok 

# 2. Setup Dataset 
dataset = new ChessDataset(aRawData)
batch_size = 256 
loader = new DataLoader(dataset, batch_size)

# 3. Model 
nClasses = 18
model = new Sequential

model.add(new Dense(6, 64))   
model.add(new Sigmoid)        
model.add(new Dense(64, 32))  
model.add(new Sigmoid)
model.add(new Dense(32, nClasses)) 
model.add(new Softmax)

# 4. Print Summary
model.summary()

# 5. Train with Adam
criterion = new CrossEntropyLoss

# Note: Adam usually needs lower LR than SGD. 0.01 is a good start.
optimizer = new Adam(0.005) 

nEpochs   = 50

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next

        if b % 5 = 0 see "."  ok

        if b % 50 = 0 callgc() ok
    next
    see nl
    avgLoss = epochLoss / loader.nBatches
    see "Epoch " + epoch + " Avg Loss: " + avgLoss + nl
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl


model.saveWeights("model/chess_model_adam.rdata")
see "Model Saved." + nl