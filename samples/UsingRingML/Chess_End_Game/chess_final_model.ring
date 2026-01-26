# File: examples/chess_final_model.ring
# Description: The Ultimate Chess Training Script with All Features
# Author: Azzeddine Remmal
 
# Adam Optimizer: For speed.
# Dropout: To prevent overfitting.
# Tanh: As an activation function (instead of Sigmoid) because it converges faster in deeper layers.
# DataLoader: For data processing.
# Summary: To display the structure.
# SaveWeights: For saving.
# File: examples/chess_final_model.ring
# Description: Training the Final Chess Model (All Data)


load "ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"

decimals(5)

see "=== RingML Final Chess Model ===" + nl

# 1. Load Data
cFile = "data/chess.csv"
if !fexists(cFile) raise("File missing: " + cFile) ok

see "Reading CSV..." + nl
aRawsData = CSV2List( read(cFile) )

# Clean Header
if len(aRawsData) > 0 
    if lower(aRawsData[1][1]) = "whitekingfile" or lower(aRawsData[1][1]) = "label"
        del(aRawsData, 1)
    ok
ok

# 2. Setup Dataset & Loader
# Note: We use ALL data for the final model (No split)
dataset = new ChessDataset(aRawsData)
batch_size = 128 
loader = new DataLoader(dataset, batch_size)

see "Dataset: " + dataset.length() + " samples." + nl
see "Batches: " + loader.nBatches + " per epoch." + nl

# 3. Build Architecture
# Topology: 6 -> 64 -> 32 -> 18
nClasses = 18
model = new Sequential

# Input Layer (6 -> 64)
model.add(new Dense(6, 64))   
model.add(new Tanh)        
model.add(new Dropout(0.2))

# Hidden Layer (64 -> 32)
model.add(new Dense(64, 32))  
model.add(new Tanh)
model.add(new Dropout(0.2))

# Output Layer (32 -> 18)
model.add(new Dense(32, nClasses)) 
model.add(new Softmax)

# 4. Print Summary
model.summary()

# 5. Training Setup
criterion = new CrossEntropyLoss
oOptim = new Adam(0.001, 0.0001) 
nEpochs   = 100

# --- SETUP VISUALIZER ---
viz = new TrainingVisualizer(nEpochs, loader.nBatches)

see "Starting Training..." + nl
tTotal = clock()

# Enable Training Mode (Activates Dropout)
model.train()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        # Forward
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        # Backward
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        # Optimizer Step
        for layer in model.getLayers() optimizer.update(layer) next
        
        # --- UPDATE VISUALIZER ---
        if b % 5 = 0
            # Passing 0 for accuracy to save time in this loop
            viz.update(epoch, b, loss, 0)
        ok
    next
    
    avgLoss = epochLoss / loader.nBatches
    
    # --- FINISH EPOCH VISUALIZATION ---
    # We pass 0 as validation accuracy since we are training on the full set
    viz.finishEpoch(epoch, avgLoss, 0)
    
    # GC Management
    if epoch % 5 = 0 callgc() ok
next

see "Training Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 6. Evaluation Mode (Disable Dropout)
model.evaluate()

# 7. Save Model
# Ensure 'model' folder exists or save to root
model.saveWeights("model/chess_final.rdata") 
see "Model Saved Successfully." + nl