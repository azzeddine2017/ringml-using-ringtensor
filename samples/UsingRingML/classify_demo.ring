# Description: Multi-class classification example (Improved)
# Author: Azzeddine Remmal

load "ringml.ring"

decimals(8) 

see "=== RingML Multi-Class Classification Demo ===" + nl

# 1. Data (Define as Lists first)
aInputs = [
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0]
]

aTargets = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# Convert to Tensors using Helper
inputs  = MakeTensor(aInputs)
targets = MakeTensor(aTargets)

# 2. Model
model = new Sequential
model.add(new Dense(4, 8)) 
model.add(new Sigmoid)     
model.add(new Dense(8, 3)) 
model.add(new Softmax)     

# 3. Training
criterion = new CrossEntropyLoss
optimizer = new SGD(0.001) 

see "Training..." + nl
for epoch = 1 to 3000
    # Forward
    preds = model.forward(inputs)
    
    # Loss
    loss = criterion.forward(preds, targets)
    
    if epoch % 500 = 0
        see "Epoch " + epoch + " Loss: " + loss + nl
    ok
    
    # Backward
    grad = criterion.backward(preds, targets)
    model.backward(grad)
    
    # Update
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next

see "=== Final Predictions ===" + nl
final = model.forward(inputs)
final.print()

see "=== Expected Targets ===" + nl
targets.print()

# --- Helper Function ---
func MakeTensor aList
    nRows = len(aList)
    nCols = len(aList[1])
    
    oTen = new Tensor(nRows, nCols)
    
    for r = 1 to nRows
        for c = 1 to nCols
            # Use setVal instead of aData[r][c]
            oTen.setVal(r, c, aList[r][c])
        next
    next
    return oTen
