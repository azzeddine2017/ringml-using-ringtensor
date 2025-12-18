# File: examples/xor_train.ring
# Description: Training a Neural Network to solve XOR problem
# Author: Azzeddine Remmal

load "ringml.ring" 

decimals(10)

see "=== RingML XOR Training Example (Fixed) ===" + nl

# 1. Prepare Data
# XOR Inputs
aRawInputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
# Convert to Tensor
inputs = listToTensor(aRawInputs)

# XOR Targets
aRawTargets = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]
targets = listToTensor(aRawTargets)

# 2. Build Model
model = new Sequential

# Hidden Layer: 2 inputs -> 4 neurons
model.add(new Dense(2, 4))
model.add(new Sigmoid)

# Output Layer: 4 inputs -> 1 neuron
model.add(new Dense(4, 1))
model.add(new Sigmoid)

# 3. Setup Loss and Optimizer
criterion = new MSELoss
optimizer = new SGD(0.5) # Increased LR for faster convergence with Sigmoid

# 4. Training Loop
nEpochs = 5000

see "Training started for " + nEpochs + " epochs..." + nl

for epoch = 1 to nEpochs
    # A. Forward Pass
    preds = model.forward(inputs)
    
    # B. Calculate Loss
    loss = criterion.forward(preds, targets)
    
    # Print progress
    if epoch % 500 = 0
        see "Epoch " + epoch + " : Loss = " + loss + nl
    ok
    
    # C. Backward Pass
    lossGrad = criterion.backward(preds, targets)
    model.backward(lossGrad)
    
    # D. Optimize
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next

see nl + "=== Training Finished ===" + nl

# 5. Verify Predictions
see "Final Predictions:" + nl
finalPreds = model.forward(inputs)
finalPreds.print()

see "Expected Targets:" + nl
targets.print()

