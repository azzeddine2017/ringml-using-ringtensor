# Description: Bottleneck Detector (Fixed for C-Pointers)

load "ringml.ring" 

decimals(4)

see "=== RingML Performance Benchmark (C-Pointer Mode) ===" + nl

# 1. Generate Data in Memory (Medium Size)
# 1000 Samples, 500 Features
nSamples = 1000
nFeatures = 1500
nHidden = 200
nClasses = 10

see "Generating Data (" + nSamples + "x" + nFeatures + ")... "
tGen = clock()

# Create big tensors directly
inputs = new Tensor(nSamples, nFeatures)
inputs.random() # Use class method

targets = new Tensor(nSamples, nClasses)
targets.fill(0.0) # Use class method

# Fake One-Hot (Fix: Use setVal instead of direct list access)
for i=1 to nSamples 
    targets.setVal(i, 1, 1.0) 
next

see "Done in " + ((clock()-tGen)/clockspersecond()) + "s" + nl

# 2. Build Model (Heavy Math)
model = new Sequential
model.add(new Dense(nFeatures, nHidden)) 
model.add(new Tanh)
model.add(new Dense(nHidden, nClasses))  
model.add(new Softmax)

model.summary()

criterion = new CrossEntropyLoss
optimizer = new Adam(0.001, 0.0001)

# 3. Benchmark Loop
nEpochs = 10
see nl + "Running Benchmark for " + nEpochs + " epochs..." + nl
see "Measuring average time per operation..." + nl

totalForward = 0
totalLoss = 0
totalBackward = 0
totalUpdate = 0
totalGC = 0

tTotalStart = clock()

for epoch = 1 to nEpochs
    
    # --- Measure Forward ---
    t1 = clock()
    preds = model.forward(inputs)
    totalForward += (clock() - t1)
    
    # --- Measure Loss ---
    t2 = clock()
    loss = criterion.forward(preds, targets)
    totalLoss += (clock() - t2)
    
    # --- Measure Backward ---
    t3 = clock()
    grad = criterion.backward(preds, targets)
    model.backward(grad)
    totalBackward += (clock() - t3)
    
    # --- Measure Optimization ---
    t4 = clock()
    for l in model.getLayers() optimizer.update(l) next
    totalUpdate += (clock() - t4)
    
    # --- Measure GC Overhead ---
    t5 = clock()
    callgc() 
    totalGC += (clock() - t5)
    
    # Carriage Return for animation effect
    see char(13) + "Epoch " + epoch + " done."
next

totalTime = (clock() - tTotalStart) / clockspersecond()

see nl + nl + "=== RESULTS (Total Time: " + totalTime + "s) ===" + nl
see "Forward Pass : " + (totalForward/clockspersecond()) + " s" + nl
see "Loss Calc    : " + (totalLoss/clockspersecond()) + " s" + nl
see "Backward Pass: " + (totalBackward/clockspersecond()) + " s" + nl
see "Optimizer    : " + (totalUpdate/clockspersecond()) + " s" + nl
see "Garbage Coll : " + (totalGC/clockspersecond()) + " s" + nl
