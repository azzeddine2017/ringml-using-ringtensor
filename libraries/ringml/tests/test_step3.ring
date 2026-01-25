# Project: RingML
# File: tests/test_step3.ring
# Description: Step 3: Full Training Step (Backprop)
# Author: Azzeddine Remmal

load "ringml.ring"

decimals(4)

see "=== Step 3: Full Training Step (Backprop) ===" + nl

# Data: Input [1.0], Target [0.0]
input = new Tensor(1, 1)
input.setVal(1, 1, 1.0)

target = new Tensor(1, 1)
target.setVal(1, 1, 0.0)

# Model: 1 -> 1 (Linear)
layer = new Dense(1, 1)
layer.oWeights.setVal(1, 1, 0.5) # Initial weight 0.5
layer.oBias.setVal(1, 1, 0.0)

optimizer = new SGD(0.1) # Learning Rate 0.1
criterion = new MSELoss

see "Initial Weight: " + layer.oWeights.getVal(1, 1) + nl

# --- Step ---
# Forward
pred = layer.forward(input) # 1.0 * 0.5 = 0.5
loss = criterion.forward(pred, target) 
# MSE = (0.5 - 0)^2 = 0.25

see "Prediction: " + pred.getVal(1, 1) + nl
see "Loss: " + loss + nl

# Backward
# Grad = 2 * (Pred - Target) / N = 2 * (0.5 - 0) / 1 = 1.0
grad = criterion.backward(pred, target)
layer.backward(grad)

# Update
# Weight = Weight - (LR * Grad)
# New = 0.5 - (0.1 * 1.0 * Input) ... roughly
# Let's just check if it changes
optimizer.update(layer)

newWeight = layer.oWeights.getVal(1, 1)
see "New Weight: " + newWeight + nl

if newWeight < 0.5
    see "   [PASS] Weight decreased towards target." + nl
else
    see "   [FAIL] Weight did not improve." + nl
ok