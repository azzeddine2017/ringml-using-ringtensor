# Project: RingML
# File: tests/test_step2.ring
# Description: Step 2: Layers Forward Pass
# Author: Azzeddine Remmal

load "ringml.ring"

decimals(4)
see "=== Step 2: Layers Forward Pass ===" + nl

# 1. Setup Input
input = new Tensor(1, 2)
input.setVal(1, 1, 0.5)
input.setVal(1, 2, 0.1)

# 2. Setup Layer manually (to check calculation)
layer = new Dense(2, 1)

# Force weights/bias to known values
# Weights = [[0.8], [0.2]]
layer.oWeights.setVal(1, 1, 0.8)
layer.oWeights.setVal(2, 1, 0.2)

# Bias = [0.1]
layer.oBias.setVal(1, 1, 0.1)

# 3. Forward
# Expected: (0.5 * 0.8) + (0.1 * 0.2) + 0.1 
#         = 0.40 + 0.02 + 0.1 = 0.52
output = layer.forward(input)
res = output.getVal(1, 1)

see "Input: [0.5, 0.1]" + nl
see "Weights: [[0.8], [0.2]] + Bias 0.1" + nl
see "Result: " + res + nl

if fabs(res - 0.52) < 0.0001
    see "   [PASS] Dense Forward Correct." + nl
else
    see "   [FAIL] Expected 0.52" + nl
ok

# 4. Activation (ReLU)
relu = new ReLU
# Input 0.52 -> Output 0.52
outRelu = relu.forward(output)
see "ReLU Output: " + outRelu.getVal(1, 1) + nl