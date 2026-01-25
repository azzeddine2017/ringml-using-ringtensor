# Project: RingML
# File: tests/test_step1.ring
# Description: Step 1: Core Tensor Operations (Pointer Mode)
# Author: Azzeddine Remmal

load "ringml.ring"

decimals(4)
see "=== Step 1: Core Tensor Operations (Pointer Mode) ===" + nl

# 1. Initialization & Fill
see "1. Testing Init & Fill..." + nl
t1 = new Tensor(2, 2)
t1.fill(1.5)
val = t1.getVal(1, 1)

if val = 1.5
    see "   [PASS] Init/Fill working." + nl
else
    see "   [FAIL] Expected 1.5, got " + val + nl
ok

# 2. Addition
see "2. Testing Add..." + nl
t2 = new Tensor(2, 2)
t2.fill(0.5)
t1.add(t2) # 1.5 + 0.5 = 2.0
val = t1.getVal(1, 1)

if val = 2.0
    see "   [PASS] Add working." + nl
else
    see "   [FAIL] Expected 2.0, got " + val + nl
ok

# 3. MatMul
see "3. Testing MatMul..." + nl
# A = [[1, 2]] (1x2)
mA = new Tensor(1, 2)
mA.setVal(1, 1, 1.0)
mA.setVal(1, 2, 2.0)

# B = [[3], [4]] (2x1)
mB = new Tensor(2, 1)
mB.setVal(1, 1, 3.0)
mB.setVal(2, 1, 4.0)

# Res = (1*3 + 2*4) = 11
mRes = mA.matmul(mB)
resVal = mRes.getVal(1, 1)

if resVal = 11.0
    see "   [PASS] MatMul working (11.0)." + nl
else
    see "   [FAIL] Expected 11.0, got " + resVal + nl
ok

# 4. Transpose
see "4. Testing Transpose..." + nl
mT = mA.transpose() # Should be (2, 1) -> [[1], [2]]
if mT.nRows = 2 and mT.getVal(2, 1) = 2.0
    see "   [PASS] Transpose working." + nl
else
    see "   [FAIL] Transpose dimensions or values wrong." + nl
ok