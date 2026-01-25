load "ringml.ring"

load "../../AdamModel2.ring"
load "../../Inference.ring"
load "../../BiDirectionalDataset.ring"

load "testSaveLoadAdamModel2.ring"
load "testMultipleSaveLoadCycles.ring"
load "testQuantizedSaveLoad.ring"
load "testSaveLoadWithDebug.ring"


? "num cores :" + tensor_get_cores()
tensor_set_threads(2)

decimals(8)


func main

   testSaveLoadWithDebug()
   testInPlaceLoad()
   
    # Test 1: Basic save/load
    testSaveLoadAdamModel2()
    
    # Test 2: Multiple cycles
    testMultipleSaveLoadCycles()
    
    # Test 3: Quantization
    testQuantizedSaveLoad()

/*
λ ring main.ring
num cores :4
═══════════════════════════════════════════════
SAVE/LOAD DEBUG TEST
═══════════════════════════════════════════════

1️⃣ Creating model...
Initializing Adam Model (1 Layers)...

  Building Block 1...
   Total weight tensors: 36

   Sample weight [1,1] BEFORE training: 0.00490585

2️⃣ Training...
   Final loss: 0.00000000
   Sample weight [1,1] AFTER training: -0.03577164

3️⃣ Saving...
[Serializer] Saving Model to test_debug.rdata...
[Serializer] Done. Size: 0.05216217 MB
   Saved 36 tensors

4️⃣ Loading...
Initializing Adam Model (1 Layers)...

  Building Block 1...
   Weight tensors in new model: 36
   Sample weight [1,1] BEFORE load: 0.00897885

[Serializer] Loading Model...
[Serializer] Loaded Successfully.
   Sample weight [1,1] AFTER load: -0.03577165

5️⃣ Comparing...
   Original weight [1,1]: -0.03577164
   Loaded weight [1,1]:   -0.03577165
   Difference:            0.00000000

   Original loss: 0.00000000
   Loaded loss:   0.00000000
   Difference:    0

✅ PASS: Weights loaded correctly!
═══════════════════════════════════════════════

═══════════════════════════════════════════════
IN-PLACE LOAD TEST
═══════════════════════════════════════════════
Original [1,1]: 1.50000000
Original [3,4]: 18
Saved.

Before load:
  Shape: 5 × 5
  [1,1]: 99

After load:
  Shape: 5 × 5
  [1,1]: 1.50000000
  [3,4]: 18

✅ PASS: In-place load works!
═══════════════════════════════════════════════

═══════════════════════════════════════════════
SAVE/LOAD TEST - AdamModel2
═══════════════════════════════════════════════

1️⃣ Creating and training model...
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
   Training for 50 iterations...
   Initial loss: ~3.0
   Final loss: 0.00000304

2️⃣ Saving model...
[Serializer] Saving Model to test_adam2_model.rdata...
[Serializer] Done. Size: 0.10088348 MB
   ✅ Model saved to: test_adam2_model.rdata
   File size: 103.30468750 KB

3️⃣ Loading into new model...
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
[Serializer] Loading Model...
[Serializer] Loaded Successfully.
   ✅ Model loaded successfully

4️⃣ Comparing models...

   Original model loss: 0.00000273
   Loaded model loss:   0.00000273
   Difference:          0.00000000

   Comparing logits (first 5 values):
     [1] Original: -8.19922971 | Loaded: -8.19922974 | Diff: 0.00000003
     [2] Original: -8.28071427 | Loaded: -8.28071431 | Diff: 0.00000004
     [3] Original: -8.26743838 | Loaded: -8.26743833 | Diff: 0.00000005
     [4] Original: -8.49760069 | Loaded: -8.49760065 | Diff: 0.00000003
     [5] Original: -6.01534267 | Loaded: -6.01534267 | Diff: 0.00000001

   Max difference: 0.00000005

5️⃣ Continuing training with loaded model...
   Loss before: 0.00000273
   Loss after:  0.00000044
   Improvement: 83.97379075%

═══════════════════════════════════════════════
RESULTS:
═══════════════════════════════════════════════
✅ Loss match: Diff = 0.00000000
✅ Logits match: Max diff = 0.00000005
✅ Can continue training

✅✅✅ ALL TESTS PASSED ✅✅✅
═══════════════════════════════════════════════

[Cleanup] Test file removed

═══════════════════════════════════════════════
MULTIPLE SAVE/LOAD CYCLES TEST
═══════════════════════════════════════════════
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
Testing 3 save/load cycles...

Cycle 1:
  Training...
  Loss: 0.15558189
[Serializer] Saving Model to test_cycle_1.rdata...
[Serializer] Done. Size: 0.10088348 MB
  Saved to: test_cycle_1.rdata
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
[Serializer] Loading Model...
[Serializer] Loaded Successfully.
  Loaded for next cycle

Cycle 2:
  Training...
  Loss: 0.00004315
[Serializer] Saving Model to test_cycle_2.rdata...
[Serializer] Done. Size: 0.10088348 MB
  Saved to: test_cycle_2.rdata
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
[Serializer] Loading Model...
[Serializer] Loaded Successfully.
  Loaded for next cycle

Cycle 3:
  Training...
  Loss: 0.00000011
[Serializer] Saving Model to test_cycle_3.rdata...
[Serializer] Done. Size: 0.10088348 MB
  Saved to: test_cycle_3.rdata

═══════════════════════════════════════════════
CYCLE RESULTS:
═══════════════════════════════════════════════
Cycle 1: Loss = 0.15558189
Cycle 2: Loss = 0.00004315
Cycle 3: Loss = 0.00000011

✅ Loss decreased monotonically across cycles
═══════════════════════════════════════════════

[Cleanup] Test files removed

═══════════════════════════════════════════════
QUANTIZED SAVE/LOAD TEST
═══════════════════════════════════════════════
1️⃣ Training model...
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
   Trained loss: 0.00007049

2️⃣ Saving with quantization...
[Serializer] Saving Model to test_quantized.rdata...
[Serializer] Done. Size: 0.10088348 MB
   Quantized size: 103.30468750 KB

3️⃣ Loading quantized model...
Initializing Adam Model (2 Layers)...

  Building Block 1...

  Building Block 2...
[Serializer] Loading Model...
[Serializer] Loaded Successfully.
   Original loss:  0.00007049
   Quantized loss: 0.00007049
   Difference:     0.00000000

4️⃣ Comparing with full precision...
[Serializer] Saving Model to test_full.rdata...
[Serializer] Done. Size: 0.20122528 MB
   Full precision: 206.05468750 KB
   Quantized:      103.30468750 KB
   Compression:    49.86540284%

✅ Quantization preserves accuracy
═══════════════════════════════════════════════

*/
