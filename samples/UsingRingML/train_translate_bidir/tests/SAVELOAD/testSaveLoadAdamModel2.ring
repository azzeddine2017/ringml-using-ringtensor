




func testSaveLoadAdamModel2
    ? "═══════════════════════════════════════════════"
    ? "SAVE/LOAD TEST - AdamModel2"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    # ═══════════════════════════════════════════════
    # STEP 1: Create and train model
    # ═══════════════════════════════════════════════
    ? nl + "1️⃣ Creating and training model..."
    
    oModel1 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oOptim = new Adam(0.001, 0.0)
    oLoss = new CrossEntropyLoss
    
    # Simple training data
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    for pos = 1 to nSeqLen
        oTarget.setVal(pos, 10, 1.0)
    next
    
    # Train for 50 iterations
    ? "   Training for 50 iterations..."
    nLoss1 = 0
    
    for iter = 1 to 50
        oLogits = oModel1.forward(oInput)
        nLoss1 = oLoss.calculate(oLogits, oTarget)
        
        oGrad = oLoss.backwardTensor()
        oModel1.backward(oGrad)
        
        aPtrs = oModel1.getAllGradients()
        tensor_clip_global_norm(aPtrs, 5.0)
        
        oModel1.updateWeights(oOptim)
    next
    
    ? "   Initial loss: ~3.0"
    ? "   Final loss: " + nLoss1
    ? ""
    
    # ═══════════════════════════════════════════════
    # STEP 2: Save model
    # ═══════════════════════════════════════════════
    ? "2️⃣ Saving model..."
    
    cModelFile = "test_adam2_model.rdata"
    
    try
        oModel1.saveModel(cModelFile)
        ? "   ✅ Model saved to: " + cModelFile
    catch
        ? "   ❌ ERROR saving model: " + cCatchError
        return
    done
    
    # Check file exists
    if !fexists(cModelFile)
        ? "   ❌ ERROR: File not created!"
        return
    ok
    
    nFileSize = len(read(cModelFile))
    ? "   File size: " + (nFileSize / 1024.0) + " KB"
    ? ""
    
    # ═══════════════════════════════════════════════
    # STEP 3: Create new model and load weights
    # ═══════════════════════════════════════════════
    ? "3️⃣ Loading into new model..."
    
    oModel2 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    
    try
        oModel2.loadModel(cModelFile)
        ? "   ✅ Model loaded successfully"
    catch
        ? "   ❌ ERROR loading model: " + cCatchError
        return
    done
    ? ""
    
    # ═══════════════════════════════════════════════
    # STEP 4: Compare predictions
    # ═══════════════════════════════════════════════
    ? "4️⃣ Comparing models..."
    ? ""
    
    # Prediction with original model
    oLogits1 = oModel1.forward(oInput)
    nLoss1 = oLoss.calculate(oLogits1, oTarget)
    
    # Prediction with loaded model
    oLogits2 = oModel2.forward(oInput)
    nLoss2 = oLoss.calculate(oLogits2, oTarget)
    
    ? "   Original model loss: " + nLoss1
    ? "   Loaded model loss:   " + nLoss2
    ? "   Difference:          " + fabs(nLoss1 - nLoss2)
    ? ""
    
    # Compare first few logits
    ? "   Comparing logits (first 5 values):"
    nMaxDiff = 0
    
    for i = 1 to 5
        val1 = oLogits1.getVal(1, i)
        val2 = oLogits2.getVal(1, i)
        diff = fabs(val1 - val2)
        
        ? "     [" + i + "] Original: " + val1 + 
          " | Loaded: " + val2 + 
          " | Diff: " + diff
        
        if diff > nMaxDiff
            nMaxDiff = diff
        ok
    next
    
    ? ""
    ? "   Max difference: " + nMaxDiff
    ? ""
    
    # ═══════════════════════════════════════════════
    # STEP 5: Verify by continuing training
    # ═══════════════════════════════════════════════
    ? "5️⃣ Continuing training with loaded model..."
    
    oOptim2 = new Adam(0.001, 0.0)
    nLossBefore = nLoss2
    
    for iter = 1 to 10
        oLogits2 = oModel2.forward(oInput)
        nLoss2 = oLoss.calculate(oLogits2, oTarget)
        
        oGrad = oLoss.backwardTensor()
        oModel2.backward(oGrad)
        
        aPtrs = oModel2.getAllGradients()
        tensor_clip_global_norm(aPtrs, 5.0)
        
        oModel2.updateWeights(oOptim2)
    next
    
    ? "   Loss before: " + nLossBefore
    ? "   Loss after:  " + nLoss2
    ? "   Improvement: " + ((nLossBefore - nLoss2) / nLossBefore * 100) + "%"
    ? ""
    
    # ═══════════════════════════════════════════════
    # STEP 6: Final verdict
    # ═══════════════════════════════════════════════
    ? "═══════════════════════════════════════════════"
    ? "RESULTS:"
    ? "═══════════════════════════════════════════════"
    
    bTestPassed = true
    
    # Check 1: Loss difference
    if fabs(nLoss1 - nLossBefore) > 0.01
        ? "❌ Loss mismatch: " + fabs(nLoss1 - nLossBefore)
        bTestPassed = false
    else
        ? "✅ Loss match: Diff = " + fabs(nLoss1 - nLossBefore)
    ok
    
    # Check 2: Logits difference
    if nMaxDiff > 0.001
        ? "❌ Logits mismatch: Max diff = " + nMaxDiff
        bTestPassed = false
    else
        ? "✅ Logits match: Max diff = " + nMaxDiff
    ok
    
    # Check 3: Can continue training
    if nLoss2 >= nLossBefore
        ? "⚠️ Warning: No improvement after loading"
    else
        ? "✅ Can continue training"
    ok
    
    ? ""
    
    if bTestPassed
        ? "✅✅✅ ALL TESTS PASSED ✅✅✅"
    else
        ? "❌ SOME TESTS FAILED"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    # Cleanup
    if fexists(cModelFile)
        remove(cModelFile)
        ? "[Cleanup] Test file removed" + nl
    ok

