


func testInPlaceLoad
    ? "═══════════════════════════════════════════════"
    ? "IN-PLACE LOAD TEST"
    ? "═══════════════════════════════════════════════"
    
    # Create and save
    oT1 = new Tensor(3, 4)
    for r = 1 to 3
        for c = 1 to 4
            oT1.setVal(r, c, r * c * 1.5)
        next
    next
    
    ? "Original [1,1]: " + oT1.getVal(1, 1)
    ? "Original [3,4]: " + oT1.getVal(3, 4)
    
    cFile = "test_inplace.bin"
    oT1.saveFile(cFile)
    ? "Saved."
    ? ""
    
    # Create another tensor
    oT2 = new Tensor(5, 5)  # Different size
    oT2.fill(99.0)
    
    ? "Before load:"
    ? "  Shape: " + oT2.nRows + " × " + oT2.nCols
    ? "  [1,1]: " + oT2.getVal(1, 1)
    ? ""
    
    # Load in-place
    oT2.loadFile(cFile)
    
    ? "After load:"
    ? "  Shape: " + oT2.nRows + " × " + oT2.nCols
    ? "  [1,1]: " + oT2.getVal(1, 1)
    ? "  [3,4]: " + oT2.getVal(3, 4)
    ? ""
    
    # Compare
    if fabs(oT1.getVal(1, 1) - oT2.getVal(1, 1)) < 0.0001 and
       fabs(oT1.getVal(3, 4) - oT2.getVal(3, 4)) < 0.0001
        ? "✅ PASS: In-place load works!"
    else
        ? "❌ FAIL: Values don't match"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    if fexists(cFile) remove(cFile) ok

func testSaveLoadWithDebug
    ? "═══════════════════════════════════════════════"
    ? "SAVE/LOAD DEBUG TEST"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 1  # ✅ طبقة واحدة للتبسيط
    
    # ═══════════════════════════════════════════════
    # 1. Create and train
    # ═══════════════════════════════════════════════
    ? nl + "1️⃣ Creating model..."
    oModel1 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    
    # Check how many weights
    aWeights1 = oModel1.getAllWeights()
    ? "   Total weight tensors: " + len(aWeights1)
    ? ""
    
    # Sample first weight before training
    if len(aWeights1) > 0
        ? "   Sample weight [1,1] BEFORE training: " + 
          aWeights1[1].getVal(1, 1)
    ok
    ? ""
    
    # Train
    ? "2️⃣ Training..."
    oOptim = new Adam(0.01, 0.0)  # Higher LR for faster convergence
    oLoss = new CrossEntropyLoss
    
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    oTarget.setVal(1, 10, 1.0)
    
    for iter = 1 to 30
        oLogits = oModel1.forward(oInput)
        nLoss = oLoss.calculate(oLogits, oTarget)
        oGrad = oLoss.backwardTensor()
        oModel1.backward(oGrad)
        aPtrs = oModel1.getAllGradients()
        tensor_clip_global_norm(aPtrs, 5.0)
        oModel1.updateWeights(oOptim)
    next
    
    oLogits1 = oModel1.forward(oInput)
    nLoss1 = oLoss.calculate(oLogits1, oTarget)
    
    ? "   Final loss: " + nLoss1
    ? "   Sample weight [1,1] AFTER training: " + 
      aWeights1[1].getVal(1, 1)
    ? ""
    
    # ═══════════════════════════════════════════════
    # 3. Save
    # ═══════════════════════════════════════════════
    ? "3️⃣ Saving..."
    cFile = "test_debug.rdata"
    oModel1.saveModel(cFile)
    ? "   Saved " + len(aWeights1) + " tensors"
    ? ""
    
    # ═══════════════════════════════════════════════
    # 4. Load
    # ═══════════════════════════════════════════════
    ? "4️⃣ Loading..."
    oModel2 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    
    # Check weights before load
    aWeights2Before = oModel2.getAllWeights()
    ? "   Weight tensors in new model: " + len(aWeights2Before)
    ? "   Sample weight [1,1] BEFORE load: " + 
      aWeights2Before[1].getVal(1, 1)
    ? ""
    
    oModel2.loadModel(cFile)
    
    # Check weights after load
    aWeights2After = oModel2.getAllWeights()
    ? "   Sample weight [1,1] AFTER load: " + 
      aWeights2After[1].getVal(1, 1)
    ? ""
    
    # ═══════════════════════════════════════════════
    # 5. Compare
    # ═══════════════════════════════════════════════
    ? "5️⃣ Comparing..."
    
    ? "   Original weight [1,1]: " + aWeights1[1].getVal(1, 1)
    ? "   Loaded weight [1,1]:   " + aWeights2After[1].getVal(1, 1)
    ? "   Difference:            " + 
      fabs(aWeights1[1].getVal(1, 1) - aWeights2After[1].getVal(1, 1))
    ? ""
    
    oLogits2 = oModel2.forward(oInput)
    nLoss2 = oLoss.calculate(oLogits2, oTarget)
    
    ? "   Original loss: " + nLoss1
    ? "   Loaded loss:   " + nLoss2
    ? "   Difference:    " + fabs(nLoss1 - nLoss2)
    ? ""
    
    # ═══════════════════════════════════════════════
    # 6. Verdict
    # ═══════════════════════════════════════════════
    
    if fabs(nLoss1 - nLoss2) < 0.01
        ? "✅ PASS: Weights loaded correctly!"
    else
        ? "❌ FAIL: Weights NOT loaded correctly!"
        ? ""
        ? "Debug info:"
        ? "  - Check if getAllWeights() returns correct references"
        ? "  - Check if tensor.loadFile() works"
        ? "  - Check tensor count: saved=" + len(aWeights1) + 
          ", loaded=" + len(aWeights2After)
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    # Cleanup
    if fexists(cFile) remove(cFile) ok