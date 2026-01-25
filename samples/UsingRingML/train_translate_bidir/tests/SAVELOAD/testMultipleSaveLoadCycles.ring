

func testMultipleSaveLoadCycles
    ? "═══════════════════════════════════════════════"
    ? "MULTIPLE SAVE/LOAD CYCLES TEST"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oOptim = new Adam(0.001, 0.0)
    oLoss = new CrossEntropyLoss
    
    # Test data
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    oTarget.setVal(1, 10, 1.0)
    
    aLosses = []
    nCycles = 3
    
    ? "Testing " + nCycles + " save/load cycles..."
    ? ""
    
    for cycle = 1 to nCycles
        ? "Cycle " + cycle + ":"
        
        # Train
        ? "  Training..."
        for iter = 1 to 20
            oLogits = oModel.forward(oInput)
            nLoss = oLoss.calculate(oLogits, oTarget)
            
            oGrad = oLoss.backwardTensor()
            oModel.backward(oGrad)
            
            aPtrs = oModel.getAllGradients()
            tensor_clip_global_norm(aPtrs, 5.0)
            
            oModel.updateWeights(oOptim)
        next
        
        # Test
        oLogits = oModel.forward(oInput)
        nLoss = oLoss.calculate(oLogits, oTarget)
        aLosses + nLoss
        
        ? "  Loss: " + nLoss
        
        # Save
        cFile = "test_cycle_" + cycle + ".rdata"
        oModel.saveModel(cFile)
        ? "  Saved to: " + cFile
        
        # Load into new model
        if cycle < nCycles
            oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
            oModel.loadModel(cFile)
            oOptim = new Adam(0.001, 0.0)
            ? "  Loaded for next cycle"
        ok
        
        ? ""
    next
    
    ? "═══════════════════════════════════════════════"
    ? "CYCLE RESULTS:"
    ? "═══════════════════════════════════════════════"
    
    for i = 1 to len(aLosses)
        ? "Cycle " + i + ": Loss = " + aLosses[i]
    next
    
    ? ""
    
    # Check monotonic decrease
    bMonotonic = true
    for i = 2 to len(aLosses)
        if aLosses[i] >= aLosses[i-1]
            ? "⚠️ Warning: Loss increased at cycle " + i
            bMonotonic = false
        ok
    next
    
    if bMonotonic
        ? "✅ Loss decreased monotonically across cycles"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    # Cleanup
    for cycle = 1 to nCycles
        cFile = "test_cycle_" + cycle + ".rdata"
        if fexists(cFile)
            remove(cFile)
        ok
    next
    
    ? "[Cleanup] Test files removed" + nl
