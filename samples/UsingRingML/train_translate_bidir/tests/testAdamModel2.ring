

func testAdamModel2
    ? "═══════════════════════════════════════════════"
    ? "ADAM MODEL 2 - MULTI-LAYER TEST"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2  # 2 transformer blocks
    
    # Create model
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oOptim = new Adam(0.001, 0.0001)
    oLoss = new CrossEntropyLoss
    
    # Input
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    # Target (flattened)
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    
    for pos = 1 to nSeqLen
        targetClass = pos + 5
        if targetClass > nVocab targetClass = nVocab ok
        oTarget.setVal(pos, targetClass, 1.0)
    next
    
    ? "Training multi-layer model..."
    ? ""
    
    nPrevLoss = 999
    
    for iter = 1 to 100
        # Forward
        oLogits = oModel.forward(oInput)
        nErr = oLoss.calculate(oLogits, oTarget)
        
        # Backward
        oGrad = oLoss.backwardTensor()
        oModel.backward(oGrad)
        
        # Clip
        aPtrs = oModel.getAllGradients()
        
        clipGlobalNorm(aPtrs, 1)
        
        # Update
        oModel.updateWeights(oOptim)
        
        if iter % 20 = 0 or iter <= 5
            improvement = ((nPrevLoss - nErr) / max(nPrevLoss, 0.001)) * 100
            ? "Iter " + iter + ": Loss = " + nErr + 
              " (improvement: " + improvement + "%)"
            nPrevLoss = nErr
        ok
    next
    

    ? ""
    if nErr < 1.0
        ? "✅ PASS: Multi-layer model learning!"
    else
        ? "❌ FAIL: Model not learning"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
