


func testGradientCollection
    ? "═══════════════════════════════════════════════"
    ? "GRADIENT COLLECTION TEST"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oLoss = new CrossEntropyLoss
    
    # Simple input/target
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    for pos = 1 to nSeqLen
        oTarget.setVal(pos, 10, 1.0)
    next
    
    # Forward
    oLogits = oModel.forward(oInput)
    nErr = oLoss.calculate(oLogits, oTarget)
    
    ? "Loss: " + nErr
    
    # Backward
    oGrad = oLoss.backwardTensor()
    oModel.backward(oGrad)
    
    # Collect gradients
    ? nl + "Collecting gradients..."
    aGrads = oModel.getAllGradients()
    
    ? "Total gradient pointers: " + len(aGrads)
    
    if len(aGrads) = 0
        ? "❌ ERROR: No gradients collected!"
        return
    ok
    
    # Check each gradient
    ? nl + "Checking gradient pointers:"
    nValidCount = 0
    nNullCount = 0
    
    for i = 1 to len(aGrads)
        pGrad = aGrads[i]
        
        if isNull(pGrad) or pGrad = 0
            nNullCount++
            if i <= 10
                ? "  Grad[" + i + "]: NULL ❌"
            ok
        else
            nValidCount++
            if i <= 10
                ? "  Grad[" + i + "]: " see pGrad + " ✅"
            ok
        ok
    next
    
    ? nl + "Summary:"
    ? "  Valid pointers: " + nValidCount
    ? "  Null pointers: " + nNullCount
    
    if nValidCount = 0
        ? nl + "❌ FAIL: All gradients are NULL!"
    elseif nNullCount > 0
        ? nl + "⚠️ WARNING: Some gradients are NULL"
    else
        ? nl + "✅ PASS: All gradient pointers valid"
        
        # Test clipping
        ? nl + "Testing gradient clipping..."
        nGradNorm = clipGlobalNorm(aGrads, 1.0)
        ? "Gradient norm: " + nGradNorm
        
        if nGradNorm = 0
            ? "❌ ERROR: Gradient norm is 0!"
        elseif nGradNorm < 0.01
            ? "⚠️ WARNING: Very small gradients (vanishing?)"
        elseif nGradNorm > 1000
            ? "⚠️ WARNING: Very large gradients (exploding?)"
        else
            ? "✅ Gradient norm in normal range"
        ok
    ok
    
    ? "═══════════════════════════════════════════════" + nl
