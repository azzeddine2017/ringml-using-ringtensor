

func testAdamModel2WithDiagnostics
    ? "═══════════════════════════════════════════════"
    ? "ADAM MODEL 2 - WITH DIAGNOSTICS"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oOptim = new Adam(0.001, 0) 
    oLoss = new CrossEntropyLoss
    
    # Input & Target
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    
    for pos = 1 to nSeqLen
        targetClass = pos + 5
        if targetClass > nVocab targetClass = nVocab ok
        oTarget.setVal(pos, targetClass, 1.0)
    next
    
    ? "Training with diagnostics..."
    ? ""
    
    nPrevLoss = 999
    nStuckCount = 0
    
    for iter = 1 to 200
        # Forward
        oLogits = oModel.forward(oInput)
        nErr = oLoss.calculate(oLogits, oTarget)
        
        # Check for NaN/Inf
        if isnull(nErr) or nErr > 1000
            ? "❌ CRITICAL: Loss exploded at iteration " + iter
            ? "   Loss = " + nErr
            exit
        ok
        
        # Backward
        oGrad = oLoss.backwardTensor()
        oModel.backward(oGrad)
        
        # Clip and get norm
        aPtrs = oModel.getAllGradients()
        nGradNorm = clipGlobalNorm(aPtrs, 1.0)  
        
        # Update
        oModel.updateWeights(oOptim)
        
        # Monitor
        if iter % 10 = 0 or iter <= 5
            improvement = 0
            if nPrevLoss > 0
                improvement = ((nPrevLoss - nErr) / nPrevLoss) * 100
            ok
            
            ? "Iter " + iter + ":"
            ? "  Loss: " + nErr
            ? "  Grad Norm: " + nGradNorm
            ? "  Improvement: " + improvement + "%"
            
            if nGradNorm > 50
                ? "  ⚠️ WARNING: High gradient norm!"
            ok
            
            if iter > 10 and nErr > nPrevLoss * 1.5
                ? "  ⚠️ WARNING: Loss increasing!"
            ok
            
            ? ""
            nPrevLoss = nErr
        ok
        
        # Early stopping if stuck
        if fabs(nErr - nPrevLoss) < 0.0001
            nStuckCount++
            if nStuckCount > 20
                ? "⚠️ Converged at iteration " + iter
                exit
            ok
        else
            nStuckCount = 0
        ok
    next
    
    ? ""
    if nErr < 1.0
        ? "✅ PASS: Model learning successfully!"
    elseif nErr < 3.0
        ? "⚠️ PARTIAL: Model learning slowly"
    else
        ? "❌ FAIL: Model not learning"
    ok
    
    ? "═══════════════════════════════════════════════" + nl

