
    

func testFullTrainingPipeline
    ? "═══════════════════════════════════════════════"
    ? "           FULL TRAINING PIPELINE -            "
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    
    # ✅ Base parameters
    base_lr = 0.001
    min_lr = 0.00001
    warmup_steps = 10
    decay_rate = 0.95
    
    oOptim = new Adam(base_lr, 0.0)
    oLoss = new CrossEntropyLoss
    
    # Setup data
    oInput = new Tensor(2, nSeqLen)
    for b = 1 to 2
        for i = 1 to nSeqLen
            oInput.setVal(b, i, i + 4 + (b - 1) * 2)
        next
    next
    
    oTarget = new Tensor(2 * nSeqLen, nVocab)
    oTarget.fill(0)
    
    for pos = 1 to 2 * nSeqLen
        targetClass = ((pos - 1) % 10) + 5
        if targetClass > nVocab targetClass = nVocab ok
        oTarget.setVal(pos, targetClass, 1.0)
    next
    
    ? "Setup:"
    ? "  Batch size: 2"
    ? "  Base LR: " + base_lr
    ? "  Warmup: " + warmup_steps + " steps"
    ? "  Decay rate: " + decay_rate
    ? ""
    
    nBestLoss = 999
    nWorseCount = 0
    nPatienceMax = 20  # Early stopping patience
    
    for iter = 1 to 300
        
        # ═══════════════════════════════════════════
        # LEARNING RATE SCHEDULE
        # ═══════════════════════════════════════════
        if iter <= warmup_steps
            # Warmup: linear increase
            current_lr = base_lr * (iter / warmup_steps)
        else
            # Decay: exponential decrease
            decay_steps = iter - warmup_steps
            current_lr = base_lr * pow(decay_rate, (decay_steps / 10.0))
            
            # Floor at min_lr
            if current_lr < min_lr
                current_lr = min_lr
            ok
        ok
        
        oOptim.lr = current_lr
        
        # ═══════════════════════════════════════════
        # TRAINING STEP
        # ═══════════════════════════════════════════
        
        # Forward
        oLogits = oModel.forward(oInput)
        nErr = oLoss.calculate(oLogits, oTarget)
        
        # Check for Null
        if isNull(nErr) or nErr > 100
            ? "❌ Training collapsed at iteration " + iter
            ? "   Loss: " + nErr
            ? "   LR: " + current_lr
            exit
        ok
        
        # Backward
        oGrad = oLoss.backwardTensor()
        oModel.backward(oGrad)
        
        # Gradient clipping
        aPtrs = oModel.getAllGradients()
        nGradNorm = tensor_clip_global_norm(aPtrs, 5.0)
        
        # ✅ ADDITIONAL SAFEGUARD
        # If gradients explode even after clipping, reduce LR immediately
        if nGradNorm > 20.0 and iter > 50
            oOptim.lr = oOptim.lr * 0.5
            ? "  ⚠️ Emergency LR reduction! New LR: " + oOptim.lr
        ok
        
        # Update
        oModel.updateWeights(oOptim)
        
        # ═══════════════════════════════════════════
        # MONITORING
        # ═══════════════════════════════════════════
        
        if iter % 15 = 0 or iter <= 5
            ? "Iter " + iter + ":"
            ? "  Loss: " + nErr
            ? "  LR: " + current_lr
            ? "  Grad norm: " + nGradNorm
            
            if nErr < nBestLoss
                nBestLoss = nErr
                nWorseCount = 0
                ? "  ✅ New best!"
            else
                nWorseCount++
                if nWorseCount >= 3
                    ? "  ⚠️ Not improving (" + nWorseCount + " iters)"
                ok
            ok
            ? ""
        ok
        
        # ═══════════════════════════════════════════
        # EARLY STOPPING
        # ═══════════════════════════════════════════
        
        if nErr < 0.05
            ? "✅ Converged at iteration " + iter + "!"
            exit
        ok
        
        if nWorseCount >= nPatienceMax
            ? "⚠️ Early stopping - no improvement for " + nPatienceMax + " iterations"
            exit
        ok
        
        # Stop if gradients are consistently exploding
        if nGradNorm > 50.0 and iter > 100
            ? "❌ Stopping - gradient explosion detected"
            exit
        ok
    next
    
    ? "═══════════════════════════════════════════════"
    ? "FINAL RESULTS:"
    ? "═══════════════════════════════════════════════"
    ? "Best loss: " + nBestLoss
    ? "Final loss: " + nErr
    ? "Final LR: " + current_lr
    ? "Final grad norm: " + nGradNorm
    ? ""
    
    if nBestLoss < 0.1
        ? "✅ EXCELLENT: Converged perfectly!"
    elseif nBestLoss < 0.5
        ? "✅ GOOD: Model learned well"
    elseif nBestLoss < 1.5
        ? "✅ PASS: Model is learning"
    else
        ? "❌ FAIL: Model struggled"
    ok
    
    ? "═══════════════════════════════════════════════" + nl