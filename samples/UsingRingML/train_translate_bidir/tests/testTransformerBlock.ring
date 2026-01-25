
    

func testTransformerBlock
    ? "═══════════════════════════════════════════════"
    ? "TRANSFORMER BLOCK TEST"
    ? "═══════════════════════════════════════════════"
    
    nDim = 32
    nSeq = 8
    nBatch = 2
    nLayer = 1
    
    oBlock = new TransformerBlock(nDim, nSeq, nLayer)
    
    # Input: (batch*seq, dim)
    oInput = new Tensor(nBatch * nSeq, nDim)
    oInput.random()
    oInput.scalarMul(0.1)
    
    ? "Forward pass..."
    oOutput = oBlock.forward(oInput)
    
    ? "  Input shape: (" + oInput.nRows + " × " + oInput.nCols + ")"
    ? "  Output shape: (" + oOutput.nRows + " × " + oOutput.nCols + ")"
    
    if oOutput.nRows != oInput.nRows or oOutput.nCols != oInput.nCols
        ? "❌ ERROR: Shape mismatch!"
        return
    ok
    
    ? "  ✅ Shapes match"
    ? ""
    
    # Backward pass
    ? "Backward pass..."
    oGrad = new Tensor(oOutput.nRows, oOutput.nCols)
    oGrad.fill(1.0)
    
    oGradInput = oBlock.backward(oGrad)
    
    ? "  Gradient shape: (" + oGradInput.nRows + " × " + oGradInput.nCols + ")"
    
    if oGradInput.nRows != oInput.nRows or oGradInput.nCols != oInput.nCols
        ? "❌ ERROR: Gradient shape mismatch!"
        return
    ok
    
    ? "  ✅ Gradient shapes correct"
    ? ""
    
    # Check gradients are not zero
    nSum = 0
    for i = 1 to min(100, oGradInput.nRows * oGradInput.nCols)
        r = ((i-1) / oGradInput.nCols) + 1
        c = ((i-1) % oGradInput.nCols) + 1
        nSum += fabs(oGradInput.getVal(r, c))
    next
    nAvg = nSum / 100
    
    ? "Average gradient magnitude: " + nAvg
    
    if nAvg < 0.001
        ? "⚠️ WARNING: Gradients very small (vanishing?)"
    elseif nAvg > 100
        ? "⚠️ WARNING: Gradients very large (exploding?)"
    else
        ? "✅ Gradients in reasonable range"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
