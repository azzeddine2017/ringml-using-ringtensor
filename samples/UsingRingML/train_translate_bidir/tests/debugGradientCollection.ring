


func debugGradientCollection
    ? "═══════════════════════════════════════════════"
    ? "DEBUG: GRADIENT COLLECTION"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 1  
    
    oModel = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oLoss = new CrossEntropyLoss
    
    # Simple setup
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    oTarget.setVal(1, 10, 1.0)
    
    # Forward + Backward
    oLogits = oModel.forward(oInput)
    nErr = oLoss.calculate(oLogits, oTarget)
    oGrad = oLoss.backwardTensor()
    oModel.backward(oGrad)
    
    ? "Loss: " + nErr
    ? nl + "Step-by-step gradient collection:"
    ? "─────────────────────────────────────────────"
    
    # 1. Token Embedding
    ? nl + "1️⃣ Token Embedding:"
    ? "   Has grads attribute: " + hasAttribute(oModel.oTokenEmbed, "grads")
    
    if hasAttribute(oModel.oTokenEmbed, "grads")
        ? "   grads type: " + type(oModel.oTokenEmbed.grads)
        ? "   Has pData: " + hasAttribute(oModel.oTokenEmbed.grads, "pData")
        
        if hasAttribute(oModel.oTokenEmbed.grads, "pData")
            pData = oModel.oTokenEmbed.grads.pData
            ? "   pData value: " see pData
            ? "   pData type: " + type(pData)
            
            # Test if it's a valid pointer
            if isNull(pData) or pData = 0
                ? "   ❌ pData is NULL or 0!"
            else
                ? "   ✅ pData looks valid"
            ok
        ok
    ok
    
    # 2. Position Embedding
    ? nl + "2️⃣ Position Embedding:"
    ? "   Has grads attribute: " + hasAttribute(oModel.oPosEmbed, "grads")
    
    if hasAttribute(oModel.oPosEmbed, "grads") and 
       hasAttribute(oModel.oPosEmbed.grads, "pData")
        ? "   ✅ Has pData"
    ok
    
    # 3. First Block
    ? nl + "3️⃣ First Block Gradients:"
    oBlock = oModel.aBlocks[1]
    
    ? "   Calling oBlock.getGradients()..."
    aBlockGrads = oBlock.getGradients()
    
    ? "   Returned type: " + type(aBlockGrads)
    ? "   Length: " + len(aBlockGrads)
    
    if len(aBlockGrads) > 0
        ? "   First 3 items:"
        for i = 1 to min(3, len(aBlockGrads))
            item = aBlockGrads[i]
            ? "     [" + i + "] type: " + type(item) + 
              " | value: " see item
        next
    ok
    
    # 4. Collect all
    ? nl + "4️⃣ Collecting all gradients:"
    aAllGrads = oModel.getAllGradients()
    
    ? "   Total collected: " + len(aAllGrads)
    
    if len(aAllGrads) > 0
        ? "   First 5 items:"
        for i = 1 to min(5, len(aAllGrads))
            item = aAllGrads[i]
            ? "     [" + i + "] " + type(item) + " : " see item
        next
    ok
    
    # 5. Test clipping
    ? nl + "5️⃣ Testing clipGlobalNorm:"
    
    if len(aAllGrads) = 0
        ? "   ❌ Can't test - no gradients!"
    else
        ? "   Calling with " + len(aAllGrads) + " items..."
        nNorm = clipGlobalNorm(aAllGrads, 1.0)
        ? "   Returned norm: " + nNorm
        
        if nNorm = 0
            ? "   ❌ Norm is 0!"
        else
            ? "   ✅ Norm is non-zero"
        ok
    ok
    
    ? "═══════════════════════════════════════════════" + nl