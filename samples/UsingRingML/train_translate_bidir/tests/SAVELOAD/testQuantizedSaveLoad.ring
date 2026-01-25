

func testQuantizedSaveLoad
    ? "═══════════════════════════════════════════════"
    ? "QUANTIZED SAVE/LOAD TEST"
    ? "═══════════════════════════════════════════════"
    
    nVocab = 20
    nSeqLen = 8
    nDim = 32
    nLayers = 2
    
    # Train model
    ? "1️⃣ Training model..."
    oModel1 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oOptim = new Adam(0.001, 0.0)
    oLoss = new CrossEntropyLoss
    
    oInput = new Tensor(1, nSeqLen)
    for i = 1 to nSeqLen
        oInput.setVal(1, i, i + 4)
    next
    
    oTarget = new Tensor(nSeqLen, nVocab)
    oTarget.fill(0)
    oTarget.setVal(1, 10, 1.0)
    
    for iter = 1 to 50
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
    ? "   Trained loss: " + nLoss1
    ? ""
    
    # Save with quantization
    ? "2️⃣ Saving with quantization..."
    oSerializer = new ModelSerializer
    oSerializer.nMode = 1  # Quantized
    
    cFile = "test_quantized.rdata"
    oSerializer.saveModel(oModel1, cFile)
    
    nSize = len(read(cFile))
    ? "   Quantized size: " + (nSize / 1024.0) + " KB"
    ? ""
    
    # Load
    ? "3️⃣ Loading quantized model..."
    oModel2 = new AdamModel2(nVocab, nSeqLen, nDim, nLayers)
    oSerializer.loadModel(oModel2, cFile)
    
    oLogits2 = oModel2.forward(oInput)
    nLoss2 = oLoss.calculate(oLogits2, oTarget)
    
    ? "   Original loss:  " + nLoss1
    ? "   Quantized loss: " + nLoss2
    ? "   Difference:     " + fabs(nLoss1 - nLoss2)
    ? ""
    
    # Compare file sizes (full vs quantized)
    ? "4️⃣ Comparing with full precision..."
    oSerializer.nMode = 0
    cFileFull = "test_full.rdata"
    oSerializer.saveModel(oModel1, cFileFull)
    
    nSizeFull = len(read(cFileFull))
    nSizeQuant = nSize
    
    ? "   Full precision: " + (nSizeFull / 1024.0) + " KB"
    ? "   Quantized:      " + (nSizeQuant / 1024.0) + " KB"
    ? "   Compression:    " + ((1 - nSizeQuant/nSizeFull) * 100) + "%"
    ? ""
    
    if fabs(nLoss1 - nLoss2) < 0.1
        ? "✅ Quantization preserves accuracy"
    else
        ? "⚠️ Significant accuracy loss from quantization"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    # Cleanup
    if fexists(cFile) remove(cFile) ok
    if fexists(cFileFull) remove(cFileFull) ok
