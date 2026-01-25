

func testDataLoaderDetailed
    ? "═══════════════════════════════════════════════"
    ? "DETAILED DATALOADER TEST"
    ? "═══════════════════════════════════════════════"
    
    # Setup tokenizer
    oTok = new Tokenizer()
    oTok.addToken("<PAD>")   # 1
    oTok.addToken("<UNK>")   # 2
    oTok.addToken("<SEP>")   # 3
    oTok.addToken("<START>") # 4
    oTok.addToken("<END>")   # 5
    oTok.addToken("TO_AR")   # 6
    oTok.addToken("TO_EN")   # 7
    
    # Build vocab from sample
    oTok.buildVocab(["hello world", "مرحبا العالم"])
    
    nVocab = oTok.nVocabSize + 1
    nSeqLen = 16
    
    ? "Vocab size: " + nVocab
    ? "Seq length: " + nSeqLen
    ? ""
    
    # Create test file
    cTestFile = "test_data.txt"
    write(cTestFile, "hello world*****مرحبا العالم" + nl +
                     "good morning*****صباح الخير")
    
    # Load dataset
    oDataset = new BiDirectionalDataset(cTestFile, oTok, nSeqLen)
    oDataset.setHeader(false)
    oDataset.setShuffle(false)
    oDataset.setSplit(false)
    oDataset.loadData()
    
    ? "Dataset size: " + oDataset.getTrainDataset().length()
    ? ""
    
    # Create loader
    oLoader = new batchDataLoader(oDataset.getTrainDataset(), 2)
    
    ? "Testing batch 1..."
    aBatch = oLoader.getBatch(1)
    
    oInput = aBatch[1]
    oTarget = aBatch[2]
    
    ? "Input shape: (" + oInput.nRows + " × " + oInput.nCols + ")"
    ? "Target shape: (" + oTarget.nRows + " × " + oTarget.nCols + ")"
    ? ""
    
    # ═══════════════════════════════════════════
    # Check target distribution
    # ═══════════════════════════════════════════
    ? "Analyzing targets..."
    
    nTotalPositions = oTarget.nRows
    nValidTargets = 0
    nPadTargets = 0
    nZeroTargets = 0
    nOtherTargets = 0
    
    aTargetClasses = []
    
    for r = 1 to nTotalPositions
        nSum = 0
        nClass = 0
        
        for c = 1 to oTarget.nCols
            val = oTarget.getVal(r, c)
            nSum += val
            if val > 0.5
                nClass = c
            ok
        next
        
        if nSum < 0.01
            # All zeros
            nZeroTargets++
        elseif nClass = 1
            # PAD token
            nPadTargets++
        elseif nClass >= 2 and nClass <= nVocab
            # Valid target
            nValidTargets++
            aTargetClasses + nClass
        else
            nOtherTargets++
        ok
    next
    
    ? "Target distribution:"
    ? "  Valid targets: " + nValidTargets
    ? "  PAD targets (class 1): " + nPadTargets
    ? "  Zero targets (masked): " + nZeroTargets
    ? "  Other: " + nOtherTargets
    ? ""
    
    # Show unique target classes
    ? "Unique target classes (first 10):"
    aSeen = []
    nCount = 0
    for c in aTargetClasses
        if find(aSeen, c) = 0
            aSeen + c
            nCount++
            ? "  Class " + c + ": " + oTok.getTokenFromId(c)
            if nCount >= 10 exit ok
        ok
    next
    ? ""
    
    # ═══════════════════════════════════════════
    # Test actual training
    # ═══════════════════════════════════════════
    ? "Testing mini-training..."
    
    oModel = new AdamModel2(nVocab, nSeqLen, 32, 2)
    oOptim = new Adam(0.001, 0.0)
    oLoss = new CrossEntropyLoss
    
    nLoss1 = 0
    for iter = 1 to 10
        oLogits = oModel.forward(oInput)
        nLoss1 = oLoss.calculate(oLogits, oTarget)
        
        if iter = 1
            ? "  Iteration 1 loss: " + nLoss1
        ok
        
        oGrad = oLoss.backwardTensor()
        oModel.backward(oGrad)
        
        aPtrs = oModel.getAllGradients()
        nNorm = tensor_clip_global_norm(aPtrs, 5.0)
        
        oModel.updateWeights(oOptim)
    next
    
    ? "  Iteration 10 loss: " + nLoss1
    ? ""
    
    # ═══════════════════════════════════════════
    # Verdict
    # ═══════════════════════════════════════════
    
    nRandomLoss = -log(1.0 / nVocab)
    
    ? "═══════════════════════════════════════════"
    ? "RESULTS:"
    ? "═══════════════════════════════════════════"
    ? "Random baseline: " + nRandomLoss
    ? ""
    
    bPassed = true
    
    if nPadTargets > 0
        ? "❌ Found PAD tokens in targets (" + nPadTargets + ")"
        bPassed = false
    else
        ? "✅ No PAD tokens in targets"
    ok
    
    if nValidTargets = 0
        ? "❌ No valid targets found!"
        bPassed = false
    else
        ? "✅ Found " + nValidTargets + " valid targets"
    ok
    
    if nLoss1 >= nRandomLoss
        ? "❌ Loss did not improve (still random)"
        bPassed = false
    else
        improvement = ((nRandomLoss - nLoss1) / nRandomLoss) * 100
        ? "✅ Loss improved by " + improvement + "%"
    ok
    
    ? ""
    
    if bPassed
        ? "✅✅✅ DATALOADER TEST PASSED ✅✅✅"
    else
        ? "❌ DATALOADER HAS ISSUES"
    ok
    
    ? "═══════════════════════════════════════════" + nl
    
    # Cleanup
    if fexists(cTestFile) remove(cTestFile) ok