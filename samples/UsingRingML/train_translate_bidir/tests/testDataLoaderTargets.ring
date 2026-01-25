func testDataLoaderTargets
    ? "═══════════════════════════════════════════════"
    ? "DATALOADER TARGET TEST"
    ? "═══════════════════════════════════════════════"
    
    # Setup minimal dataset
    cDataPath = "../data/en-ar-small.txt"
    
    oTok = new Tokenizer()

    cContent = read(cDataPath)
    aLines = str2list(cContent)
    
    aAllText = []
    nLimit = len(aLines)
    if nLimit > 5000 nLimit = 5000 ok
    
    for i = 1 to nLimit
        aParts = split(aLines[i], "*****")
        if len(aParts) >= 2 
            aAllText + aParts[1] + aParts[2]
        ok
    next
    
    oTok.buildVocab(aAllText)
    
    info("    Vocab Size: ")
    see oTok.nVocabSize + nl

    ? "Vocab size: " + oTok.nVocabSize
    ? "PAD ID: 1"
    ? "UNK ID: 2"
    ? "SEP ID: 3"
    ? ""

    nSeqLen = 8
    nBatchSize = 2

    # Create dataset
    oDataset = new BiDirectionalDataset(cDataPath, oTok, nSeqLen)
    oDataset.setHeader(false)
    oDataset.setShuffle(false)
    oDataset.setSplit(false)
    oDataset.loadData()
    
    oLoader = new batchDataLoader(oDataset.getTrainDataset(), nBatchSize)
    
    # Get batch
    aBatch = oLoader.getBatch(1)
    oInput = aBatch[1]
    oTarget = aBatch[2]
    
    ? "Batch info:"
    ? "  Input shape: (" + oInput.nRows + " × " + oInput.nCols + ")"
    ? "  Target shape: (" + oTarget.nRows + " × " + oTarget.nCols + ")"
    ? ""
    
    ? "Checking targets for PAD tokens:"
    nPadCount = 0
    nValidCount = 0
    nZeroCount = 0
    
    for r = 1 to oTarget.nRows
        nSum = 0
        nTargetClass = 0
        
        for c = 1 to oTarget.nCols
            val = oTarget.getVal(r, c)
            nSum += val
            if val > 0.5
                nTargetClass = c
            ok
        next
        
        if nSum < 0.01
            # All zeros (padding position)
            nZeroCount++
        elseif nTargetClass = 1
            # Target is PAD token
            nPadCount++
            if nPadCount <= 3
                ? "  ⚠️ Row " + r + ": Target=PAD (class 1)"
            ok
        else
            nValidCount++
        ok
    next
    
    ? ""
    ? "Summary:"
    ? "  Valid targets: " + nValidCount
    ? "  PAD targets: " + nPadCount
    ? "  Zero targets (masked): " + nZeroCount
    ? ""
    
    if nPadCount > 0
        ? "❌ WARNING: Dataset contains PAD tokens as targets!"
        ? "   This will confuse training."
    else
        ? "✅ No PAD tokens in targets"
    ok
    
    if nZeroCount > 0
        ? "✅ Padding positions are masked (all zeros)"
    ok
    
    ? "═══════════════════════════════════════════════" + nl