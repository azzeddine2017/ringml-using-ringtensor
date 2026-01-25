



func testBiDirectionalDataset
    ? "═══════════════════════════════════════════════"
    ? "BI-DIRECTIONAL DATASET TEST"
    ? "═══════════════════════════════════════════════"
    
    # Setup
    oTok = new Tokenizer()
    oTok.buildVocab(["hello world", "مرحبا العالم"])
    
    nVocab = oTok.nVocabSize + 1
    nSeqLen = 16
    
    ? "Special tokens:"
    ? "  PAD=1, UNK=2, SEP=3, START=4, END=5"
    ? "  TO_AR=6, TO_EN=7"
    ? "  Vocab size: " + nVocab
    ? ""
    
    # Create fake dataset file
    cTestFile = "test_bidir.txt"
    write(cTestFile, "hello*****مرحبا" + nl + "world*****العالم")
    
    # Load dataset
    oDataset = new BiDirectionalDataset(cTestFile, oTok, nSeqLen)
    oDataset.loadData()

    oTrainDataset = oDataset.getTrainDataset()

    ? "Dataset loaded: " + oTrainDataset.length() + " samples"
    ? ""
	
    
    # Check first sample
    aSample = oTrainDataset.getData(1)
    
    if isNull(aSample)
        ? "❌ ERROR: Sample is NULL!"
        return
    ok
    
    aInput = aSample[1]
    aTarget = aSample[2]
    
    ? "Sample 1 (En → Ar):"
    ? "  Input length: " + len(aInput)
    ? "  Target length: " + len(aTarget)
    ? ""
    
    # Check for NULL values
    nNullInput = 0
    nNullTarget = 0
    
    for i = 1 to len(aInput)
        if isNull(aInput[i])
            nNullInput++
        ok
    next
    
    for i = 1 to len(aTarget)
        if isNull(aTarget[i])
            nNullTarget++
        ok
    next
    
    if nNullInput > 0
        ? "❌ ERROR: Input has " + nNullInput + " NULL values!"
    else
        ? "✅ Input: No NULL values"
    ok
    
    if nNullTarget > 0
        ? "❌ ERROR: Target has " + nNullTarget + " NULL values!"
    else
        ? "✅ Target: No NULL values"
    ok
    ? ""
    
    # Print first 10 positions
    ? "First 10 positions:"
    ? "  Input:  " 
    see "    ["
    for i = 1 to 10
        if i > len(aInput) exit ok
        see aInput[i]
        if i < 10 see ", " ok
    next
    see "]" + nl
    
    ? "  Target: "
    see "    ["
    for i = 1 to 10
        if i > len(aTarget) exit ok
        see aTarget[i]
        if i < 10 see ", " ok
    next
    see "]" + nl
    ? ""
    
    # Count valid targets
    nValidTargets = 0
    nZeroTargets = 0
    
    for i = 1 to len(aTarget)
        if aTarget[i] = 0
            nZeroTargets++
        else
            nValidTargets++
        ok
    next
    
    ? "Target statistics:"
    ? "  Valid targets: " + nValidTargets
    ? "  Zero targets (masked): " + nZeroTargets
    ? ""
    
    if nValidTargets > 0 and nNullTarget = 0 and nNullInput = 0
        ? "✅✅✅ DATASET IS CORRECT ✅✅✅"
    else
        ? "❌ DATASET HAS ISSUES"
    ok
    
    ? "═══════════════════════════════════════════════" + nl
    
    # Cleanup
    if fexists(cTestFile) remove(cTestFile) ok