



# ---------------------------------------------------------
#  Helper: Inference Function
# ---------------------------------------------------------

func generateTranslation oModel, oTok, cSrc, cTaskToken, nSeq
    # Input: [TaskToken] + [Source] + [SEP]
    aIds = []
    aIds + oTok.getTokenId(cTaskToken)
    
    aSrcIds = oTok.encode(cSrc)
    for x in aSrcIds aIds + x next
    
    aIds + 3 # SEP
    
    cGenerated = ""
    
    for stepe = 1 to 20
        oInput = new Tensor(1, nSeq)
        for k = 1 to nSeq oInput.setVal(1, k, 1) next # Fill PAD
        
        for k = 1 to len(aIds)
            if k > nSeq exit ok
            v = aIds[k] if v=0 v=2 ok
            oInput.setVal(1, k, v)
        next
        
        oLogits = oModel.forward(oInput)
        oLogits.softmax()
        
        nLastPos = len(aIds)
        if nLastPos > nSeq nLastPos = nSeq ok
        
        maxV = -1 maxId = 0
        for v = 1 to oLogits.nCols 
            val = oLogits.getVal(nLastPos, v) 
            if val > maxV maxV = val maxId = v ok
        next
        
        aIds + maxId
        cToken = oTok.getTokenFromId(maxId)
        
        if maxId = 1 exit ok
        if maxId = 4 exit ok
        
        cGenerated += cToken
    next
    
    info("   Gen (" + cTaskToken + "): " + cSrc + " -> " + cGenerated)

