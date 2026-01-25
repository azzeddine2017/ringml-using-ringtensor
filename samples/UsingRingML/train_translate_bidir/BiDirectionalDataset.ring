# Project: RingML
# File: BiDirectionalDataset.ring
# Description: Bi-Directional Dataset Class
# Author: Azzeddine Remmal

# ---------------------------------------------------------
#  Bi-Directional Dataset Class for TransformerDataLoader Class
# ---------------------------------------------------------

class BiDirectionalDataset from UniversalDataset

    oTok        
    nSeqLen     
    nVocab      
    
    nPadID = 1
    nUnkID = 2
    nSepID = 3
    nStartID = 4
    nEndID = 5
    nToArID = 6
    nToEnID = 7

    func init cFile, oTokenizer, nSeq
        cFilePath = cFile
        oTok      = oTokenizer
        nSeqLen   = nSeq
        nVocab    = oTok.nVocabSize + 1
        
    func setSequenceLength nNewLen
        nSeqLen = nNewLen
        return self

    func loadData
        info("Loading and Augmenting Data..." + nl)
        cContent = read(cFilePath)
        aLines   = str2list(cContent)
        aRawData = []
        
        for cLine in aLines
            aParts = split(cLine, "*****")
            if len(aParts) < 2 loop ok
            
            cEn = trim(aParts[1])
            cAr = trim(aParts[2])
            
            # Task 1: En → Ar
            aRawData + [nToArID, cEn, cAr]
            
            # Task 2: Ar → En
            aRawData + [nToEnID, cAr, cEn]
        next
        
        processSplit() 

    func rowToTensor aRow
        nTaskID = aRow[1]
        cSrc    = aRow[2]
        cTgt    = aRow[3]
        
        # Encode
        aSrcIds = oTok.encode(cSrc)
        aTgtIds = oTok.encode(cTgt)
        
        # Build sequence: [TASK, START, SRC..., SEP, TGT..., END]
        aFullSeq = []
        aFullSeq + nTaskID   
        aFullSeq + nStartID  
        for x in aSrcIds aFullSeq + x next 
        aFullSeq + nSepID    
        for x in aTgtIds aFullSeq + x next 
        aFullSeq + nEndID    
        
        # Initialize with 0 instead of NULL
        aInputList  = []
        aTargetList = []
        
        # Fill to nSeqLen
        for i = 1 to nSeqLen
            aInputList + 0
            aTargetList + 0
        next
        
        nLen = len(aFullSeq)
        
        # Fill actual values
        # Track which positions should be masked
        for k = 1 to nSeqLen
            if k < nLen
                nInID = aFullSeq[k]
                nTgID = aFullSeq[k+1]
            else
                nInID = nPadID
                nTgID = nPadID
            ok
            
            # Input
            if nInID = 0 nInID = nUnkID ok
            
            aInputList[k] = nInID
            
            # Target - mask special tokens في الـ INPUT
            if nTgID = 0 or nTgID = nPadID
                # Mask: padding or predicting PAD
                aTargetList[k] = 0
            elseif nInID = nTaskID or nInID = nStartID
                # Mask: don't predict after task/start tokens
                aTargetList[k] = 0
            elseif nTgID > 0 and nTgID <= nVocab
                # Valid target
                aTargetList[k] = nTgID
            else
                aTargetList[k] = 0
            ok
        next
        
        return [aInputList, aTargetList]



/*class BiDirectionalDataset from UniversalDataset

    oTok        
    nSeqLen     
    nVocab      
    
    nPadID = 1
    nUnkID = 2
    nSepID = 3
    nStartID = 4
    nEndID = 5
    nToArID = 6
    nToEnID = 7

    func init cFile, oTokenizer, nSeq
        cFilePath = cFile
        oTok      = oTokenizer
        nSeqLen   = nSeq
        nVocab    = oTok.nVocabSize + 1
        
    func setSequenceLength nNewLen
        nSeqLen = nNewLen
        return self

    func loadData
        info("Loading and Augmenting Data..." + nl)
        cContent = read(cFilePath)
        aLines   = str2list(cContent)
        aRawData = []
        for cLine in aLines
            aParts = split(cLine, "*****")
            if len(aParts) < 2 loop ok
            cEn = trim(aParts[1])
            cAr = trim(aParts[2])
            aRawData + [nToArID, cEn, cAr]
            aRawData + [nToEnID, cAr, cEn]
        next
        processSplit() 

    func rowToTensor aRow
        nTaskID = aRow[1]
        cSrc    = aRow[2]
        cTgt    = aRow[3]
        
        aSrcIds = oTok.encode(cSrc)
        aTgtIds = oTok.encode(cTgt)
        
        
        aFullSeq = []
        aFullSeq + nTaskID   
        aFullSeq + nStartID  
        for x in aSrcIds aFullSeq + x next 
        aFullSeq + nSepID    
        for x in aTgtIds aFullSeq + x next 
        aFullSeq + nEndID    
        
        aInputList  = list(nSeqLen)
        aTargetList = list(nSeqLen)
        
        nLen = len(aFullSeq)
        
        for k = 1 to nSeqLen
            if k < nLen
                nInID = aFullSeq[k]     
                nTgID = aFullSeq[k+1]   
            else
                nInID = nPadID
                nTgID = nPadID
            ok
            
            if nInID = 0 nInID = nUnkID ok
            if nTgID = 0 nTgID = nUnkID ok
            
            aInputList[k]  = nInID
            
            if nTgID > 0 and nTgID <= nVocab and nTgID != nPadID
                aTargetList[k] = nTgID
            ok
        next
        
        return [aInputList, aTargetList]


# ---------------------------------------------------------
#  Bi-Directional Dataset Class for DataLoader Class
# ---------------------------------------------------------
class BiDirectionalDataset from UniversalDataset

    oTok        
    nSeqLen     
    nVocab      
    nPadID = 1
    nUnkID = 2
    nSepID = 3
    nToArID = 4
    nToEnID = 5

    func init cFile, oTokenizer, nSeq
        # We handle loading manually to augment data
        cFilePath = cFile
        oTok      = oTokenizer
        nSeqLen   = nSeq
        nVocab    = oTok.nVocabSize + 1
        
    func loadData
        info("Loading and Augmenting Data..." + nl)
        cContent = read(cFilePath)
        aLines   = str2list(cContent)
        
        aRawData = []
        
        # Create double samples for each line
        for cLine in aLines
            aParts = split(cLine, "*****")
            if len(aParts) < 2 loop ok
            
            cEn = trim(aParts[1])
            cAr = trim(aParts[2])
            
            # Sample 1: EN -> AR
            # Format: [DirectionID, Source, Target]
            aRawData + [nToArID, cEn, cAr]
            
            # Sample 2: AR -> EN
            aRawData + [nToEnID, cAr, cEn]
        next
        
        # Shuffle is crucial here to mix tasks
        processSplit() 

    func rowToTensor aRow
        # aRow is [TaskID, SourceString, TargetString]
        
        nTaskID = aRow[1]
        cSrc    = aRow[2]
        cTgt    = aRow[3]
        
        # 1. Encode
        aSrcIds = oTok.encode(cSrc)
        aTgtIds = oTok.encode(cTgt)
        
        # 2. Construct Sequence: [TASK] + [SRC] + [SEP] + [TGT]
        aFullSeq = []
        aFullSeq + nTaskID
        for x in aSrcIds aFullSeq + x next
        aFullSeq + nSepID
        for x in aTgtIds aFullSeq + x next
        
        # 3. Tensors
        oInput  = new Tensor(nSeqLen, 1)
        oTarget = new Tensor(nSeqLen, nVocab)
        oTarget.zeros()
        
        nLen = len(aFullSeq)
        
        for k = 1 to nSeqLen
            if k < nLen
                nInID = aFullSeq[k]
                nTgID = aFullSeq[k+1]
            else
                nInID = nPadID
                nTgID = nPadID
            ok
            
            if nInID = 0 nInID = nUnkID ok
            if nTgID = 0 nTgID = nUnkID ok
            
            oInput.setVal(k, 1, nInID)
            
            if nTgID > 0 and nTgID <= nVocab and nTgID != nPadID
                oTarget.setVal(k, nTgID, 1.0)
            ok
        next
        
        return [oInput, oTarget]
*/