# File: src/data/StandardDataset.ring

class StandardDataset from UniversalDataset

    oTok        
    nSeqLen     
    nVocab      
    
    # Default IDs
    nPadID = 1
    nUnkID = 2

    func init cFile, oTokenizer, nSeq
        super.init(cFile)
        oTok    = oTokenizer
        nSeqLen = nSeq
        nVocab  = oTok.nVocabSize + 1 

    func setSequenceLength nLen
        nSeqLen = nLen

    # --- The Universal Logic (GPT Style) ---
    # It assumes the Cleaner has already formatted the string as:
    # "Prefix + Input + Separator + Output + Suffix"
    # So we just treat it as ONE long sequence.
    
    func rowToTensor rowString
        # Task: Next Token Prediction
        
        aIds = oTok.encode(rowString)
        
        # Need at least 2 tokens (Input + Target)
        if len(aIds) < 2 return NULL ok 
        
        # --- RETURN LISTS (NOT TENSORS) ---
        # Pre-allocate lists for speed
        aInputList  = list(nSeqLen)
        aTargetList = list(nSeqLen)
        
        nLen = len(aIds)
        
        for k = 1 to nSeqLen
            
            # Logic: Input at k, Target at k+1
            if k < nLen
                nInID = aIds[k]
                nTgID = aIds[k+1]
            else
                nInID = nPadID
                nTgID = nPadID
            ok
            
            # Handle Unknowns
            if nInID = 0 nInID = nUnkID ok
            if nTgID = 0 nTgID = nUnkID ok
            
            # Store in Lists
            aInputList[k]  = nInID
            aTargetList[k] = nTgID
        next
        
        # Return Raw Lists to DataLoader
        # DataLoader will convert them to Tensors efficiently
        return [aInputList, aTargetList]