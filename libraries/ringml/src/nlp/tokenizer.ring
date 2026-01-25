# Project: Jabr
# File: src/utils/Tokenizer.ring
# Author: Azzeddine Remmal
# Description: A character-level tokenizer for preparing text data 
#              High-Speed Tokenizer using AlQalam HashMap



class Tokenizer

    # --- Configuration ---
    aSpecialTokens
    nVocabSize
    
    # --- The Engine (C++ HashMap) ---
    oIndexMap      # QalamIndex (String -> ID)
    
    func init
        oIndexMap      = new QalamIndex()
        aSpecialTokens = []
        nVocabSize     = 0
        return self
    
    func addToken cToken
        # Add to special tokens list
        aSpecialTokens + cToken
        # Add to Map
        nVocabSize++
        oIndexMap.define(cToken, nVocabSize)

    func buildVocab aTextList
        see "[Tokenizer] Building Vocabulary (Accelerated)..." + nl
        
        # Process Text
        # Assuming Character Level for now (or Word Level if split)
        # Using a Ring List loop is fine here because 'setKey' is fast
        
        for cText in aTextList
            nLen = len(cText)
            for i = 1 to nLen
                cChar = cText[i]
                
                # Check if exists (Fast O(1) Check)
                nId = oIndexMap.recall(cChar)
                
                if nId = 0 # Not found (0.0 from C++)
                    nVocabSize++
                    oIndexMap.define(cChar, nVocabSize)
                ok
            next
        next
        
        see "    Final Vocab Size: " + nVocabSize + nl

    func encode cText
    aIds = []
    nPos = 1
    nLen = len(cText)

    while nPos <= nLen
        bFoundTag = false
        
        # 1. Try matching the special tags first
        for cTag in aSpecialTokens
            nTagLen = len(cTag)
            if substr(cText, nPos, nTagLen) = cTag
                aIds + oIndexMap.recall(cTag)
                nPos += nTagLen
                bFoundTag = true
                exit # Exit tag matching loop
            ok
        next

        if bFoundTag loop ok

        nCharLen = 1
        cChar = substr(cText, nPos, nCharLen)
        nId = oIndexMap.recall(cChar)
        
        if nId = 0 
            aIds + 2 # <UNK>
        else 
            aIds + nId 
        ok
        
        nPos += nCharLen
    end
    return aIds

    func decode aIds
        cStr = ""
        for id in aIds
            # Check bounds
            if id > 0 and id <= oIndexMap.size()
                cToken = oIndexMap.recallKey(id)
                
                # Skip Special Tokens logic (Optional)
                //if cToken = "<PAD>" loop ok
                //if cToken = "<UNK>" cToken = " " ok
                cStr += cToken
            ok
        next
        return cStr

    func getTokenId cToken
        nId = oIndexMap.recall(cToken)
        if nId = 0 return 2 ok # Return UNK
        return nId

    func getTokenFromId nId
        if nId > 0 and nId <= oIndexMap.size()
            return oIndexMap.recallKey(nId)
        ok
        return "<UNK>"
    /*
        Function: saveVocab
        Description: Persists the vocabulary list to disk.
                     We save the Reverse List (ID -> String) because it contains all info needed.
    */
    func saveVocab cFilePath
        # Serialization: Convert list to Ring code string
        oIndexMap.saveBinary(cFilePath)

    /*
        Function: loadVocab
        Description: Loads vocabulary from disk and rebuilds the High-Speed C++ Index.
    */
    func loadVocab cFilePath
        if !fexists(cFilePath) raise("Tokenizer: File not found -> " + cFilePath) ok
        
        # Load the List (Ring Level)
        oIndexMap.loadBinary(cFilePath)
        
        # Rebuild the C++ Hash Map (The Engine)
        # We must re-populate AlQalam to get O(1) speed back
        
        nVocabSize = oIndexMap.size()
        
        # Re-indexing loop
        for i = 1 to nVocabSize
            cToken = oIndexMap.recallKey(i)
            # ID is simply the index i
            oIndexMap.define(cToken, i)
        next


