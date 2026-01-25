/*
    Project: Adam
    File: src/utils/universal_cleaner.ring
    Description: High-Performance Data ETL Tool powered by AlQalam.
                 - Uses QalamInk for Zero-Overhead string accumulation.
                 - Single-pass processing (Extract + Transform + Clean + Load).
*/


# --- Test Section ---
if isMainSourceFile()
    
    oClean = new UniversalCleaner()
    
    see "[Cleaner] Universal Cleaner initialized (AlQalam Engine)." + nl
    
    # 1. Translation Task
    oClean {
        clear()
        loadFile("downloads/en-ar-small.txt")
        setSeparator("*****") 
        setTaskWrappers("<TO_AR> ", " <SEP> ", " <END>")
        adaptCsv(1, 2, "*****") 
        save("data/ready/train_translation.txt")
    }

    # 2. Chat Task
    oClean {
        clear()
        loadFile("downloads/alpaca_data.jsonl")
        setTaskWrappers("<CHAT> ", "\nBot: ", " <END>") 
        adaptAlpacaJsonl()
        save("data/ready/train_chat.txt")
    }
    
    see "Data Preparation Complete." + nl
ok
# --------------------

class UniversalCleaner

    # Input Data (Ring List - Needed for logic iteration)
    aRawLines 
    
    # Output Data (AlQalam Ink - High Speed Buffer)
    oInk 
    
    # Configuration
    cSeparator 
    cPrefix     
    cInfix      
    cSuffix     

    func init
        oInk = new QalamInk()
        aRawLines = []
        return self

    # --- Configuration ---
    func setSeparator cSep
        cSeparator = cSep
        return self

    func setTaskWrappers cPre, cIn, cSuf
        cPrefix = cPre
        cInfix  = cIn
        cSuffix = cSuf
        return self

    func clear
        aRawLines = []
        oInk.wipe()
        return self

    # --- 1. EXTRACT ---
    
    func loadFile cPath
        see "[Cleaner] Loading: " + cPath + "..." + nl
        if !fexists(cPath) raise("File not found: " + cPath) ok
        
        cContent = read(cPath)
        aRawLines = str2list(cContent)
        return self

    # --- 2. TRANSFORM & REFINE (Single Pass) ---

    /*
        Func: adaptCsv
        Desc: Processes delimiter-separated values (CSV/TSV/Custom)
              Applies cleaning and templates on the fly.
    */
    func adaptCsv nColSrc, nColTgt, cDelim
        see "[Cleaner] Adapting CSV format..." + nl
        
        nLimit = len(aRawLines)
        
        for i = 1 to nLimit
            cLine = aRawLines[i]
            if len(trim(cLine)) = 0 loop ok
            
            aRow = split(cLine, cDelim)
            
            if len(aRow) >= max(nColSrc, nColTgt)
                cSrc = trim(aRow[nColSrc])
                cTgt = trim(aRow[nColTgt])
                
                # Filter bad data
                if len(cSrc) < 2 or len(cTgt) < 2 loop ok
                
                # Apply normalization
                cSrc = normalizeText(cSrc)
                cTgt = normalizeText(cTgt)
                
                # Build Template
                # Format: PREFIX + Src + INFIX + Tgt + SUFFIX + NewLine
                cFinal = cPrefix + cSrc + cInfix + cTgt + cSuffix + nl
                
                # Write to C++ Memory directly
                oInk.inscribe(cFinal)
            ok
        next
        return self

    /*
        Func: adaptAlpacaJsonl
        Desc: Extracts Instruction/Input/Output from JSONL.
    */
    func adaptAlpacaJsonl
        see "[Cleaner] Adapting Alpaca JSONL..." + nl
        
        nLimit = len(aRawLines)
        
        for i = 1 to nLimit
            cLine = aRawLines[i]
            
            # Fast Extraction
            cInst = extractJsonVal(cLine, "instruction")
            cInp  = extractJsonVal(cLine, "input")
            cOut  = extractJsonVal(cLine, "output")
            
            if cInst = "" loop ok 
            
            # Normalize
            cInst = normalizeText(cInst)
            cInp  = normalizeText(cInp)
            cOut  = normalizeText(cOut)
            
            # Template Construction
            cFullInput = cInst
            if len(cInp) > 0 cFullInput += " Context: " + cInp ok
            
            cFinal = cPrefix + cFullInput + cInfix + cOut + cSuffix + nl
            
            # Write to C++ Memory
            oInk.inscribe(cFinal)
        next
        return self

    # --- 3. INTERNAL HELPERS ---
    
    func normalizeText cText
        # 1. Remove Nulls
        cText = substr(cText, char(0), "")
        # 2. Normalize Tabs
        cText = substr(cText, char(9), " ")
        # 3. Collapse multiple spaces (Simple pass)
        # For heavy regex, we'd need a C++ kernel, but substr loop is fine for now
        # skipping for speed in Ring, relying on Tokenizer later.
        return cText

    func extractJsonVal cJson, cKey
        # Look for "key": "value"
        # We search for "key": " to find the start
        cSearch = '"' + cKey + '": "'
        nPos = substr(cJson, cSearch)
        if nPos = 0 return "" ok
        
        nStart = nPos + len(cSearch)
        
        # Search for closing quote NOT preceded by backslash
        # Simplified: Just find next quote (fastest)
        # Advanced JSON parsing should be done via library if strictness required
        nEnd = substr(cJson, '"', nStart)
        
        if nEnd = 0 return "" ok
        
        return substr(cJson, nStart, nEnd - nStart)

    # --- 4. LOAD (Export) ---
    
    func save cOutPath
        # Reveal the ink from C++ memory
        cFinalData = oInk.reveal()
        
        if len(cFinalData) = 0
            see "[Cleaner] Warning: No data processed!" + nl
            return
        ok
        
        see "[Cleaner] Saving to " + cOutPath + "..." + nl
        write(cOutPath, cFinalData)
        see "[Cleaner] Done." + nl

    /*
        Pour the cleaned data into a vector database
        to allow models to train from it directly
    */
    func injectToVectorDB oVDB, oEmbedderModel, cTaskLabel
        see "[Cleaner] Injecting cleaned data into VectorDB for task: " + cTaskLabel + nl
        
        # 1. Reveal the ink from C++ memory
        cFullText = oInk.reveal()
        aLines = str2list(cFullText)
        
        # 2. Smart Storage
        for cLine in aLines
            if len(trim(cLine)) < 5 loop ok
            
            # Convert the sentence to a vector (Embedding) via the Manager or Main Model
            oVec = oEmbedderModel.getEmbedding(cLine)
            
            # Store the text with the "important tag" (Task Tag) in the metadata
            # So that any expert can retrieve their data later
            oVDB.addDocument(cTaskLabel + "::" + cLine, oVec)
        next
        
        see "[Cleaner] Injection Complete." + nl