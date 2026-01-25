# File: src/serialize/ModelSerializer.ring
# Description: Model serializer (Model Multi-Block) for RingML
# Author: Azzeddine Remmal


class ModelSerializer

    # Save mode: 0 = Full resolution, 1 = Compressed (Quantized)
    nMode = 1 
    
    func saveModel oModel, cFileName
        oStyl.green(:NONE, "[Serializer] Saving Model to " + cFileName + "..." + nl)
        
        cTotalBinary = ""
        
        # 1. Compiling all tensors into one list
        aAllTensors = oModel.getAllWeights() 

        # 2. Save header (Number of tensors + mode)
        # We use Int2Str (4 bytes) to encode numbers
        cTotalBinary += int2bytes(len(aAllTensors))
        cTotalBinary += int2bytes(nMode)
        
        # 3. Save tensors
        for oTensor in aAllTensors
            cTempFile = "temp_save.bin"
            
            if nMode = 1
                oTensor.saveFileQuantized(cTempFile)
            else
                oTensor.saveFile(cTempFile)
            ok
            
            # Read binary data and merge it
            cBin = read(cTempFile)
            cTotalBinary += cBin
            
            # Delete temporary file
            remove(cTempFile) 
        next
        
        # 4. Writing the final file
        write(cFileName, cTotalBinary)
        oStyl.green(:NONE, "[Serializer] Done. Size: " + (len(cTotalBinary)/1024/1024) + " MB" + nl)




    func loadModel oModel, cFileName
        oStyl.green(:NONE, "[Serializer] Loading Model..." + nl)
    
        if !fexists(cFileName) 
            raise("File not found: " + cFileName) 
        ok
        
        cAllData = read(cFileName)
        nPtr = 1
        
        # ════════════════════════════════════════════════
        # 1. Read header
        # ════════════════════════════════════════════════
        nTensors = bytes2int(substr(cAllData, nPtr, 4))
        nPtr += 4
        nSavedMode = bytes2int(substr(cAllData, nPtr, 4))
        nPtr += 4
        
        # ════════════════════════════════════════════════
        # 2. Get model tensors
        # ════════════════════════════════════════════════
        aTensors = oModel.getAllWeights()
        
        if len(aTensors) != nTensors
            raise("Model Mismatch: File has " + nTensors + 
                " tensors, Model has " + len(aTensors))
        ok
        
        # ════════════════════════════════════════════════
        # 3. Load each tensor
        # ════════════════════════════════════════════════
        cTempFile = "temp_load_" + random(999999) + ".bin"
        
        for i = 1 to nTensors
            # ─────────────────────────────────────────
            # A. Read dimensions (8 bytes)
            # ─────────────────────────────────────────
            cHeader = substr(cAllData, nPtr, 8)
            nRows = bytes2int(substr(cHeader, 1, 4))
            nCols = bytes2int(substr(cHeader, 5, 4))
            
            # ─────────────────────────────────────────
            # B. Calculate block size
            # ─────────────────────────────────────────
            nElemSize = 8  # double
            if nSavedMode = 1 
                nElemSize = 4  # float (quantized)
            ok
            
            nDataBytes = nRows * nCols * nElemSize
            nTotalBlock = 8 + nDataBytes
            
            # ─────────────────────────────────────────
            # C. Extract block
            # ─────────────────────────────────────────
            cChunk = substr(cAllData, nPtr, nTotalBlock)
            write(cTempFile, cChunk)
            
            # ─────────────────────────────────────────
            # D. Load into tensor
            # ─────────────────────────────────────────
            # ✅ CRITICAL: Use the correct method
            if nSavedMode = 1
                # Quantized mode
                aTensors[i].loadFileQuantized(cTempFile)
            else
                # Full precision
                aTensors[i].loadFile(cTempFile)
            ok
            
            # ✅ Update dimensions (if needed)
            # Note: loadFile/loadQuantized should handle this internally
            # But we set them explicitly just to be safe
            if hasAttribute(aTensors[i], "nRows")
                aTensors[i].nRows = nRows
            ok
            if hasAttribute(aTensors[i], "nCols")
                aTensors[i].nCols = nCols
            ok
            if hasAttribute(aTensors[i], "size")
                aTensors[i].size = nRows * nCols
            ok
            
            # ─────────────────────────────────────────
            # E. Move pointer
            # ─────────────────────────────────────────
            nPtr += nTotalBlock
        next
        
        # ════════════════════════════════════════════════
        # 4. Cleanup
        # ════════════════════════════════════════════════
        if fexists(cTempFile)
            remove(cTempFile)
        ok
        
        oStyl.green(:NONE, "[Serializer] Loaded Successfully." + nl)
