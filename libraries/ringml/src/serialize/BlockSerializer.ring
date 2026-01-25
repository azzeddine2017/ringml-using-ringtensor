# File: src/serialize/BlockSerializer.ring
# Description: Block Transformer serializer for RingML
# Author: Azzeddine Remmal


class BlockSerializer

    # Save mode: 0 = Full resolution, 1 = Compressed (Quantized)
    nMode = 1 

    func saveBlock oBlock, cFileName
        oStyl.green(:NONE, "Saving Model to Single File: " + cFileName + "...")
        
        # We will collect all binary data here
        cTotalBinary = ""
        
        # List of tensors to save (in order)
        aTensors = getAllTensors(oBlock)
        
        # 1. Save header (Number of tensors + mode)
        # We use Int2Str (4 bytes) to encode numbers
        cTotalBinary += int2bytes(len(aTensors))
        cTotalBinary += int2bytes(nMode)
        
        # 2. Save tensors
        for oTensor in aTensors
            cTempFile = "temp_save.bin"
            
            if nMode = 1
                oTensor.saveFileQuantized(cTempFile)
            else
                oTensor.save(cTempFile)
            ok
            
            # Read binary data and merge it
            cBin = read(cTempFile)
            cTotalBinary += cBin
            
            # Delete temporary file
            remove(cTempFile) 
        next
        
        # 3. Write final file
        write(cFileName, cTotalBinary)
        oStyl.green(:NONE, "Done. Size: " + (len(cTotalBinary)/1024/1024) + " MB")

    func loadBlock oBlock, cFileName
        oStyl.green(:NONE, "Loading Model from Single File...")
        
        if !fexists(cFileName) raise("File not found") ok
        
        cAllData = read(cFileName)
        nPtr = 1 # Read pointer
        
        # 1. Read header
        nTensors = bytes2int(substr(cAllData, nPtr, 4))
        nPtr += 4
        nSavedMode = bytes2int(substr(cAllData, nPtr, 4))
        nPtr += 4
        
        aTensors = getAllTensors(oBlock)
        
        if len(aTensors) != nTensors
            raise("Model Mismatch: File has " + nTensors + " tensors, Model has " + len(aTensors))
        ok
        
        # 2. Extract tensors
        for i = 1 to nTensors
            # A. Read dimensions (First 8 bytes in the block: Rows, Cols)
            # Each int = 4 bytes
            cHeader = substr(cAllData, nPtr, 8)
            nRows = bytes2int(substr(cHeader, 1, 4))
            nCols = bytes2int(substr(cHeader, 5, 4))
            
            # B. Calculate data size
            # Double = 8 bytes, Float = 4 bytes
            nElemSize = 8
            if nSavedMode = 1 nElemSize = 4 ok
            
            nDataBytes = nRows * nCols * nElemSize
            nTotalBlock = 8 + nDataBytes
            
            # C. Extract block and write to temporary file
            cChunk = substr(cAllData, nPtr, nTotalBlock)
            cTempFile = "temp_load.bin"
            write(cTempFile, cChunk)
            
            # D. Load to C
            if nSavedMode = 1
                aTensors[i].loadFileQuantized(cTempFile)
            else
                aTensors[i].loadFile(cTempFile)
            ok
            
            # Update properties in Ring manually here
            aTensors[i].nRows = nRows
            aTensors[i].nCols = nCols
            
            # Move pointer
            nPtr += nTotalBlock
            
            # Clean up
            # system("del " + cTempFile) # Optional for speed
        next
        
        oStyl.green(:NONE, "Model Loaded Successfully." + nl)


    # --- Collect all tensors in a flat list ---
    func getAllTensors oBlock
        aOutputList = []
        
        # Attention
        for item in oBlock.oAttention.getAttentionWeightsList()
            aOutputList + item
        next
        
        # Norm 1
        aOutputList + oBlock.oNorm1.gamma
        aOutputList + oBlock.oNorm1.beta
        
        # FFN
        # (Assuming FactorizedDense or Dense)
        # If Factorized: Down.W, Down.B, Up.W, Up.B
        # If Dense: W, B
        
        # FFN 1
        aOutputList + oBlock.oFFN_1.oWeights # Or sublayers
        aOutputList + oBlock.oFFN_1.oBias
        
        # FFN 2
        aOutputList + oBlock.oFFN_2.oWeights
        aOutputList + oBlock.oFFN_2.oBias
        
        # Norm 2
        aOutputList + oBlock.oNorm2.gamma
        aOutputList + oBlock.oNorm2.beta
        
        return aOutputList


