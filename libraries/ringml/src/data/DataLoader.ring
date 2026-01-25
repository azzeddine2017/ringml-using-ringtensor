/*
    file: src/data/DataLoader.ring
    Description: High-Performance Batch Constructor for Transformers
    Author: Azzeddine Remmal
*/


class DataLoader
    oDataset
    nBatchSize
    nBatches
    bDesply     = false
    
    func init oData, nBatch
        oDataset   = oData
        nBatchSize = nBatch
        nTotal     = oDataset.length()
        
        # Avoid division by zero
        if nBatchSize < 1 nBatchSize = 1 ok
        
        nBatches   = floor(nTotal / nBatchSize)
        nDrop      = nTotal % nBatchSize

        if bDesply
            oStyl.cyan(:BOLD, "    Total Samples: ")
            ? oStyl.green(:BOLD, " " + nTotal)
            oStyl.cyan(:BOLD, "    Batch Size:    ")
            oStyl.green(:BOLD, " " + nBatch)
            oStyl.cyan(:BOLD, "    Full Batches:  ")
            oStyl.green(:BOLD, " " + nBatches)
            oStyl.cyan(:BOLD," ( Dropped ") 
            oStyl.green(:BOLD, " " + nDrop)
            ? oStyl.cyan(:BOLD," samples )")
        ok
        
        # Handle case where total < batch size
        if nBatches = 0 and nTotal > 0 nBatches = 1 ok
        return self

    func getBatch nBatchIndex
        nStart = (nBatchIndex - 1) * nBatchSize + 1
        nEnd   = nStart + nBatchSize - 1
        
        # Bounds check
        if nEnd > oDataset.length() nEnd = oDataset.length() ok
        
        nCurrentBatchSize = nEnd - nStart + 1
        if nCurrentBatchSize <= 0 return [] ok
        
        # We must look at the structure of the data items to allocate memory correctly.
        
        firstItem = oDataset.getData(nStart)
        
        # Get dimensions of the single item tensors
        # Example GPT Target: Rows=32, Cols=91
        nInRowsPerItem  = firstItem[1].nRows
        nInCols         = firstItem[1].nCols
        
        nTgRowsPerItem  = firstItem[2].nRows
        nTgCols         = firstItem[2].nCols
        
        # Calculate Total Batch Rows (Stacking)
        # For MLP: 32 items * 1 row = 32 rows total.
        # For GPT: 1 item * 32 rows = 32 rows total.
        nTotalInRows = nCurrentBatchSize * nInRowsPerItem
        nTotalTgRows = nCurrentBatchSize * nTgRowsPerItem
        
        # --- 2. ALLOCATE BATCH TENSORS ---
        oBatchInputs  = new Tensor(nTotalInRows, nInCols)
        oBatchTargets = new Tensor(nTotalTgRows, nTgCols)
        
        # --- 3. FILL DATA (STACKING) ---
        # Pointers to track where we are writing in the big batch tensor
        nInWriteRow = 1
        nTgWriteRow = 1
        
        for i = nStart to nEnd
            item = oDataset.getData(i) 
            oItemIn = item[1]
            oItemTg = item[2]
            
            # Copy Input (Handle multi-row items)
            for r = 1 to nInRowsPerItem
                for c = 1 to nInCols
                    val = oItemIn.getVal(r, c)
                    oBatchInputs.setVal(nInWriteRow, c, val)
                next
                nInWriteRow++
            next
            
            # Copy Target (Handle multi-row items - Critical for GPT)
            for r = 1 to nTgRowsPerItem
                for c = 1 to nTgCols
                    val = oItemTg.getVal(r, c)
                    oBatchTargets.setVal(nTgWriteRow, c, val)
                next
                nTgWriteRow++
            next
        next
        
        return [oBatchInputs, oBatchTargets]