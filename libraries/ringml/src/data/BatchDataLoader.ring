/*
    file: src/data/BatchDataLoader.ring
    Description: High-Performance Batch Constructor for Transformers
    Author: Azzeddine Remmal
*/


class BatchDataLoader

    oDataset
    nBatchSize
    nBatches
    nTotal
    
    # --- The Pen Buffers (Reusable Memory) ---
    oPenInput    = NULL
    oPenTarget   = NULL
    
    func init oData, nBatch
        oDataset   = oData
        nBatchSize = nBatch
        nTotal     = oDataset.length()
        
        if nBatchSize < 1 nBatchSize = 1 ok
        nBatches = floor(nTotal / nBatchSize)
        if nBatches = 0 and nTotal > 0 nBatches = 1 ok
        
        # 1. Initialize The Pen (Once)
        # We assume max size roughly (Batch * SeqLen)
        # 32 batch * 32 seq = 1024 elements. Safe to reserve 5000.
        oPenInput  = new QalamVector(5000)
        oPenTarget = new QalamVector(5000)

    func getBatch nBatchIndex
        # ... (Calculating the start and end dates and verifying the data)...
        nStart = (nBatchIndex - 1) * nBatchSize + 1
        nEnd   = nStart + nBatchSize - 1
        
        if nEnd > nTotal nEnd = nTotal ok
        
        nCurrentBatchSize = nEnd - nStart + 1
        if nCurrentBatchSize <= 0 return [] ok
        
        # Metadata check
        firstItem = oDataset.getData(nStart)
        if isNull(firstItem) return [] ok
        
        nSeqPerItem = len(firstItem[1])
        
        nVocab = 0
        
        if isAttribute(oDataset.oParent, "nVocab")
            nVocab = oDataset.oParent.nVocab
        ok
        
        if nVocab = 0 
            raise("DataLoader Error: Property 'nVocab' not found in Dataset. Please define it.") 
        ok
        # --- 1. Prepare The Pen (C++ Vector) ---
        oPenInput.wipe()
        oPenTarget.wipe()
        
        # --- 2. High-Speed Aggregation ---
        for i = nStart to nEnd
            item = oDataset.getData(i) 
            if !isNull(item)
               # Merge lists into C++ memory directly
               # (use flowBatch if added, or fast flow loop)
               for val in item[1] oPenInput.flow(val) next
               for val in item[2] oPenTarget.flow(val) next
            ok
        next
        
        # --- 3. Zero-Copy / Fast Kernels ---
        
        # A. Input Tensor (Zero-Copy View)
        # Convert C++ memory to Tensor C directly
        oBatchInput = oPenInput.toTensor(nCurrentBatchSize, nSeqPerItem)
        
        # B. Target Tensor (Fast Scatter)
        nTotalRows = nCurrentBatchSize * nSeqPerItem
        oBatchTarget = new Tensor(nTotalRows, nVocab)
        oBatchTarget.zeros()
        
        # The magic here: Pass the C++ memory address directly to the kernel
        # No need to extract data to Ring and return it
        pIndicesAddr = oPenTarget.getRawPointer()
        nIndicesCount = oPenTarget.size()
        
        oBatchTarget.setOneHotFromPtr(pIndicesAddr, nIndicesCount, 1.0)
        
        return [oBatchInput, oBatchTarget]
