# File: src/data/dataset.ring
# Description: Dataset and DataLoader for batch processing
# Author: Azzeddine Remmal

class Dataset
    func length 
        raise("Method length() not implemented")
    func getData itemIndex
        raise("Method getData() not implemented")

class TensorDataset from Dataset
    oInputs
    oTargets
    nSamples

    func init oInTensor, oTargetTensor
        oInputs  = oInTensor
        oTargets = oTargetTensor
        nSamples = oInputs.nRows
        
    func length
        return nSamples
        
    func getData nIdx
        # Extract Single Row Tensor
        oInRow = new Tensor(1, oInputs.nCols)
        for c=1 to oInputs.nCols 
            oInRow.setVal(1, c, oInputs.getVal(nIdx, c)) 
        next
        
        oTargetRow = new Tensor(1, oTargets.nCols)
        for c=1 to oTargets.nCols 
            oTargetRow.setVal(1, c, oTargets.getVal(nIdx, c)) 
        next
        
        return [oInRow, oTargetRow]

class DataLoader
    oDataset
    nBatchSize
    nBatches
    
    func init oData, nBatch
        oDataset   = oData
        nBatchSize = nBatch
        nTotal     = oDataset.length()
        nBatches   = ceil(nTotal / nBatchSize)

    func getBatch nBatchIndex
        nStart = (nBatchIndex - 1) * nBatchSize + 1
        nEnd   = nStart + nBatchSize - 1
        if nEnd > oDataset.length() nEnd = oDataset.length() ok
        
        nCurrentBatchSize = nEnd - nStart + 1
        if nCurrentBatchSize <= 0 return [] ok
        
        # Peek dims
        firstItem = oDataset.getData(1)
        nFeats  = firstItem[1].nCols
        nLabels = firstItem[2].nCols
        
        # Create Batch Tensors
        oBatchInputs  = new Tensor(nCurrentBatchSize, nFeats)
        oBatchTargets = new Tensor(nCurrentBatchSize, nLabels)
        
        # Fill Batch
        rowCounter = 1
        for i = nStart to nEnd
            item = oDataset.getData(i) 
            
            # Copy Input Row (Element by Element)
            for c = 1 to nFeats
                val = item[1].getVal(1, c)
                oBatchInputs.setVal(rowCounter, c, val)
            next
            
            # Copy Target Row
            for c = 1 to nLabels
                val = item[2].getVal(1, c)
                oBatchTargets.setVal(rowCounter, c, val)
            next
            
            rowCounter++
        next
        
        return [oBatchInputs, oBatchTargets]