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