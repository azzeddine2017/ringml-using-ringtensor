# File: src/layers/dense.ring
# Description: Fully Connected (Dense) Layer with Forward & Backward
# Author: Azzeddine Remmal


class Dense from Layer
    oWeights        
    oBias           
    oInput          
    
    oGradWeights    
    oGradBias       

    nInputSize
    nNeurons

    func init nIn, nOut
        nInputSize = nIn
        nNeurons   = nOut
        
        # DEBUG INFO
        //see ">> Init Dense (" + nInputSize + ", " + nNeurons + ")... " 
        
        # 1. Weights Init
        oWeights = new Tensor(nInputSize, nNeurons)
        
        # Manual Random Init (-1 to 1)
        # We assume Tensor creates lists of INT 0s. We must overwrite ALL of them.
        
        countFilled = 0
        for r = 1 to nInputSize
            for c = 1 to nNeurons
                val = (random(4000) / 10000.0) - 0.2
                oWeights.aData[r][c] = val
                countFilled++
            next
        next
        
        # DEBUG CHECK
        /*if countFilled != (nInputSize * nNeurons)
           see "ERROR: Only filled " + countFilled + " items!" + nl
        else
           see "OK (Filled " + countFilled + ")" + nl
        ok*/

        # 2. Bias Init (Small Randoms)
        oBias = new Tensor(1, nNeurons)
        oBias.zeros()
        /*for c = 1 to nNeurons
             val = (random(100) / 10000.0) 
             oBias.aData[1][c] = val
        next*/
        
        # 3. Gradients Init
        oGradWeights = new Tensor(nInputSize, nNeurons)
        oGradWeights.zeros()
        oGradBias    = new Tensor(1, nNeurons)
        oGradBias.zeros()
        
    func forward oInputTensor
        oInput = oInputTensor
        oOutput = oInput.matmul(oWeights)
        
        biasList = oBias.aData[1]
        for r = 1 to oOutput.nRows
            row = oOutput.aData[r]
            for c = 1 to oOutput.nCols
                row[c] += biasList[c]
            next
        next
        
        return oOutput

    func backward oGradOutput
        oInputCopy = oInput.copy()
        oInputCopy.transpose()
        oGradWeights = oInputCopy.matmul(oGradOutput)
        
        oGradBias = oGradOutput.sum(0) 
        
        oWeightsCopy = oWeights.copy()
        oWeightsCopy.transpose()
        dInput = oGradOutput.matmul(oWeightsCopy)
        
        return dInput