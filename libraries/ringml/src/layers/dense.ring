# File: src/layers/dense.ring
# Description: Fully Connected (Dense) Layer with Forward & Backward
# Author: Azzeddine Remmal

class Dense from Layer

    # Properties
    oWeights        
    oBias           
    
    oGradWeights    
    oGradBias       

    oInputCache
    nInputSize
    nNeurons

    bTrainable = true

    cName = "Dense"

    func init nIn, nOut
        nInputSize = nIn
        nNeurons   = nOut
        
        # 1. Weights
        oWeights = new Tensor(nInputSize, nNeurons)
        oWeights.random() 
        oWeights.subScalar(0.5) 
        
        # He Uniform
        nLimit = sqrt(6.0 / nIn)
        nFactor = 2.0 * nLimit
        oWeights.scalarMul(nFactor) 
        
        # 2. Bias
        oBias = new Tensor(1, nNeurons)
        oBias.zeros()
        
        # 3. Gradients
        oGradWeights = new Tensor(nInputSize, nNeurons)
        oGradWeights.zeros()
        
        oGradBias    = new Tensor(1, nNeurons)
        oGradBias.zeros()

    func forward oInput
        oInputCache = oInput
        
        # MatMul
        oOutput = oInput.matmul(oWeights)
        
        # Add Bias (Broadcast Row)
        return oOutput.addRowVec(oBias)
        
        return oOutput

    func backward oGradOutput
        # 1. Weights Gradient: Input^T * Grad
        oInputT = oInputCache.transpose()
        oGw = oInputT.matmul(oGradOutput)
        oGradWeights.add(oGw)
        
        # 2. Bias Gradient: Sum(Grad, Axis=0)
        oGb = oGradOutput.sum(0)
        oGradBias.add(oGb)
        
        # 3. Input Gradient: Grad * Weights^T
        oWeightsT = oWeights.transpose()
        oGradInput = oGradOutput.matmul(oWeightsT)
        
        return oGradInput

    func freeze
        bTrainable = false

    func unfreeze
        bTrainable = true

    func updateWeights oOptimizer
        if !bTrainable return ok 
        oOptimizer.updateTensor(oWeights, oGradWeights)
        oOptimizer.updateTensor(oBias, oGradBias)
        
        oGradWeights.fill(0)
        oGradBias.fill(0)

    func getParams
        return [ [oWeights, oGradWeights], [oBias, oGradBias] ]