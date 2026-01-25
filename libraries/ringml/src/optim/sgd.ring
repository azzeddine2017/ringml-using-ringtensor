# File: src/optim/sgd.ring
# Description: Stochastic Gradient Descent Optimizer
# Author: Azzeddine Remmal

class SGD
    lr = 0.01
    nLearningRate = 0.01 # Legacy alias

    func init nLR
        lr = nLR
        nLearningRate = nLR
    ok

    func update oLayer
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok
        
        if hasAttribute(oLayer, "oWeights")
            tensor_update_sgd(
                oLayer.oWeights.pData, 
                oLayer.oGradWeights.pData, 
                lr
            )
        ok

        if hasAttribute(oLayer, "oBias")
            tensor_update_sgd(
                oLayer.oBias.pData, 
                oLayer.oGradBias.pData, 
                lr
            )
        ok