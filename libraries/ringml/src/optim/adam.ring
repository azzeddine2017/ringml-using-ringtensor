# File: src/optim/adam.ring
# Description: Adam Optimizer (Manual Implementation for Stability)
# Author: Azzeddine Remmal

class Adam
    nLR = 0.001
    nBeta1 = 0.9
    nBeta2 = 0.999
    nEpsilon = 0.00000001
    
    func init nLearningRate
        nLR = nLearningRate

    func update oLayer
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok
        if !hasAttribute(oLayer, "oWeights") return ok

        # --- Initialize State ---
        if !hasAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_vw")
            oLayer.adam_mw = oLayer.oWeights.copy().zeros()
            oLayer.adam_vw = oLayer.oWeights.copy().zeros()
            
            addAttribute(oLayer, "adam_mb")
            addAttribute(oLayer, "adam_vb")
            oLayer.adam_mb = oLayer.oBias.copy().zeros()
            oLayer.adam_vb = oLayer.oBias.copy().zeros()
            
            addAttribute(oLayer, "adam_t")
            oLayer.adam_t = 0
        ok

        # Time Step
        oLayer.adam_t++
        nT = oLayer.adam_t

        # --- Fused Kernel Update (Passing Pointers) ---
        
        # Update Weights
        tensor_update_adam(
            oLayer.oWeights.pData, 
            oLayer.oGradWeights.pData, 
            oLayer.adam_mw.pData, 
            oLayer.adam_vw.pData, 
            nLR, nBeta1, nBeta2, nEpsilon, nT
        )

        # Update Bias
        tensor_update_adam(
            oLayer.oBias.pData, 
            oLayer.oGradBias.pData, 
            oLayer.adam_mb.pData, 
            oLayer.adam_vb.pData, 
            nLR, nBeta1, nBeta2, nEpsilon, nT
        )

    