# File: src/optim/adam.ring
# Description: Adam Optimizer (Universal Adam Optimizer (Tensor-Level))
# Author: Azzeddine Remmal


class Adam

    # Hyperparameters
    lr          = 0.0001
    beta1       = 0.9
    beta2       = 0.999
    epsilon     = 0.0000001
    weightDecay = 0.0    # L2 Regularization
    
    func init nLearningRate, nWd
        lr = nLearningRate
        weightDecay = nWd 

    # --- New Mode (Transformer/Tensor) ---
    func updateTensor oWeights, oGrads
        
        # 1. Initialize State (Attached to Tensor)
        if !hasAttribute(oWeights, "adam_m")
            addAttribute(oWeights, "adam_m")
            addAttribute(oWeights, "adam_v")
            addAttribute(oWeights, "adam_t")
            
            oWeights.adam_m = new Tensor(oWeights.nRows, oWeights.nCols)
            oWeights.adam_m.zeros()
            oWeights.adam_v = new Tensor(oWeights.nRows, oWeights.nCols)
            oWeights.adam_v.zeros()
            oWeights.adam_t = 0
        ok

        # 2. Update Time
        oWeights.adam_t++
        nT = oWeights.adam_t
        
        # 3. Kernel Update
        tensor_update_adam(
            oWeights.pData, 
            oGrads.pData, 
            oWeights.adam_m.pData, 
            oWeights.adam_v.pData, 
            lr, beta1, beta2, epsilon, nT,
            weightDecay
        )

    # --- Legacy Mode (MLP Examples) ---
    func update oLayer
        
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok
        if !hasAttribute(oLayer, "oWeights") return ok

        # Initialize State (Attached to Layer)
        if !hasAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_vw")
            oLayer.adam_mw = oLayer.oWeights.copy()
            oLayer.adam_mw.zeros() # Explicit zeroing
            oLayer.adam_vw = oLayer.oWeights.copy()
            oLayer.adam_vw.zeros()
            
            addAttribute(oLayer, "adam_mb")
            addAttribute(oLayer, "adam_vb")
            oLayer.adam_mb = oLayer.oBias.copy()
            oLayer.adam_mb.zeros()
            oLayer.adam_vb = oLayer.oBias.copy()
            oLayer.adam_vb.zeros()
            
            addAttribute(oLayer, "adam_t")
            oLayer.adam_t = 0
        ok

        # Time Step
        oLayer.adam_t++
        nT = oLayer.adam_t

        # Kernel Updates
        tensor_update_adam(
            oLayer.oWeights.pData, 
            oLayer.oGradWeights.pData, 
            oLayer.adam_mw.pData, 
            oLayer.adam_vw.pData, 
            lr, beta1, beta2, epsilon, nT,
            weightDecay
        )

        tensor_update_adam(
            oLayer.oBias.pData, 
            oLayer.oGradBias.pData, 
            oLayer.adam_mb.pData, 
            oLayer.adam_vb.pData, 
            lr, beta1, beta2, epsilon, nT,
            weightDecay
        )


