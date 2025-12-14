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
        # 1. Validation
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok
        if !hasAttribute(oLayer, "oWeights") return ok

        # 2. State Initialization (Run once per layer)
        if !hasAttribute(oLayer, "adam_mw")
            # State for Weights
            addAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_vw")
            oLayer.adam_mw = oLayer.oWeights.copy().zeros()
            oLayer.adam_vw = oLayer.oWeights.copy().zeros()
            
            # State for Bias
            addAttribute(oLayer, "adam_mb")
            addAttribute(oLayer, "adam_vb")
            oLayer.adam_mb = oLayer.oBias.copy().zeros()
            oLayer.adam_vb = oLayer.oBias.copy().zeros()
            
            addAttribute(oLayer, "adam_t")
            oLayer.adam_t = 0
        ok

        # 3. Time Step
        oLayer.adam_t++
        nT = oLayer.adam_t

        # 4. Updates (Optimized Calls)
        update_param_manual(
            oLayer.oWeights, 
            oLayer.oGradWeights, 
            oLayer.adam_mw, 
            oLayer.adam_vw, 
            nT
        )

        update_param_manual(
            oLayer.oBias, 
            oLayer.oGradBias, 
            oLayer.adam_mb, 
            oLayer.adam_vb, 
            nT
        )

    func update_param_manual oParam, oGrad, oM, oV, nT
        # Pre-calc corrections to save CPU
        correction1 = 1.0 - pow(nBeta1, nT)
        correction2 = 1.0 - pow(nBeta2, nT)
        
        # Safety for very first step or precision errors
        if correction1 = 0 correction1 = 0.00000001 ok
        if correction2 = 0 correction2 = 0.00000001 ok

        nRows = oParam.nRows
        nCols = oParam.nCols
        
        # Extract Lists (References)
        aW = oParam.aData
        aG = oGrad.aData
        aM = oM.aData
        aV = oV.aData
        
        for r = 1 to nRows
            for c = 1 to nCols
                g = aG[r][c]
                
                # --- Adam Logic ---
                
                # 1. Update Momentum (m)
                # m = beta1 * m + (1-beta1) * g
                aM[r][c] = (nBeta1 * aM[r][c]) + ((1.0 - nBeta1) * g)
                
                # 2. Update Velocity (v)
                # v = beta2 * v + (1-beta2) * g^2
                aV[r][c] = (nBeta2 * aV[r][c]) + ((1.0 - nBeta2) * (g * g))
                
                # 3. Bias Correction
                m_hat = aM[r][c] / correction1
                v_hat = aV[r][c] / correction2
                
                # 4. Calculate Step
                # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
                
                # Abs safety to prevent NaN from sqrt(-0.000...)
                if v_hat < 0 v_hat = 0 ok 
                
                nDenom = sqrt(v_hat) + nEpsilon
                nStep = (nLR * m_hat) / nDenom
                
                # 5. Apply Update
                aW[r][c] -= nStep
            next
        next
        
        # Force update list back to object (Just in case of reference loss)
        oParam.aData = aW
        oM.aData = aM
        oV.aData = aV