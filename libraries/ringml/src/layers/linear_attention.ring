# Project: Jabr
# File: src/layers/linear_attention.ring
# Description: Linear Attention Mechanism O(N)
# Formula: ReLU(Q) * (ReLU(K)^T * V)



class LinearAttention

    nEmbedDim
    nHeads
    nHeadDim
    nScale
    nLastBatch

    aQueryLayers = []
    aKeyLayers   = []
    aValueLayers = []
    oOutputLayer
    
    # Cache lists
    aCachedQ = []
    aCachedK = []
    aCachedV = []

    oInputCached 

    bTrainable = true
    cName = "LinearAttention"

    func init nDim, nHeadCount
        nEmbedDim = nDim
        nHeads    = nHeadCount
        nHeadDim  = nDim / nHeads
        nScale    = sqrt(nHeadDim)
        
        for i = 1 to nHeads
            aQueryLayers + new Dense(nDim, nHeadDim)
            aKeyLayers   + new Dense(nDim, nHeadDim)
            aValueLayers + new Dense(nDim, nHeadDim)
        next
        oOutputLayer = new Dense(nDim, nDim)

    func forward oInput
        if paraCount() >= 2 nBatch = para(2) else nBatch = 1 ok
        if paraCount() >= 3 nSeq = para(3) else nSeq = oInput.nRows ok

        oInputCached = oInput
        nLastBatch = nBatch
        
        # --- FIX 1: Clear Cache Lists ---
        aCachedQ = []
        aCachedK = []
        aCachedV = []
        
        oConcat = new Tensor(oInput.nRows, nEmbedDim)
        
        for i = 1 to nHeads
            # Projections
            oQ = aQueryLayers[i].forward(oInput)
            aCachedQ + oQ
            oKey = aKeyLayers[i].forward(oInput)
            aCachedK + oKey
            oV = aValueLayers[i].forward(oInput)
            aCachedV + oV
            
            # ReLU Feature Map
            oQ.relu()
            oKey.relu()
            
            # Kernel
            oHeadOut = new Tensor(oInput.nRows, nHeadDim)
            tensor_attention_linear_optimized(
                oQ.pData, oKey.pData, oV.pData, 
                oHeadOut.pData, 
                1.0 / nScale,
                nBatch 
            )
            
            # Concat
            nStartCol = ((i-1) * nHeadDim) + 1
            oConcat.insertColumns(oHeadOut, nStartCol)
        next
        
        oFinal = oOutputLayer.forward(oConcat)
        return oFinal

    func backward oGradOutput
        
        # 1. Output Projection
        oGradConcat = oOutputLayer.backward(oGradOutput)
        
        # 2. Input Accumulator
        oGradInput = new Tensor(oGradOutput.nRows, oGradOutput.nCols)
        oGradInput.zeros()
        
        for i = 1 to nHeads
            nStartCol = ((i-1) * nHeadDim) + 1
            oHeadGrad = oGradConcat.selectColumns(nStartCol, nHeadDim)
            
            # --- FIX 2: Create FRESH tensors inside the loop ---
            # These must be (Seq x HeadDim) every time
            oTempGQ = new Tensor(oInputCached.nRows, nHeadDim)
            oTempGK = new Tensor(oInputCached.nRows, nHeadDim)
            oTempGV = new Tensor(oInputCached.nRows, nHeadDim)
            
            # Note: C Kernel overwrites them, no need to zero if fully written,
            # but safer to zero if logic is sparse. C kernel handles it.
            
            # 3. Call C-Kernel
            tensor_attention_linear_backward(
                aCachedQ[i].pData, 
                aCachedK[i].pData, 
                aCachedV[i].pData,
                oHeadGrad.pData,
                oTempGQ.pData, 
                oTempGK.pData, 
                oTempGV.pData,
                1.0 / nScale,
                nLastBatch
            )
            
            # 4. Propagate through Dense
            # Use different variable names to avoid overwriting the containers
            oG_Q_Final = aQueryLayers[i].backward(oTempGQ)
            oG_K_Final = aKeyLayers[i].backward(oTempGK)
            oG_V_Final = aValueLayers[i].backward(oTempGV)
            
            # 5. Accumulate
            oGradInput.add(oG_Q_Final)
            oGradInput.add(oG_K_Final)
            oGradInput.add(oG_V_Final)
        next
        
        return oGradInput

    func updateWeights oOptim
        if !bTrainable return ok
        oOutputLayer.updateWeights(oOptim)
        for i = 1 to nHeads
            aQueryLayers[i].updateWeights(oOptim)
            aKeyLayers[i].updateWeights(oOptim)
            aValueLayers[i].updateWeights(oOptim)
        next
        
     # --- Freeze/Unfreeze Control ---
    func freeze
        bTrainable = false
        oOutputLayer.freeze()
        for i=1 to nHeads 
            aQueryLayers[i].freeze()
            aKeyLayers[i].freeze()
            aValueLayers[i].freeze()
        next

    func unfreeze
        bTrainable = true
        oOutputLayer.unfreeze()
        for i=1 to nHeads 
            aQueryLayers[i].unfreeze()
            aKeyLayers[i].unfreeze()
            aValueLayers[i].unfreeze()
        next

    func getGradients
        aGradients = []
        
        # 1. Attention Gradients (Multi-Head)
        # We iterate over all heads and collect gradients from internal Dense layers
        for i = 1 to nHeads
            # Query
            aGradients + aQueryLayers[i].oGradWeights.pData
            aGradients + aQueryLayers[i].oGradBias.pData
            
            # Key
            aGradients + aKeyLayers[i].oGradWeights.pData
            aGradients + aKeyLayers[i].oGradBias.pData
            
            # Value
            aGradients + aValueLayers[i].oGradWeights.pData
            aGradients + aValueLayers[i].oGradBias.pData
        next
        
        # Output Projection of Attention
        aGradients + oOutputLayer.oGradWeights.pData
        aGradients + oOutputLayer.oGradBias.pData
        
        return aGradients   