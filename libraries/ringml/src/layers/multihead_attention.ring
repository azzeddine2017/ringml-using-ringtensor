/*
    Project: Jabr
    File: src/layers/multihead_attention.ring
    Description: Unified Attention Layer.
                 Supports: Standard (O(N^2)) and Linear (O(N)) modes.
                 Handles Causal Masking automatically.
*/

class MultiHeadAttention

    # --- Configuration ---
    nEmbedDim       
    nHeads
    nHeadDim
    nScale
    
    # Mode: :STANDARD or :LINEAR
    cAttentionType
    bCausal
    
    # --- Layers (Merged Projections) ---
    oW_Q
    oW_K
    oW_V
    oOutputLayer
    
    # --- Cache (For Backward) ---
    oCachedQ
    oCachedK
    oCachedV
    nLastBatch
    nLastSeq

    bTrainable = true
    cName = "MultiHeadAttention"

    func init nDim, nHeadCount, lIsCausal, cType
        nEmbedDim = nDim
        nHeads    = nHeadCount
        bCausal   = lIsCausal
        
        if cType = NULL cType = :STANDARD ok
        cAttentionType = cType
        
        if (nDim % nHeads) != 0
            raise("Error: Dim not divisible by Heads")
        ok
        
        nHeadDim = nDim / nHeads
        nScale   = sqrt(nHeadDim)
        
        oW_Q = new Dense(nDim, nDim)
        oW_K = new Dense(nDim, nDim)
        oW_V = new Dense(nDim, nDim)
        
        oOutputLayer = new Dense(nDim, nDim)

    func forward oInput , nBatch, nSeq
       
        nLastBatch = nBatch
        nLastSeq   = nSeq
        
        oQ = oW_Q.forward(oInput)
        oKey = oW_K.forward(oInput)
        oV = oW_V.forward(oInput)
        
        oAttnOut = new Tensor(oInput.nRows, nEmbedDim)
        nCausalFlag = 0 if bCausal nCausalFlag = 1 ok
        
        if cAttentionType = :STANDARD
            oCachedQ = oQ 
            oCachedK = oKey
            oCachedV = oV
            
            if oQ.bGraphMode
                oAttnOut = oQ.returnGraphNodeAttention(oQ, oKey, oV, 1.0 / nScale, nBatch, nSeq, nHeads, nCausalFlag, 0)
            else
                tensor_attention_multihead(
                    oQ.pData, oKey.pData, oV.pData, oAttnOut.pData,
                    1.0 / nScale,
                    nBatch, nSeq, nHeads, nCausalFlag
                )
            ok
            
        elseif cAttentionType = :LINEAR
            oQ.relu()
            oKey.relu()
            oCachedQ = oQ
            oCachedK = oKey
            oCachedV = oV
            
            if oQ.bGraphMode
                nAttnType = 1 if bCausal nAttnType = 1 else nAttnType = 2 ok
                oAttnOut = oQ.returnGraphNodeAttention(oQ, oKey, oV, 1.0, nBatch, nSeq, nHeads, nCausalFlag, nAttnType)
            else
                if bCausal
                    tensor_attention_linear_causal(oQ.pData, oKey.pData, oV.pData, oAttnOut.pData, 1.0, nBatch)
                else
                    tensor_attention_linear_optimized(oQ.pData, oKey.pData, oV.pData, oAttnOut.pData, 1.0, nBatch)
                ok
            ok
        ok
        
        oFinal = oOutputLayer.forward(oAttnOut)
        return oFinal

    func backward oGradOutput
        oGradConcat = oOutputLayer.backward(oGradOutput)
        
        oGradQ = new Tensor(oGradOutput.nRows, nEmbedDim)
        oGradK = new Tensor(oGradOutput.nRows, nEmbedDim)
        oGradV = new Tensor(oGradOutput.nRows, nEmbedDim)
        
        nCausalFlag = 0 if bCausal nCausalFlag = 1 ok
        
        if cAttentionType = :STANDARD
            tensor_attention_multihead_backward(
                oCachedQ.pData, oCachedK.pData, oCachedV.pData,
                oGradConcat.pData,
                oGradQ.pData, oGradK.pData, oGradV.pData,
                1.0 / nScale,
                nLastBatch, nLastSeq, nHeads, nCausalFlag
            )
        elseif cAttentionType = :LINEAR
            if bCausal
                tensor_attention_linear_backward(oCachedQ.pData, oCachedK.pData, oCachedV.pData, oGradConcat.pData, oGradQ.pData, oGradK.pData, oGradV.pData, 1.0, nLastBatch)
            else
                tensor_attention_linear_global_backward(oCachedQ.pData, oCachedK.pData, oCachedV.pData, oGradConcat.pData, oGradQ.pData, oGradK.pData, oGradV.pData, 1.0, nLastBatch)
            ok
        ok
        
        gQ = oW_Q.backward(oGradQ)
        gK = oW_K.backward(oGradK)
        gV = oW_V.backward(oGradV)
        
        oGradInput = new Tensor(oGradOutput.nRows, oGradOutput.nCols)
        oGradInput.zeros().add(gQ).add(gK).add(gV)
        return oGradInput
    
    func updateWeights oOptim
        if !bTrainable return ok
        oOutputLayer.updateWeights(oOptim)
        oW_Q.updateWeights(oOptim)
        oW_K.updateWeights(oOptim)
        oW_V.updateWeights(oOptim)
        
    func freeze
        bTrainable = false
        oW_Q.freeze() oW_K.freeze() oW_V.freeze()
        oOutputLayer.freeze()

    func unfreeze
        bTrainable = true
        oW_Q.unfreeze() oW_K.unfreeze() oW_V.unfreeze()
        oOutputLayer.unfreeze()

    func getGradients
        aGradients = []
        aGradients + oW_Q.oGradWeights.pData
        aGradients + oW_Q.oGradBias.pData
        aGradients + oW_K.oGradWeights.pData
        aGradients + oW_K.oGradBias.pData
        aGradients + oW_V.oGradWeights.pData
        aGradients + oW_V.oGradBias.pData
        aGradients + oOutputLayer.oGradWeights.pData
        aGradients + oOutputLayer.oGradBias.pData
        return aGradients   
    
    func getAttentionWeightsList
        aWeights = []
        aWeights + oW_Q.oWeights
        aWeights + oW_Q.oBias
        aWeights + oW_K.oWeights
        aWeights + oW_K.oBias
        aWeights + oW_V.oWeights
        aWeights + oW_V.oBias
        aWeights + oOutputLayer.oWeights
        aWeights + oOutputLayer.oBias
        return aWeights   

    func getParams
        aParams = []
        aParams + oW_Q.getParams()
        aParams + oW_K.getParams()
        aParams + oW_V.getParams()
        aParams + oOutputLayer.getParams()
        # Flatten
        aFlat = []
        for aLayerParams in aParams
            for aPair in aLayerParams
                aFlat + aPair
            next
        next
        return aFlat

    func setMode bGraph
        aLayerMode = getAttentionWeightsList()

        for aPair in aLayerMode
            aPair.setGraphMode(bGraph)
        next
