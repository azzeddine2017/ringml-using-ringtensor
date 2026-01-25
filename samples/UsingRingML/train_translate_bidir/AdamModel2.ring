# Project: Adam
# File: src/core/AdamModel2.ring
# Description: The Adam Multi Block (4 Max) Architecture


# ============================================================
#                   AdamModel2 Class
# ============================================================ 
class AdamModel2

    oSerializer
    nVocabSize
    nSeqLen
    nEmbedDim
    nLayers

    bWeightTying = false
    oPosRowCache
    oTokenEmbed
    oPosEmbed
    aBlocks = []
    oHead
    

    # Cache for backward
    oTransposedWeights
    oContextCache
    
    func init nVocab, nSeq, nDim, nLayerCount
        nVocabSize = nVocab
        nSeqLen    = nSeq
        nEmbedDim  = nDim
        nLayers    = nLayerCount
        
        if nLayers < 1 nLayers = 1 ok
        
        see "Initializing Adam Model (" + nLayers + " Layers)..." + nl
        
        # 1. Embeddings
        oTokenEmbed = new Embedding(nVocab, nDim)
        oPosEmbed   = new Embedding(nSeq, nDim)
        
        # New LayerNorm just for Embeddings
        oEmbNorm    = new LayerNorm(nDim) 

        # 2. Transformer Stack
        aBlocks = []
        for i = 1 to nLayers
            see nl + "  Building Block " + i + "..." + nl
            oTransformer = new TransformerBlock(nDim, nSeq, nLayers)
            aBlocks + oTransformer
            aBlocks[i].nSeqLen = nSeq
        next

        if not bWeightTying
            # 3. Head
            oHead = new Dense(nDim, nVocab)
           // oHead.oWeights.scalarMul(0.01)
        ok

    # ========================================
    # FORWARD PASS
    # ========================================

    /* func forward oInputTensor
        
        see nl + "[DEBUG] >> START AdamModel.forward" + nl
        
        # Check Input
        if isnull(oInputTensor) 
            see "[CRASH IMPMINENT] Input Tensor is NULL!" + nl 
            return NULL 
        ok
        see "[DEBUG] Input Shape: " + oInputTensor.nRows + " x " + oInputTensor.nCols + 
            " | Mode: " + oInputTensor.bGraphMode + nl
        
        # ---------------------------------------------------------
        # 1. Token Embeddings
        # ---------------------------------------------------------
        see "[DEBUG] 1. Executing Token Embedding..." + nl
        oTokVec = oTokenEmbed.forward(oInputTensor)
        
        if isnull(oTokVec) see "[CRASH] oTokVec is NULL" + nl return NULL ok
        see "[DEBUG]    Done. Shape: " + oTokVec.nRows + " x " + oTokVec.nCols + nl

        # ---------------------------------------------------------
        # 2. Position Embeddings (Crash Suspect)
        # ---------------------------------------------------------
        see "[DEBUG] 2. Handling Positional Embeddings..." + nl
        nSeq = oInputTensor.nCols 
        
        # A. Cache Check
        if isNull(oPosRowCache) 
             see "[DEBUG]    Initializing oPosRowCache (New)..." + nl
             oPosRowCache = new Tensor(1, nSeq)
             for s=1 to nSeq oPosRowCache.setVal(1, s, s) next
        else
             see "[DEBUG]    Using Existing oPosRowCache..." + nl
             # Re-verify size just in case curriculum changed it
             if oPosRowCache.nCols != nSeq
                 see "[DEBUG]    Resizing oPosRowCache..." + nl
                 oPosRowCache = new Tensor(1, nSeq)
                 for s=1 to nSeq oPosRowCache.setVal(1, s, s) next
             ok
        ok 
        
        # B. Graph Registration (The Dangerous Part)
        if oInputTensor.bGraphMode 
            see "[DEBUG]    Registering Constant in Graph..." + nl
            if oPosRowCache.nGraphNodeID = -1
                oPosRowCache.asGraphConstant()
            ok
        else
            # CRITICAL SAFETY: Ensure we are NOT in Graph Mode during Inference
            if oPosRowCache.bGraphMode
                 see "[DEBUG]    WARNING: Resetting PosCache Mode for Inference..." + nl
                 oPosRowCache.bGraphMode = false
                 oPosRowCache.nGraphNodeID = -1
            ok
        ok
        
        # C. Forward
        see "[DEBUG]    Running PosEmbed Forward..." + nl
        oPosVecSingle = oPosEmbed.forward(oPosRowCache)
        see "[DEBUG]    PosEmbed Done." + nl
        
        # D. Broadcast
        nBatch = oTokVec.nRows / nSeq
        see "[DEBUG]    Broadcasting (Batch=" + nBatch + ")..." + nl
        
        if nBatch > 1
            oPosVec = oPosVecSingle.repeatRows(nBatch)
        else
            oPosVec = oPosVecSingle
        ok

        # ---------------------------------------------------------
        # 3. Combine
        # ---------------------------------------------------------
        see "[DEBUG] 3. Adding Embeddings..." + nl
        oCurrent = oTokVec.add(oPosVec)
        
        # ---------------------------------------------------------
        # 4. Transformer Stack
        # ---------------------------------------------------------
        see "[DEBUG] 4. Entering Transformer Stack (" + nLayers + " layers)..." + nl
        
        for i = 1 to nLayers
            see "[DEBUG]    > Block " + i + " Forward..." + nl
            oCurrent = aBlocks[i].forward(oCurrent)
            see "[DEBUG]    < Block " + i + " Done." + nl
        next

        # ---------------------------------------------------------
        # 5. Head
        # ---------------------------------------------------------
        see "[DEBUG] 5. Output Head..." + nl
        
        if not bWeightTying
            oLogits = oHead.forward(oCurrent)
        else
            see "[DEBUG]    Using Weight Tying..." + nl
            oContextCache = oCurrent
            
            # Check Weights Tensor
            if isnull(oTokenEmbed.weights) see "[CRASH] Token Weights NULL" + nl return NULL ok
            
            see "[DEBUG]    Transposing Weights..." + nl
            oTransposedWeights = oTokenEmbed.weights.transpose()
            
            see "[DEBUG]    Final MatMul..." + nl
            oLogits = oCurrent.matmul(oTransposedWeights)
        ok
        
        see "[DEBUG] << END AdamModel.forward" + nl
        return oLogits
 */


    func forward oInputTensor
        
        # 1. Token Embeddings
        oTokVec = oTokenEmbed.forward(oInputTensor)
        
        # --- 2. Position Embeddings (Stable) ---
        nSeq = oInputTensor.nCols 
        
        # A. Update Cache if size changed
        if isNull(oPosRowCache) or oPosRowCache.nCols != nSeq
             oPosRowCache = new Tensor(1, nSeq)
             for s=1 to nSeq oPosRowCache.setVal(1, s, s) next
             # Ensure clean state
             oPosRowCache.nGraphNodeID = -1 
             oPosRowCache.bGraphMode = false
        ok 
        
        # B. Handle Graph/Eager Mode safely
        if oInputTensor.bGraphMode 
            if oPosRowCache.nGraphNodeID = -1
                oPosRowCache.asGraphConstant()
            ok
        else
            # FORCE EAGER MODE for Cache if Input is Eager
            if oPosRowCache.bGraphMode
                 oPosRowCache.bGraphMode = false
                 oPosRowCache.nGraphNodeID = -1
            ok
        ok
        
        # C. Forward & Broadcast
        oPosVecSingle = oPosEmbed.forward(oPosRowCache)
        
        nBatch = oTokVec.nRows / nSeq
        if nBatch > 1
            oPosVec = oPosVecSingle.repeatRows(nBatch)
        else
            oPosVec = oPosVecSingle
        ok

        # 3. Combine & Scale
        # Optional: Add Scaling here if not in Brain class
        # oTokVec.scalar_mul(sqrt(nEmbedDim)) 
        
        oCurrent = oTokVec.add(oPosVec)
        
        # 4. Transformer Stack
        for i = 1 to nLayers
            oCurrent = aBlocks[i].forward(oCurrent)
        next

        # 5. Output Head
        if not bWeightTying
            oLogits = oHead.forward(oCurrent)
        else
            oContextCache = oCurrent
            oTransposedWeights = oTokenEmbed.weights.transpose()
            oLogits = oCurrent.matmul(oTransposedWeights)
        ok
        
        return oLogits
    
    # ========================================
    # AUTO-GRAPH COMPILATION
    # ========================================
    func compile oInputTensor, oTargetTensor, oLoss
        # 1. Enable Graph Mode for all weights
        aParams = getParams()
        for aPair in aParams
            oW = aPair[1]
            oG = aPair[2]
            oW.asGraphWeight(oG)
        next
        
        # 2. Set Input as Graph Input
        oInputTensor.asGraphInput()
        oTargetTensor.asGraphInput()
        
        # 3. Record Forward Pass
        oLogits = forward(oInputTensor)
        
        # 4. Record Loss
        oLossNode = oLoss.calculate(oLogits, oTargetTensor)
        
        # 5. Return Loss Node
        return oLossNode

    func getParams
        aParams = []
        
        # Embeddings
        for aPair in oTokenEmbed.getParams() aParams + aPair next
        for aPair in oPosEmbed.getParams()   aParams + aPair next
        
        # Blocks
        for i = 1 to nLayers
            aBlockParams = aBlocks[i].getParams()
            for aPair in aBlockParams
                aParams + aPair
            next
        next
        
        if not bWeightTying
            # Head
            for aPair in oHead.getParams() aParams + aPair next
        ok
        
        return aParams
        
    func setMode bGraph
        # 1. Embeddings
        oTokenEmbed.weights.setGraphMode(bGraph)
        oPosEmbed.weights.setGraphMode(bGraph)
        
        # 2. Transformer Blocks
        for oBlock in aBlocks
            oBlock.setMode(bGraph)
        next
        
        # 3. Head (if exists)
        if !bWeightTying and !isnull(oHead)
            oHead.oWeights.setGraphMode(bGraph)
            oHead.oBias.setGraphMode(bGraph)
        ok

    # ========================================
    # BACKWARD PASS (FIXED)
    # ========================================
    func backward oGradOutput
        if oGradOutput.bGraphMode
            oGradOutput.backward()
            return NULL
        ok

        if not bWeightTying
            # 1. Head Backward
            oCurrentGrad = oHead.backward(oGradOutput)
        else
            # 1. Output Head Backward (Weight Tying)
            oGradContext = oGradOutput.matmul(oTokenEmbed.weights)
            
            # dL/dW_embed = Context^T × dL/dLogits
            oContextT = oContextCache.transpose()
            oGradW_Head = oContextT.matmul(oGradOutput)
            oGradW_Final = oGradW_Head.transpose()
            
            # Accumulate to embedding gradients
            oTokenEmbed.grads.add(oGradW_Final)
            
            # 2. Backprop through Transformer Stack (REVERSE)
            oCurrentGrad = oGradContext
        ok
        
        for i = nLayers to 1 step -1
            oCurrentGrad = aBlocks[i].backward(oCurrentGrad)
        next
        
        # 3. Split gradients for Token + Position embeddings
        oGradTokVec = oCurrentGrad.copy()
        oGradPosVec = oCurrentGrad.copy()
        
        # 4. Reverse the scaling (Handled by Graph or manual if Eager)
        # Removed manual scaling to prevent explosion
        
        # 5. Embedding Backward
        oTokenEmbed.backward(oGradTokVec)
        oPosEmbed.backward(oGradPosVec)
        
        return NULL
    
    # ========================================
    # UPDATE WEIGHTS
    # ========================================
    func updateWeights oOptimizer
        oTokenEmbed.updateWeights(oOptimizer)
        oPosEmbed.updateWeights(oOptimizer)
        
        for i = 1 to nLayers
            aBlocks[i].updateWeights(oOptimizer)
        next

        if not bWeightTying
            oHead.updateWeights(oOptimizer)
        ok

    #==============================================
    #           Weights Collection For Saving
    #==============================================
    func getAllWeights
        aWeightsList = []
        
        # Embeddings
        aWeightsList + oTokenEmbed.weights
        # PosEmbed (If learned)
        aWeightsList + oPosEmbed.weights 
        
        # Blocks
        if isList(aBlocks)
            for oBlock in aBlocks
                for oTensor in oBlock.getWeightsList() # We collect the list of tensors
                    aWeightsList + oTensor
                next
            next
        else
            for oTensor in oTransformer.getWeightsList()
                aWeightsList + oTensor
            next
        ok
        
        if not bWeightTying
            # Head
            if hasAttribute(oHead, "oWeights")
                aWeightsList + oHead.oWeights
            ok
            if hasAttribute(oHead, "oBias")
                aWeightsList + oHead.oBias
            ok
        ok

        return aWeightsList

    # ========================================
    # GRADIENT COLLECTION FOR Clearing
    # ========================================
    func getAllGradients
        aAllGrads = []
        
        # Token embedding
        aAllGrads + oTokenEmbed.grads.pData
        
        # Position embedding
        if hasAttribute(oPosEmbed, "grads")
            aAllGrads + oPosEmbed.grads.pData
        ok
        
        # 3. All Blocks (CRITICAL FIX!)
        for i = 1 to nLayers
            aBlockGrads = aBlocks[i].getGradients()
            
            if isList(aBlockGrads)
                for pGrad in aBlockGrads
                    aAllGrads + pGrad
                next
            else
                aAllGrads + aBlockGrads
            ok
        next

        if not bWeightTying
            # 4. Head
            if hasAttribute(oHead, "oGradWeights")
                aAllGrads + oHead.oGradWeights.pData
            ok

            if hasAttribute(oHead, "oGradBias")
                aAllGrads + oHead.oGradBias.pData
            ok  
        ok
        return aAllGrads

    # ========================================
    # MODEL STATS
    # ========================================
    func getModelStats
        nTotal = 0
        nTrainable = 0
        aBlockStats = []
        
        # Embeddings (approximate)
        nEmbedParams = (nVocabSize * nEmbedDim) + (nSeqLen * nEmbedDim)
        nTotal += nEmbedParams
        nTrainable += nEmbedParams
        
        # Blocks
        for i = 1 to nLayers
            aStats = aBlocks[i].getParams()
            aBlockStats + aStats
            nTotal += aStats[:TotalParams]
            nTrainable += aStats[:Trainable]
        next
        
        return [
            :TotalParams = nTotal,
            :Trainable   = nTrainable,
            :Frozen      = nTotal - nTrainable,
            :Blocks      = aBlockStats,
            :Layers      = nLayers
        ]
    
    # ========================================
    # UTILITIES
    # ========================================
    func blocksSummary
        ? oStyl.blue(:BOLD, nl + "===================================")
        ? oStyl.blue(:BOLD,"Adam Model Summary")
        ? oStyl.blue(:BOLD,"===================================")
          oStyl.blue(:BOLD,"Vocab Size: ") 
        ? oStyl.cyan(:BOLD,"" +  nVocabSize)
          oStyl.blue(:BOLD,"Sequence Length: ") 
        ? oStyl.cyan(:BOLD,"" + nSeqLen)
          oStyl.blue(:BOLD,"Embedding Dim: ") 
        ? oStyl.cyan(:BOLD,"" + nEmbedDim)
          oStyl.blue(:BOLD,"Number of Layers: ") 
        ? oStyl.cyan(:BOLD,"" + nLayers)
        if bWeightTying ? oStyl.blue(:BOLD,"Practise Weight Tying Mode") ok
        ? oStyl.blue(:BOLD,"===================================")
        
        for block = 1 to nLayers
              oStyl.green(:BOLD,nl + "Block ")
              oStyl.bright_blue(:BOLD,"" + block)
            ? oStyl.green(:BOLD," Summary:")

            aBlocks[block].summary()
        next
        
        ? nl + "==================================="
        aStats = getModelStats()
          oStyl.blue(:BOLD,"Total Parameters: ")
        ? oStyl.bright_green(:BOLD,"" + aStats[:TotalParams] )
          oStyl.blue(:BOLD,"Trainable: ")
        ? oStyl.bright_green(:BOLD,"" + aStats[:Trainable])
        ? oStyl.blue(:BOLD,"===================================")

    #=========================================================
    # Save and Load Weights
    #=========================================================
    func saveModel cFile
        oSerializer = new ModelSerializer
        oSerializer.saveModel(self, cFile)

    # ---------------------
    func loadModel cFile
        oSerializer = new ModelSerializer
        oSerializer.loadModel(self, cFile)

