# Project: RingML / Adam
# File: src/model/transformer_block.ring
# Description: Transformer Encoder Block with MultiHeadAttention & Freeze Support
# Author: Azzeddine Remmal

class TransformerBlock

    # Layers
    oAttention
    oNorm1
    
    oFFN_1      
    oGelu
    oFFN_2      
    oNorm2
    
    # Config
    nDim
    nHiddenDim
    bDisplay = false
    nHeads = 4
    # We need to know SeqLen to calculate BatchSize dynamically
    nSeqLen = 0 
    
    bTrainable = true

    func init nModelDim, nSequenceLength, nTotalLayers
        nDim    = nModelDim
        nSeqLen = nSequenceLength
        nHiddenDim = nModelDim * 4 

        oAttention = new MultiHeadAttention(nDim, nHeads, true, :STANDARD)
        oNorm1     = new LayerNorm(nDim)
        
        oFFN_1     = new Dense(nDim, nHiddenDim)
        oGelu      = new GELU
        oFFN_2     = new Dense(nHiddenDim, nDim)
        oNorm2     = new LayerNorm(nDim)

        applyResidualInit(nTotalLayers)

    func forward oInput
        nTotalRows = oInput.nRows
        nBatchSize = nTotalRows / nSeqLen
        
        if (nTotalRows % nSeqLen) != 0
            raise("Error: Input rows ("+nTotalRows+") not divisible by SeqLen ("+nSeqLen+")")
        ok
        
        oNormed1 = oNorm1.forward(oInput)
        oAttnOut = oAttention.forward(oNormed1, nBatchSize, nSeqLen)
        oRes1 = oInput.add(oAttnOut)
        
        oNormed2 = oNorm2.forward(oRes1)
        h1 = oFFN_1.forward(oNormed2)
        h2 = oGelu.forward(h1) 
        oFFNOut = oFFN_2.forward(h2)
        
        oFinalOutput = oRes1.add(oFFNOut)
        return oFinalOutput

    func backward oGradOutput
        oGradRes2_Skip = oGradOutput 
        g_ffn2 = oFFN_2.backward(oGradOutput)
        g_act  = oGelu.backward(g_ffn2)
        g_ffn1 = oFFN_1.backward(g_act)
        oGradNorm2 = oNorm2.backward(g_ffn1)
        oGradRes1 = oGradRes2_Skip.copy().add(oGradNorm2)
        
        oGradRes1_Skip = oGradRes1
        g_attn  = oAttention.backward(oGradRes1)
        g_norm1 = oNorm1.backward(g_attn)
        oFinalGrad = oGradRes1_Skip.copy().add(g_norm1)
        
        return oFinalGrad

    func updateWeights oOptimizer
        if !bTrainable return ok
        oAttention.updateWeights(oOptimizer)
        oNorm1.updateWeights(oOptimizer)
        oFFN_1.updateWeights(oOptimizer)
        oFFN_2.updateWeights(oOptimizer)
        oNorm2.updateWeights(oOptimizer)
    
    func setMode bGraph
        oAttention.setMode(bGraph)
        oNorm1.gamma.setGraphMode(bGraph)
        oNorm1.beta.setGraphMode(bGraph)
        oFFN_1.oWeights.setGraphMode(bGraph)
        oFFN_1.oBias.setGraphMode(bGraph)
        oFFN_2.oWeights.setGraphMode(bGraph)
        oFFN_2.oBias.setGraphMode(bGraph)
        oNorm2.gamma.setGraphMode(bGraph)
        oNorm2.beta.setGraphMode(bGraph)
        
    func freeze
        bTrainable = false
        oAttention.freeze()
        oNorm1.freeze()
        oFFN_1.freeze()
        oFFN_2.freeze()
        oNorm2.freeze()

    func unfreeze
        bTrainable = true
        oAttention.unfreeze()
        oNorm1.unfreeze()
        oFFN_1.unfreeze()
        oFFN_2.unfreeze()
        oNorm2.unfreeze()

    func saveWeights cFile
        oSerializer = new BlockSerializer
        oSerializer.saveBlock(self, cFile)

    func loadWeights cFile
        oSerializer = new BlockSerializer
        oSerializer.loadBlock(self, cFile)
        
    func applyResidualInit nLayers
        if nLayers < 1 nLayers = 1 ok
        nScale = 1.0 / sqrt(2.0 * nLayers)
        oAttention.oOutputLayer.oWeights.scalarMul(nScale)
        oAttention.oOutputLayer.oBias.fill(0)
        oFFN_2.oWeights.scalarMul(nScale)
        oFFN_2.oBias.fill(0)

    func getGradients
        aGradients = []
        aAttnGrads = oAttention.getGradients()
        for item in aAttnGrads aGradients + item next
        aGradients + oNorm1.g_gamma.pData
        aGradients + oNorm1.g_beta.pData
        aGradients + oFFN_1.oGradWeights.pData
        aGradients + oFFN_1.oGradBias.pData
        aGradients + oFFN_2.oGradWeights.pData
        aGradients + oFFN_2.oGradBias.pData
        aGradients + oNorm2.g_gamma.pData
        aGradients + oNorm2.g_beta.pData
        return aGradients

    func getWeightsList
        aTensors = []
        aAttentionHeads = oAttention.getAttentionWeightsList()
        for item in aAttentionHeads aTensors + item next
        aTensors + oNorm1.gamma
        aTensors + oNorm1.beta
        aTensors + oFFN_1.oWeights
        aTensors + oFFN_1.oBias
        aTensors + oFFN_2.oWeights
        aTensors + oFFN_2.oBias
        aTensors + oNorm2.gamma
        aTensors + oNorm2.beta
        return aTensors

    func getParams
        aFlat = []
        for aPair in oAttention.getParams() aFlat + aPair next
        for aPair in oNorm1.getParams() aFlat + aPair next
        for aPair in oFFN_1.getParams() aFlat + aPair next
        for aPair in oFFN_2.getParams() aFlat + aPair next
        for aPair in oNorm2.getParams() aFlat + aPair next
        return aFlat

    func summary
        if isNull(oStyl) oStyl = new Styler() ok
        nCol2Width = 32 
        see nl
        ? oStyl.cyan(:bold,"__________________________________________________________________________")
        ? oStyl.white(:bold,pad("Component (Type)", 25) + pad("Details [State]", nCol2Width) + "Param #" )
        ? oStyl.cyan(:bold,"==========================================================================")
        nTotal = 0
        nTrainableTotal = 0
        nAttnParams = 4 * ((nDim * nDim) + nDim)
        cState = getLayerState(oAttention)
        printRow("MultiHeadAttn", "Heads="+oAttention.nHeads+", D="+nDim + cState, nAttnParams, nCol2Width)
        nTotal += nAttnParams
        if oAttention.bTrainable nTrainableTotal += nAttnParams ok
        nNorm1Params = 2 * nDim
        cState = getLayerState(oNorm1)
        printRow("LayerNorm (1)", "Dim="+nDim + cState, nNorm1Params, nCol2Width)
        nTotal += nNorm1Params
        if oNorm1.bTrainable nTrainableTotal += nNorm1Params ok
        nFFN1 = (nDim * nHiddenDim) + nHiddenDim
        nFFN2 = (nHiddenDim * nDim) + nDim
        cState1 = getLayerState(oFFN_1)
        cState2 = getLayerState(oFFN_2)
        printRow("FFN Expansion", "Dense->"+nHiddenDim + cState1, nFFN1, nCol2Width)
        printRow("FFN Projection", "Dense->"+nDim + cState2, nFFN2, nCol2Width)
        nTotal += (nFFN1 + nFFN2)
        if oFFN_1.bTrainable nTrainableTotal += nFFN1 ok
        if oFFN_2.bTrainable nTrainableTotal += nFFN2 ok
        nNorm2Params = 2 * nDim
        cState = getLayerState(oNorm2)
        printRow("LayerNorm (2)", "Dim="+nDim + cState, nNorm2Params, nCol2Width)
        nTotal += nNorm2Params
        if oNorm2.bTrainable nTrainableTotal += nNorm2Params ok
        ? oStyl.cyan(:NONE,"==========================================================================")
        oStyl.white(:NONE,"Total Params:     ")
        oStyl.cyan(:NONE,"" + nTotal + nl)
        oStyl.white(:NONE,"Trainable Params: ")
        oStyl.green(:NONE,"" + nTrainableTotal + nl)
        oStyl.white(:NONE,"Frozen Params:    ")
        if (nTotal - nTrainableTotal) > 0
            oStyl.red(:NONE,"" + (nTotal - nTrainableTotal) + nl)
        else
            oStyl.white(:NONE,"0" + nl)
        ok
        ? oStyl.cyan(:NONE,"__________________________________________________________________________" + nl)

    private
    func getLayerState oLayer
        if oLayer.bTrainable return " [TRAIN]" else return " [FROZEN]" ok
    func printRow cName, cDetail, nParams, nCol2W
        cCol1 = pad(cName, 25)
        cCol2 = pad(cDetail, nCol2W)
        cCol3 = "" + nParams
        oStyl.yellow(:NONE,cCol1)
        if right(cDetail, 8) = "[FROZEN]"
            oStyl.red(:NONE,cCol2)
        else
            oStyl.cyan(:NONE,cCol2)
        ok
        ? oStyl.green(:NONE,cCol3 )
        ? oStyl.cyan(:NONE,"--------------------------------------------------------------------------")
    func pad cStr, nLen
        if len(cStr) >= nLen return cStr ok
        return cStr + copy(" ", nLen - len(cStr))