# File: src/model/sequential.ring
# Description: Sequential Model Container with Save/Load support
# Author: Azzeddine Remmal

class Sequential
    aLayers = []

    func init
        return self

    func add oLayer
        aLayers + oLayer
        return self

    func forward oInput
        oCurrent = oInput
        nBatch = oInput.nBatch
        if nBatch = 0 nBatch = 1 ok
        nSeq   = oInput.nRows / nBatch
        for oLayer in aLayers
            if classname(oLayer) = "multiheadattention"
                oCurrent = oLayer.forward(oCurrent, nBatch, nSeq)
            else
                oCurrent = oLayer.forward(oCurrent)
            ok
        next
        return oCurrent

    func compile oInputTemplate
        oInputTemplate.setGraphMode(true)
        oOutput = forward(oInputTemplate)
        return oOutput # Returns the output virtual tensor (Graph Node)

    func backward oGradOutput
        oCurrentGrad = oGradOutput
        for i = len(aLayers) to 1 step -1
            oCurrentGrad = aLayers[i].backward(oCurrentGrad)
        next
        return oCurrentGrad
        
    func getLayers
        return aLayers

    # --- Mode Switching ---
    func train
        for oLayer in aLayers oLayer.train() next
    func evaluate
        for oLayer in aLayers oLayer.evaluate() next

    # --- Save & Load ---
    func saveWeights cFileName
        see "Saving model to " + cFileName + "..." + nl
        
        aAllParams = []
        for oLayer in aLayers
            if hasParams(oLayer)
                # FIX: Convert Pointer-Tensor to List for saving
                aAllParams + oLayer.oWeights.toList()
                aAllParams + oLayer.oBias.toList()
            ok
        next
        
        cData = SerializeData(aAllParams)
        write(cFileName, cData)
        see "Done." + nl

    func loadWeights cFileName
        see "Loading model from " + cFileName + "..." + nl
        if !fexists(cFileName) raise("Error: File not found") ok
        
        cCode = "return " + read(cFileName)
        aAllParams = eval(cCode)
        
        nIdx = 1
        for oLayer in aLayers
            if hasParams(oLayer)
                if nIdx > len(aAllParams) raise("Architecture Mismatch") ok
                
                # FIX: Fill Pointer-Tensor from List
                oLayer.oWeights.fromList(aAllParams[nIdx])
                nIdx++
                oLayer.oBias.fromList(aAllParams[nIdx])
                nIdx++
            ok
        next
        see "Done." + nl

    func getParams
        aAll = []
        for oLayer in aLayers
            if isMethod(oLayer, "getParams")
                aLayerParams = oLayer.getParams()
                for aPair in aLayerParams
                    aAll + aPair
                next
            ok
        next
        return aAll

	 # --- Model Summary (Colorful Version) ---

    func summary
        # Ensure console colors are available (usually loaded by ringml.ring)
        # If not, this might throw error, but we assume environment is set.
        
        see nl
        ? oStyl.cyan(:bold,"_________________________________________________________________")
        ? oStyl.white(:bold,pad("Layer (Type)", 29) + pad("Output Shape", 26) + "Param #" )
        ? oStyl.cyan(:bold,"=================================================================")
        
        nTotalParams = 0
        nTrainableParams = 0
        nNonTrainableParams = 0
        
        cLastOutputShape = "Input" 

        for oLayer in aLayers
            cName = classname(oLayer)
            nParams = 0
            
            # --- Logic to extract info ---
            if cName = "dense"
                nW = oLayer.nInputSize * oLayer.nNeurons
                nB = oLayer.nNeurons
                nParams = nW + nB
                cOutputShape = "(None, " + oLayer.nNeurons + ")"
                cLastOutputShape = cOutputShape
            else
                nParams = 0
                cOutputShape = cLastOutputShape 
            ok
            
            nTotalParams += nParams
            
            # --- Count Trainable vs Non-Trainable ---
            if hasAttribute(oLayer, "bTrainable") and oLayer.bTrainable
                nTrainableParams += nParams
            else
                nNonTrainableParams += nParams
            ok
            
            # --- Colorful Printing ---
            cCol1 = pad(cName, 29)
            cCol2 = pad(cOutputShape, 26)
            cCol3 = "" + nParams
            
            # Name in Yellow
            oStyl.yellow(:NONE,cCol1)
            
            # Shape in Cyan
            oStyl.cyan(:NONE,cCol2)
            
            # Params in Green
            ? oStyl.green(:NONE,cCol3 ) 
            
            # Separator in Dark Gray (Subtle)
            ? oStyl.cyan(:NONE,"_________________________________________________________________")
        next
        
        # --- Footer ---
        see nl
        oStyl.white(:NONE,"Total params:         ")
        ? oStyl.cyan(:NONE,"" + nTotalParams )
        
        oStyl.white(:NONE,"Trainable params:     ")
        ? oStyl.green(:NONE,"" + nTrainableParams )
        
        oStyl.white(:NONE,"Non-trainable params: ")
        
        if nNonTrainableParams > 0
            ? oStyl.red(:NONE,"" + nNonTrainableParams )
        else
            ? oStyl.white(:NONE,"0")
        ok
        
         ? oStyl.cyan(:NONE,"_________________________________________________________________" + nl)

    private
    
    # Helper to check if layer is trainable
	func hasParams oLayer
        aAttrs = attributes(oLayer)
        bHasW = false
        bHasB = false
        for cAttr in aAttrs
            if lower(cAttr) = "oweights" bHasW = true ok
            if lower(cAttr) = "obias"    bHasB = true ok
        next
        return (bHasW and bHasB)

	# Helper for string padding
    func pad cStr, nLen
        if len(cStr) >= nLen return cStr ok
        return cStr + copy(" ", nLen - len(cStr))
