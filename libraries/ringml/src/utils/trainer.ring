# Project: RingML
# File: src/utils/trainer.ring
# Description: Advanced Trainer with LR Scheduler & Curriculum Learning
# Author: Azzeddine Remmal


class Trainer

    oModel
    oTrainLoader
    oOptim
    oLoss
    oTokenizer            
    oDashboard 

    nEpochs         = 10
    nLogInterval    = 50
    nSaveInterval   = 10    
    cCheckpointDir  = "model/"
    cModelName      = "model"
    
    bUseScheduler   = true
    nWarmupSteps    = 5
    nBaseLR         = 0.005
    current_lr      = 0
    

    bUseCurriculum  = false
    nMaxSeqLen      = 32    
    nCurrSeqLen     = 32    
    
    bGraphMode      = false
    oGraphOutput    = NULL
    
    # Graph Fixed Tensors
    oInputTensor    = NULL
    oTargetTensor   = NULL
    nInputID        = -1
    nTargetID       = -1

    nCurrentEpoch   = 0
    nTotalLoss = 0

    func init oModelRef, oLoader, oOptimizer, oLossFn
        oModel       = oModelRef
        oTrainLoader = oLoader
        oOptim       = oOptimizer
        oLoss        = oLossFn
        
        if !dirExists(cCheckpointDir) 
            makeDir(cCheckpointDir) 
        ok
        oDashboard = new RogueutilTransformerDashboard()
        

        return self
    
    func enableCurriculum nMaxLen
        bUseCurriculum = true
        nMaxSeqLen     = nMaxLen
        nCurrSeqLen    = 8 
        oDashboard.log("[Trainer]: Curriculum Enabled (Max: "+nMaxLen+")")

    func fit
        oDashboard.start(oModel, nEpochs, oTrainLoader.nBatches)

        oDashboard.log("[Trainer]:Starting Training Epochs:" + nEpochs + " Batches:" + oTrainLoader.nBatches)

        oDashboard.startTimer()

        for i = 1 to nEpochs
           
            # Update Class Attribute manually
            nCurrentEpoch = i

            if bUseScheduler applyScheduler() ok

            if bUseCurriculum updateCurriculum() ok
           
            trainOneEpoch()

            saveCheckpoint()
            
            if !isnull(oTokenizer) generateDemo("I'm old enough to vote.") ok //أنا كبير بما يكفي للتصويت.
        next
        
        oDashboard.finishTraining()

    func trainOneEpoch 
        nTotalLoss = 0
        nValidSamples = 0
        nLastLoss = 0
        
        for b = 1 to oTrainLoader.nBatches
            
            aBatch = oTrainLoader.getBatch(b)

            if !isValidBatch(aBatch) 
                oDashboard.log("[WARNING]: Skipping Batch " + b + ": Invalid format.")
                loop 
            ok

            # Safe Unpack
            oIn  = aBatch[1]
            oTg  = aBatch[2]
            
            # Forward & Backward
            if bGraphMode
                if isnull(oGraphOutput)
                    # 1. Initialize Fixed Tensors for Graph (ONLY ONCE)
                    oInputTensor  = new Tensor(oIn.nRows, oIn.nCols)
                    oTargetTensor = new Tensor(oTg.nRows, oTg.nCols)
                    
                    # 2. Compile Model (Standardized Interface)
                    if isMethod(oModel, "compile")
                        oGraphOutput = oModel.compile(oInputTensor, oTargetTensor, oLoss)
                    else
                        # Manual Fallback
                        oInputTensor.asGraphInput()
                        oTargetTensor.asGraphInput()
                        oLogitsNode = oModel.forward(oInputTensor)
                        oGraphOutput = oLoss.calculate(oLogitsNode, oTargetTensor)
                    ok
                    
                    # Save IDs
                    nInputID  = oInputTensor.nGraphNodeID
                    nTargetID = oTargetTensor.nGraphNodeID

                    # Set optimizer in graph
                    if lower(classname(oOptim)) = "adam"
                        graph_set_optimizer(1) # 1 for Adam
                    else
                        graph_set_optimizer(0) # 0 for SGD
                    ok
                ok
                
                # 3. Copy Data to Fixed Tensors
                oInputTensor.copyData(oIn)
                oTargetTensor.copyData(oTg)
                
                # 4. Update inputs in graph (using saved IDs)
                graph_set_input(nInputID, oInputTensor.pData)
                graph_set_input(nTargetID, oTargetTensor.pData)
                
                # 5. Run Graph
                graph_run(1, oOptim.lr, 5.0)
                
                # 6. Get loss
                pLossNode = graph_get_output(oGraphOutput.nGraphNodeID)
                nErr = tensor_get(pLossNode, 1, 1)
            else
                oLogits = oModel.forward(oIn)
                nErr    = oLoss.calculate(oLogits, oTg)
                
                oGrad   = oLoss.backwardTensor()
                oModel.backward(oGrad)
                
                aAllGrads = oModel.getAllGradients()
                clipGlobalNorm(aAllGrads, 1.0)
                
                oModel.updateWeights(oOptim)
            ok
            
            nTotalLoss += nErr
            nValidSamples++
            
            if b % nLogInterval = 0
                oDashboard.update(nCurrentEpoch, b, nErr, nLastLoss, oOptim.lr)
                nLastLoss = nErr
            ok
            
        next
        
        oDashboard.finishEpoch()
        

    # --- Helpers ---

    func updateCurriculum
        nOldSeq = nCurrSeqLen
        if nCurrentEpoch <= 10 nCurrSeqLen = 8
        elseif nCurrentEpoch <= 20 nCurrSeqLen = 16
        else nCurrSeqLen = nMaxSeqLen ok
        
        if nCurrSeqLen != nOldSeq
            # Update Dataset & Model
            if isMethod(oTrainLoader.oDataset.oParent, "setSequenceLength")
                oTrainLoader.oDataset.oParent.setSequenceLength(nCurrSeqLen)
            ok
            if hasAttribute(oModel, "oTransformer")
                oModel.oTransformer.nSeqLen = nCurrSeqLen
            ok
            # Reset Graph because shapes changed
            oGraphOutput = NULL
            oDashboard.log("[Curriculum]: Level Up! Len -> " + pad(nCurrSeqLen, 5))
        ok

    func applyScheduler
        if nCurrentEpoch <= nWarmupSteps
            currentlr = nBaseLR * (nCurrentEpoch / nWarmupSteps)
        else
            currentlr = nBaseLR * pow(0.95, (nCurrentEpoch - nWarmupSteps))
        ok
        
        if currentlr < 0.00001 currentlr = 0.00001 ok
        oOptim.lr = currentlr
        oDashboard.log("[scheduler]: LR -> " + pad("" +currentlr, 7))

    func saveCheckpoint
        if (nCurrentEpoch % nSaveInterval = 0) or (nCurrentEpoch = nEpochs)
            cPath = cCheckpointDir + cModelName + "_ep" + nCurrentEpoch + ".bin"
            oModel.oTransformer.saveWeights(cPath)
            oDashboard.log("[Checkpoint]: Saved to " + cPath)
        ok

    func generateDemo cPrompt
        nSeq = nMaxSeqLen 
        aIds = oTokenizer.encode(cPrompt)
        cGen = ""
        
        for i=1 to nSeq
            oIn = new Tensor(1, nSeq)
            for k=1 to nSeq oIn.setVal(1, k, 1) next
            
            nTotalIds = len(aIds)
            nStartIdx = 1
            if nTotalIds > nSeq nStartIdx = nTotalIds - nSeq + 1 ok
            
            nCurrentPos = 1
            for k = nStartIdx to nTotalIds
                v = aIds[k] if v=0 v=2 ok
                if nCurrentPos <= nSeq 
                    oIn.setVal(1, nCurrentPos, v)
                    nCurrentPos++
                ok
            next
            
            oLogits = oModel.forward(oIn)
            oLogits.softmax()
            
            nPredPos = nCurrentPos - 1
            if nPredPos < 1 nPredPos = 1 ok
            
            maxV = -1 maxId = 0
            for v=1 to oLogits.nCols 
                val=oLogits.getVal(nPredPos, v) 
                if val>maxV 
                    maxV=val 
                    maxId=v 
                ok 
            next
            
            aIds + maxId
            cToken = oTokenizer.getTokenFromId(maxId)
            
            if maxId = 1 exit ok
            cGen += cToken
        next
        if !isnull(oDashboard) 
            oDashboard.Output(cPrompt + "->" + cGen)
        ok
       

    func isValidBatch aBatch
        if !isList(aBatch) return false ok
        if len(aBatch) < 2 return false ok
        return true

    func pad cStr, nLen
        if len(cStr) >= nLen return cStr ok
        return cStr + copy(" ", nLen - len(cStr))
