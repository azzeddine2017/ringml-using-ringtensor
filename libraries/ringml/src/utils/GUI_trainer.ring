# Project: RingML
# File: src/utils/GUI_trainer.ring
# Description: Advanced Trainer with LR Scheduler & Curriculum Learning
# Author: Azzeddine Remmal


oTrainer = null

class GUI_Trainer

    # Components
    oModel 
	oTrainLoader 
	oOptim 
	oLoss 
	oTok
    oScheduler
    
    # Config
    nEpochs         
    nLogInterval   
    nSaveInterval  
    nSeqLen
    cCheckpointDir  = "model/"
    cModelName      = "model"
    cObjectName
    bMontoring      = false

    # Scheduler & Curriculum
    bUseScheduler = true 
    nBaseLR = 0.0001
    nMinLR = 0.00001
    nWarmupSteps = 50 
    nDecayRate = 0.98
    nGlobalStep = 0
    current_lr = 0
    pi = 3.14159

    # Curriculum
    bUseCurriculum = false 
	nMaxSeqLen = 32 
	nCurrSeqLen = 32
    nBatchSize = 0
    # --- State Machine Variables ---
    nCurrentEpoch = 1
    nCurrentBatch = 1
    nTotalBatches = 0
    
    # Stats Accumulators
    nEpochLoss = 0
    nValidSamples = 0
    nLastLoss = 0

    # Early Stopping
    nBestLoss = 99
    nWorseCount = 0
    
    nPatienceMax = 30   # Early stopping patience
    
    
    # GUI & Timer
    oGUI
    oTimer
    oDiag

    # ---------------------------------------------------------
    #  Constructor
    # ---------------------------------------------------------
    func init oModelRef, oTokenizer, oLoader, oOptimizer, oLossFn, cObjectN
        oModel = oModelRef
        oTok   = oTokenizer
        oTrainLoader = oLoader
        oOptim = oOptimizer
        oLoss = oLossFn
        cObjectName = cObjectN
        nBatchSize = oTrainLoader.nBatchSize

        # Checkpoints
        if !dirExists(cCheckpointDir) makeDir(cCheckpointDir) ok

        # GUI
        oGUI = new GUI_Dashboard() 
        nTotalBatches = oTrainLoader.nBatches
        
        # Scheduler
        nFactor     = 0.5       # Reduction factor (50%)
        nPatience   = 3         # Number of patience batches
        oScheduler = new SmartScheduler(oOptim, nPatience, nFactor)
        oScheduler.bControlWD = true 
        
        return self

    # ---------------------------------------------------------
    #  Enable Curriculum
    # ---------------------------------------------------------
    func enableCurriculum nMaxLen
        bUseCurriculum = true
        nMaxSeqLen = nMaxLen
        nCurrSeqLen = 8
        info("[Trainer] Curriculum Enabled (Max: " + nMaxLen + ")")

    # ---------------------------------------------------------
    #  Run entry point
    # ---------------------------------------------------------
    func fit
        ? oStyl.green(:NONE, nl + ">>> Starting GUI Training Engine <<<")
        if bMontoring
           ? oStyl.green(:NONE, "Setup:")
            oStyl.green(:NONE, "  Batch size: ")
           ? oStyl.yellow(:NONE,"" + nBatchSize)
             oStyl.green(:NONE, "  Base LR: " )
           ? oStyl.yellow(:NONE,"" + nBaseLR)
             oStyl.green(:NONE, "  Warmup: ")
             oStyl.yellow(:NONE,"" + nWarmupSteps ) 
           ? oStyl.green(:NONE," steps")
             oStyl.green(:NONE, "  Decay rate: ")
           ? oStyl.yellow(:NONE,"" + nDecayRate)
            ? oStyl.green(:NONE, "")
        ok  
        
        # Create Diagnostics
        oDiag = new GradientDiagnostics(oModel)

        # Timer
        oTimer = new qTimer(oGUI.oWin) 
        oTimer.setInterval(1) 
        oTimer.setTimeoutEvent(cObjectName + ".trainingStep()") 
        oTimer.start()
        
        # 3. Start Timer
        GuiStartTimer()
        
        # 4. Qt
        oGUI.oApp.exec()
    
    # ---------------------------------------------------------
    #  Training Step
    # ---------------------------------------------------------
    func trainingStep
        
        # 1. if we reach the end of epochs
        if nCurrentEpoch > nEpochs
            oTimer.stop()
            info("Training Finished.")
            oGUI.finishTraining()
            return
        ok
        
        # 2. process batch
        processBatch()
        
        # 3. go to the next batch
        nCurrentBatch++
        
        # 4. if we reach the end of the epoch
        if nCurrentBatch > nTotalBatches
            endOfEpochLogic()
            
            # Reset for next epoch
            nCurrentEpoch++
            nCurrentBatch = 1
            nEpochLoss = 0
            nValidSamples = 0
            //nLastLoss = 100
            GuiStartTimer() # Reset epoch timer
        ok

    # ---------------------------------------------------------
    #  Process Batch
    # ---------------------------------------------------------
    func processBatch

        nGlobalStep ++

        #  update LR every step behind warmup
        if nGlobalStep <= nWarmupSteps
            current_lr = nBaseLR * (nGlobalStep / nWarmupSteps)
            if current_lr > nBaseLR current_lr = nBaseLR ok
        ok
        oOptim.lr = current_lr

        # Get Batch
        aBatch = oTrainLoader.getBatch(nCurrentBatch)
        
        if !isList(aBatch) return ok
        
        # Unpack
        if isList(aBatch[1]) 
             oInput = aBatch[1][1] 
             oTarget = aBatch[1][2]
        else
             oInput = aBatch[1]
             oTarget = aBatch[2]
        ok
        
        # Curriculum Logic
       /* if bUseCurriculum
            aSliced = sliceBatch(oInput, oTarget)
            oInput  = aSliced[1]
            oTarget = aSliced[2]
            if hasAttribute(oModel, "oTransformer")
                oModel.oTransformer.nSeqLen = nCurrSeqLen
            ok
        ok*/
        
        # Neural Operations
        oLogits = oModel.forward(oInput)
        nErr    = oLoss.calculate(oLogits, oTarget)

        # --- Instant autopilot call---
        oScheduler.stepBatch(nErr)
        
        if bMontoring
            # Check for Null
            if isNull(nErr) or nErr > 100
                  oStyl.red(:NONE, "❌ Training collapsed at Epoch " )
                ? oStyl.yellow(:NONE,"" +  nCurrentEpoch)
                  oStyl.red(:NONE, "   Loss: " )
                ? oStyl.yellow(:NONE,"" + nErr)
                  oStyl.red(:NONE, "   LR: " )
                ? oStyl.yellow(:NONE,"" + current_lr)

                oGUI.oApp.quit()
            ok
        ok

        oGrad   = oLoss.backwardTensor()
        oModel.backward(oGrad)
        
        # Diagnose before clipping
        if nCurrentBatch % 50 = 0
            oDiag.diagnose("Epoch" + nCurrentEpoch + "_Batch" + nCurrentBatch)
        ok
        
        # Gradient Clipping (Mega Kernel)
        # 1. We gather all the indicators (very quickly)
        aPtrs = oModel.getAllGradients()

        nGradNorm = calcListNorm(aPtrs)
        if nCurrentBatch % 10 = 0
            oStyl.cyan(:BOLD, "✅ Gradient Norm: ")
            oStyl.white(:NONE, "" + nGradNorm)
            ? oStyl.cyan(:BOLD, "")
        ok

        if nGradNorm > 5 # Explosion threshold (was 20, make it 5 to detect it early)
            
            # Instead of drastic reduction, we return to a "safe" but not too small learning rate
            # We also re-activate the Warmup for a short period
            
            oOptim.lr = nBaseLR * 0.5 # Return to half of the maximum learning rate
            
            # Trick: We reset the Scheduler counter a bit back
            # So it can gradually raise the rate again (smaller Warmup)
            if nCurrentEpoch > nWarmupSteps
                nCurrentEpoch = nWarmupSteps - 2 
            ok
            
            if bMontoring 
                oStyl.red(:BOLD, "✅ Gradient Spike (")
                oStyl.yellow(:NONE, "" + nGradNorm)
                ? oStyl.red(:BOLD, ") -> Performing Soft Reset & Re-Warmup") 
            ok

            # We clip the gradients strongly this time only to prevent the current catastrophe
            clipGlobalNorm(aPtrs, 5.0) 
            if bMontoring 
                nClipedGradNorm = calcListNorm(aPtrs)
                oStyl.red(:BOLD, "Gradient Spike | Cliped Gradient Norm: " )
                ? oStyl.yellow(:NONE, "" + nClipedGradNorm)
            ok
        /*else
            # Normal state: regular clipping
            clipGlobalNorm(aPtrs, 5.0)
            if bMontoring
                nClipedGradNorm = calcListNorm(aPtrs)
                oStyl.cyan(:BOLD, "✅ regular clipping Gradient Norm: " )
                ? oStyl.white(:NONE, "" + nClipedGradNorm)
            ok*/
        ok
        
        oModel.updateWeights(oOptim)

        # Check for problems BEFORE backward
    if isNull(nErr) or isNull(nErr) or nErr > 50
        oStyl.red(:NONE, "❌ INVALID LOSS at Epoch ")
        oStyl.yellow(:NONE, "" + nCurrentEpoch)
        oStyl.red(:NONE, " Batch ")
        oStyl.yellow(:NONE, "" + nCurrentBatch)
        ? oStyl.red(:NONE, " | Loss: " + nErr)
        
        # Reduce LR immediately
        oOptim.lr = oOptim.lr * 0.1
        ? oStyl.yellow(:NONE, "  → Reduced LR to: " + oOptim.lr)
        
        # Skip this batch
        return
    ok
    if nCurrentBatch % 5 = 0
        oStyl.green(:NONE, "  ✅ New best!: ")
        ? oStyl.yellow(:NONE, "" + nErr)
    ok

    # If Loss improved
    /*if nErr < nBestLoss
            nBestLoss = nErr
            nWorseCount = 0
           if bMontoring  
                oStyl.green(:NONE, "  ✅ New best!: ")
                ? oStyl.yellow(:NONE, "" + nErr)
           ok
        else
            nWorseCount++
            # Reduce LR slowly
            if nWorseCount >= 10   
                oOptim.lr = oOptim.lr * 0.8
                if bMontoring
                    oStyl.red(:NONE, "  ⚠️ No improvement for ")
                    oStyl.yellow(:NONE, "" + nWorseCount)
                    oStyl.red(:NONE, " batches | New LR: ")
                    ? oStyl.yellow(:NONE, "" + oOptim.lr)
                ok

                # Reset counter
                nWorseCount = 0
            ok
        ok
    */
        # ═══════════════════════════════════════════
        # EARLY STOPPING
        # ═══════════════════════════════════════════
        
        if nErr < 0.05
               oStyl.green(:NONE, "✅ Converged at iteration " )
             ? oStyl.yellow(:NONE,"" + nCurrentEpoch + "!") 
            oGUI.oApp.quit()
        ok
        
        if nWorseCount >= nPatienceMax
               oStyl.red(:NONE, "⚠️ Early stopping - no improvement for " )
               oStyl.yellow(:NONE,"" + nPatienceMax)
             ? oStyl.red(:NONE, " iterations") 
            oGUI.oApp.quit()
        ok
        
        # Stop if gradients are consistently exploding
         if nGradNorm > 20.0 
             ? oStyl.red(:NONE, "❌ Stopping - gradient explosion detected " + nGradNorm) 
            //oGUI.oApp.quit()
        ok 

        # Stats
        nEpochLoss += nErr
        nValidSamples++
        
        # Update GUI (Every N steps)
        if nCurrentBatch % nLogInterval = 0
            # Update GUI
            oGUI.update(nCurrentEpoch, nEpochs, nCurrentBatch, nTotalBatches, nErr, oOptim.lr)
           
        ok

    # ---------------------------------------------------------
    #  End of Epoch Logic
    # ---------------------------------------------------------
    func endOfEpochLogic
        
        # Calculate Avg Loss
        nAvgLoss = 0 
        if nValidSamples > 0 nAvgLoss = nEpochLoss / nValidSamples ok
        
        # Update GUI Final Stats
        oGUI.finishEpoch(nCurrentEpoch, nAvgLoss)
        
        # Scheduler
        if bUseScheduler applyScheduler() ok
        
        # Curriculum
        if bUseCurriculum updateCurriculum() ok
        
        # Generate Demo  
        if !isnull(oTok) /*and nBestLoss < 1*/ and nCurrentEpoch % 1 = 0 
            nSeqLen = oModel.nSeqLen
            cGen = generateDemoText(oModel, oTok, "Above all, be patient.", "<TO_AR>", nSeqLen)
            oGUI.setOutput("Epoch " + nCurrentEpoch + ": " + cGen)
            
            cGen = generateDemoText(oModel, oTok, "أهم شيء أن تكن صبوراً.", "<TO_EN>", nSeqLen)
            oGUI.setOutput("Epoch " + nCurrentEpoch + ": " + cGen)
        ok

        # Save
        if nCurrentEpoch % nSaveInterval = 0 saveCheckpoint() ok

    # ---------------------------------------------------------
    #  Slice Batch
    # --------------------------------------------------------- 
    func sliceBatch oIn, oTg
        # 1. Slice Input
        # Input is (1, MaxSeq). We just want the first nCurrSeqLen columns.
        # Since it's 1 row, columns act like elements.
        # We can use our new sliceRows if we treat it as (Seq, 1)? 
        # No, for Input (1, N) we need 'selectColumns' which we already have!
        
        # Use existing C function for columns:
        oNewInput = oIn.selectColumns(1, nCurrSeqLen)
        
        # 2. Slice Target (Optimization)
        # Target is (MaxSeq, Vocab). We want first nCurrSeqLen rows.
        # USE THE NEW C KERNEL
        
        oNewTarget = oTg.sliceRows(1, nCurrSeqLen)
        
        return [oNewInput, oNewTarget]

    # ---------------------------------------------------------
    #  Apply Learning Rate Schedule
    # ---------------------------------------------------------
	func applyScheduler

        nGlobalStep = (nCurrentEpoch - 1) * nTotalBatches
        
        if nGlobalStep < nWarmupSteps
            # Warmup: Linear
            current_lr = nBaseLR * (nGlobalStep / nWarmupSteps)
            
            # Don't exceed base
            if current_lr > nBaseLR
                current_lr = nBaseLR
            ok
        else
            # Decay: Faster
            decay_steps = nGlobalStep - nWarmupSteps
            current_lr = nBaseLR * pow(nDecayRate, (decay_steps / (nTotalBatches * 5)))
            
            # Cosine decay
            #progress = (nGlobalStep - nWarmupSteps) / (nTotalBatches - nWarmupSteps)
            #current_lr = nBaseLR * 0.5 * (1 + cos(pi * progress))
            
            # Floor
            if current_lr < nMinLR
                current_lr = nMinLR
            ok
        ok
        
        oOptim.lr = current_lr
        
        if bMontoring
            info("[Scheduler]: Step=" + nGlobalStep + 
                " | LR=" + current_lr)
        ok

    # ---------------------------------------------------------
    #  Update Curriculum
    # ---------------------------------------------------------
    func updateCurriculum
        nOldSeq = nCurrSeqLen
        if nCurrentEpoch <= 5 nCurrSeqLen = 8
        elseif nCurrentEpoch <= 10 nCurrSeqLen = 16
        else nCurrSeqLen = nMaxSeqLen ok
        
        if nCurrSeqLen != nOldSeq
            # Update Dataset & Model
            if isMethod(oTrainLoader.oDataset.oParent, "setSequenceLength")
                oTrainLoader.oDataset.oParent.setSequenceLength(nCurrSeqLen)
            ok
            if hasAttribute(oModel, "oTransformer")
                oModel.oTransformer.nSeqLen = nCurrSeqLen
            ok
            info("[Curriculum]: Level Up! Len -> " + pad(nCurrSeqLen, 5))
        ok

    # ---------------------------------------------------------
    #  Generate Demo Text
    # --------------------------------------------------------- 
    func generateDemoText oModel, oTok, cSrc, cTaskToken, nSeq
        
        # 1. Prepare the Prompt in the same order as the training.
        aIds = []
        
        # Add Task Token
        nTaskID = oTok.getTokenId(cTaskToken) 
        aIds + nTaskID
        
        # Add Start Token (as in Dataset)
        aIds + 4 
        
        # Add Source
        aSrcIds = oTok.encode(cSrc)
        for x in aSrcIds aIds + x next
        
        # Add Separator (here we stop and ask the model to complete)
        aIds + 3
        # Important Note: Don't add END (5) here! 
        
        cGenerated = ""
        
        # 2. Generate Loop
        for s = 1 to nSeq
            # Prepare the Tensor
            oInput = new Tensor(1, nSeq)
            for k = 1 to nSeq oInput.setVal(1, k, 1) next # Pad with 1
            
            # Fill the Context
            for k = 1 to len(aIds)
                if k > nSeq exit ok
                v = aIds[k] if v=0 v=2 ok
                oInput.setVal(1, k, v)
            next
            
            # Get the Prediction
            oLogits = oModel.forward(oInput)
            oLogits.softmax()
            
            # Get the Last Prediction
            nLastPos = len(aIds)
            if nLastPos > nSeq nLastPos = nSeq ok
            
            maxV = -1 maxId = 0
            for v = 1 to oLogits.nCols 
                val = oLogits.getVal(nLastPos, v) 
                if val > maxV maxV = val maxId = v ok
            next
            
            # Add the Expected Character to the History
            aIds + maxId
            
            # Stop Conditions
            if maxId = 1 exit ok # PAD
            if maxId = 5 exit ok # END (Stop when the model decides the end)
            if maxId = 3 exit ok # SEP (Additional Safety)
            
            cToken = oTok.getTokenFromId(maxId)
            cGenerated += cToken
        next
        
        ? " Gen (" + cTaskToken + "): " + cSrc + " -> " + cGenerated

        return "Gen (" + cTaskToken + "): " + cSrc + " -> " + cGenerated
		
	# ---------------------------------------------------------
    #  Save Checkpoint
    # ---------------------------------------------------------
	func saveCheckpoint
        if (nCurrentEpoch % nSaveInterval = 0) or (nCurrentEpoch = nEpochs)
            cPath = cCheckpointDir + cModelName + "_ep" + nCurrentEpoch + ".rdata"
            oModel.saveModel(cPath)
            info("[Checkpoint]: Saved to " + cPath)
           ok

	# ---------------------------------------------------------
    #  Info
    # ---------------------------------------------------------
	func info cMsg
        see cMsg + nl
        if !isnull(oGUI)
            # Send to GUI Log
            oGUI.log(cMsg)
        ok

    # ---------------------------------------------------------
	func GuiStartTimer
        # Helper to reset timer in GUI if needed or just track locally
        oGUI.tEpochStart = clock()