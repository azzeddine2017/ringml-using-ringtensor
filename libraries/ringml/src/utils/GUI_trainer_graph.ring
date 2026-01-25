# Project: RingML
# File: src/utils/GUI_trainer_graph.ring
# Description: Advanced Trainer using Auto-Graph Engine for Maximum Performance
# Author: Azzeddine Remmal
oTrainer = null
class GUI_Trainer_Graph

    # Components
    oModel 
	oTrainLoader 
	oOptim 
	oLoss 
	oTok
    
    # Config
    nEpochs         
    nLogInterval   
    nSaveInterval  
    nSeqLen
    cCheckpointDir  = "model/"
    cModelName      = "model"
    cObjectName
    bMonitoring      = false

    # Scheduler
    nBaseLR = 0.0001
    nMinLR = 0.00001
    nWarmupSteps = 50 
    nDecayRate = 0.98
    nGlobalStep = 0
    current_lr = 0

    # State Machine
    nCurrentEpoch = 1
    nCurrentBatch = 1
    nTotalBatches = 0
    
    # Stats
    nEpochLoss = 0
    nValidSamples = 0
    
    # Graph State
    oLossNode     = null
    oInputTensor  = null
    oTargetTensor = null
    # GUI
    oGUI
    oTimer

    func init oModelRef, oTokenizer, oLoader, oOptimizer, oLossFn, cObjectN
        oModel = oModelRef
        oTok   = oTokenizer
        oTrainLoader = oLoader
        oOptim = oOptimizer
        oLoss = oLossFn
        cObjectName = cObjectN
        
        nTotalBatches = oTrainLoader.nBatches

        # Checkpoints
        if !dirExists(cCheckpointDir) makeDir(cCheckpointDir) ok

        # GUI
        oGUI = new GUI_Dashboard() 
        
        return self

    func fit
        ? oStyl.green(:BOLD, nl + ">>> Starting Graph-Accelerated Training Engine <<<")
        
        # --- التحقيق في القيم ---
        /*see "DEBUG: Passed TrainLoader Batch Size: " + oTrainLoader.nBatchSize + nl
        see "DEBUG: Configured SeqLen: " + nSeqLen + nl
        
        nBatchSize = oTrainLoader.nBatchSize
        
        # إذا كانت 256، سنعرف المصدر الآن
        if nBatchSize = 256
             see "!!! CAUGHT IT: Batch Size is 256 inside Trainer !!!" + nl
             # الحل المؤقت القسري:
             # nBatchSize = 1 
        ok
        
        nTotalSize = nBatchSize * nSeqLen
        see "DEBUG: Target Tensor Allocated Size: " + nTotalSize + nl */
        # ----------------------
        
        # 1. Prepare Tensors for Graph
        # We need fixed-size tensors that we update in each step
        nBatchSize = oTrainLoader.nBatchSize
        oInputTensor = new Tensor(nBatchSize, nSeqLen)
        oTargetTensor = new Tensor(nBatchSize * nSeqLen, oModel.nVocabSize)
        
        # 2. Compile Graph
        info("[Graph] Compiling Model...")
        oLossNode = oModel.compile(oInputTensor, oTargetTensor, oLoss)
        info("[Graph] Compiled Successfully. Loss Node ID: " + oLossNode.nGraphNodeID)
        
        # 3. Set Optimizer in Graph
        graph_set_optimizer(1) # Adam

       /* see "Checking Weight Initialization..." + nl
        tensor_print_stats(oModel.oTokenEmbed.weights.pData)
        tensor_print_stats(oModel.aBlocks[1].oFFN_1.oWeights.pData)*/
        
        oTimer = new qTimer(oGUI.oWin) 
        oTimer.setInterval(1) 
        oTimer.setTimeoutEvent(cObjectName + ".trainingStep()") 
        oTimer.start()
        
        oGUI.tEpochStart = clock()
        oGUI.oApp.exec()

    func trainingStep
        if nCurrentEpoch > nEpochs
            oTimer.stop()
            info("Training Finished.")
            oGUI.finishTraining()
            return
        ok
        
        processBatch()
        
        nCurrentBatch++
        if nCurrentBatch > nTotalBatches
            endOfEpochLogic()
            nCurrentEpoch++
            nCurrentBatch = 1
            nEpochLoss = 0
            nValidSamples = 0
            oGUI.tEpochStart = clock()
        ok

    func processBatch
        nGlobalStep++
        
        # 1. Update LR
        updateLR()
        
        # 2. Get Data
        aBatch = oTrainLoader.getBatch(nCurrentBatch)
        if !isList(aBatch) return ok
        
        # Unpack
        if isList(aBatch[1]) 
             oIn = aBatch[1][1] 
             oTg = aBatch[1][2]
        else
             oIn = aBatch[1]
             oTg = aBatch[2]
        ok
        
        # 3. Update Graph Input/Target Memory
        # Instead of creating new tensors, we copy data to the bound memory
        oInputTensor.copyData(oIn)
        # --- فحص هل نجح النسخ؟ ---
        # اقرأ قيمة من المصدر وقيمة من الهدف
        /*nSrcVal = tensor_get(oIn.pData, 1, 1)
        nDstVal = tensor_get(oInputTensor.pData, 1, 1)
        
        if nSrcVal != nDstVal
             see "CRITICAL ERROR: Copy Failed! Src=" + nSrcVal + " Dst=" + nDstVal + nl
             raise("Stop")
        ok*/
        # -------------------------
        oTargetTensor.copyData(oTg)

        # ----------------------------------------------------
        # !!! تشخيص الطوارئ (Emergency Check) !!!
        # سنقرأ التنسور "الثابت" الذي سيستخدمه الجراف الآن
        # ----------------------------------------------------
        
        # فحص الهدف (Target): هل يحتوي على أي "1.0"؟
        /*nSumTarget = oTargetTensor.sumsquares() # دالة سريعة تجمع القيم
        
        if nSumTarget = 0
            see nl + ">>> CRITICAL: Graph Target Tensor is EMPTY (All Zeros)!" + nl
            see "    This explains the Zero Gradients." + nl
            see "    Source Tensor Sum: " + oTg.sumsquares() + nl
            raise("Data Copy Failed")
        ok*/
        
        # ----------------------------------------------------
        

        # 4. RUN GRAPH (Forward + Backward + CLIP + Update)
        # Parameters: (Epochs, LR, MaxNorm)
        graph_run(1, current_lr, 5.0)
        
        # 5. Get Loss from Graph
        if isnull(oLossNode) return ok
        pLossData = graph_get_output(oLossNode.nGraphNodeID)
        if isnull(pLossData) return ok
        nErr = tensor_get(pLossData, 1, 1)
        
        # Stats
        nEpochLoss += nErr
        nValidSamples++
        
        # Update GUI
        if nCurrentBatch % nLogInterval = 0
            oGUI.update(nCurrentEpoch, nEpochs, nCurrentBatch, nTotalBatches, nErr, current_lr)
        ok

    func updateLR
        if nGlobalStep <= nWarmupSteps
            current_lr = nBaseLR * (nGlobalStep / nWarmupSteps)
        else
            current_lr = nBaseLR * pow(nDecayRate, (nGlobalStep - nWarmupSteps) / nTotalBatches)
        ok
        if current_lr < nMinLR current_lr = nMinLR ok

    func endOfEpochLogic
        nAvgLoss = 0 
        if nValidSamples > 0 nAvgLoss = nEpochLoss / nValidSamples ok
        oGUI.finishEpoch(nCurrentEpoch, nAvgLoss)
        
        # Demo
        if !isnull(oTok) and nCurrentEpoch % 1 = 0 
            # Temporarily disable graph mode for inference
            oModel.setMode(false)

            cGen = generateDemoText(oModel, oTok, "Above all, be patient.", "<TO_AR>", nSeqLen)
            oGUI.setOutput("Epoch " + nCurrentEpoch + ": " + cGen)
            
            # Re-enable graph mode
           oModel.setMode(true)
        ok
        
        if nCurrentEpoch % nSaveInterval = 0 saveCheckpoint() ok

    /* func generateDemoText oModel, oTok, cSrc, cTaskToken, nSeq
        
        # --- DEBUG INIT ---
        see nl + ">>> [DEBUG] Entering generateDemoText <<<" + nl
        see "    Prompt: " + cSrc + nl
        
        # 1. Prepare the Prompt
        try
            aIds = []
            nTaskID = oTok.getTokenId(cTaskToken) 
            aIds + nTaskID
            aIds + 4 # START
            aSrcIds = oTok.encode(cSrc)
            for x in aSrcIds aIds + x next
            aIds + 3 # SEP
        catch
            see "!!! [CRASH] Error in Tokenization step: " + cCatchError + nl
            return "ERROR"
        done
        
        cGenerated = ""
        
        # 2. Generate Loop
        for s = 1 to nSeq
            see "    [DEBUG] Step " + s + " | Context Len: " + len(aIds) + " ... "
            
            # A. Tensor Creation
            try
                oInput = new Tensor(1, nSeq)
                oInput.fill(1) # PAD
                
                for k = 1 to len(aIds)
                    if k > nSeq exit ok
                    v = aIds[k] if v=0 v=2 ok
                    oInput.setVal(1, k, v)
                next
            catch
                see nl + "!!! [CRASH] Error creating/filling input tensor" + nl
                return "ERROR"
            done
            
            # B. CRITICAL CHECK: Mode Mismatch
            # يجب أن يكون الإدخال Eager (GraphMode=false)
            # ويجب أن تكون أوزان المودل Eager أيضاً
            if oInput.bGraphMode 
                see nl + "!!! [CRASH WARNING] Input Tensor is in Graph Mode!" + nl
            ok
            
            # فحص عينة من أوزان المودل (مثلاً Embedding)
            if oModel.oTokenEmbed.weights.bGraphMode
                see nl + "!!! [CRASH DETECTED] Model Weights are still in Graph Mode!" + nl
                see "    You cannot run Inference (Eager) on Graph Weights." + nl
                see "    Solution: Use createInferenceCopy() or setMode(false)." + nl
                return "CRASH_AVOIDED"
            ok
            
            # C. Forward Pass (The most likely crash point)
            see "Forward... "
            try
                oLogits = oModel.forward(oInput)
            catch
                see nl + "!!! [CRASH] Error inside oModel.forward(): " + cCatchError + nl
                return "ERROR"
            done
            
            if isnull(oLogits)
                see nl + "!!! [CRASH] oLogits returned NULL" + nl
                return "ERROR"
            ok
            
            # D. Softmax & Argmax
            see "Softmax... "
            oLogits.softmax()
            
            nLastPos = len(aIds)
            if nLastPos > nSeq nLastPos = nSeq ok
            
            maxV = -100000.0 maxId = 0
            
            # Argmax Loop check
            try
                for v = 1 to oLogits.nCols 
                    val = oLogits.getVal(nLastPos, v) 
                    if val > maxV maxV = val maxId = v ok
                next
            catch
                see nl + "!!! [CRASH] Error in Argmax loop (Access violation?)" + nl
                return "ERROR"
            done
            
            see "Got ID: " + maxId + nl
            
            aIds + maxId
            if maxId = 1 or maxId = 5 or maxId = 3 exit ok
            
            cToken = oTok.getTokenFromId(maxId)
            cGenerated += cToken
        next
        
        see ">>> [DEBUG] Generation Finished Successfully." + nl
        return "Gen (" + cTaskToken + "): " + cSrc + " -> " + cGenerated
 */

     func generateDemoText oModel, oTok, cSrc, cTaskToken, nSeq
        # 1. Prepare the Prompt
        aIds = []
        nTaskID = oTok.getTokenId(cTaskToken) 
        aIds + nTaskID
        aIds + 4 # START
        aSrcIds = oTok.encode(cSrc)
        for x in aSrcIds aIds + x next
        aIds + 3 # SEP
        
        cGenerated = ""
        
        # 2. Generate Loop
        for s = 1 to nSeq
            oInput = new Tensor(1, nSeq)
            oInput.fill(1) # PAD
            for k = 1 to len(aIds)
                if k > nSeq exit ok
                v = aIds[k] if v=0 v=2 ok
                oInput.setVal(1, k, v)
            next
            
            oLogits = oModel.forward(oInput)
            oLogits.softmax()
            
            nLastPos = len(aIds)
            if nLastPos > nSeq nLastPos = nSeq ok
            
            maxV = -1 maxId = 0
            for v = 1 to oLogits.nCols 
                val = oLogits.getVal(nLastPos, v) 
                if val > maxV maxV = val maxId = v ok
            next
            
            aIds + maxId
            if maxId = 1 or maxId = 5 or maxId = 3 exit ok
            
            cToken = oTok.getTokenFromId(maxId)
            cGenerated += cToken
        next
        
        return "Gen (" + cTaskToken + "): " + cSrc + " -> " + cGenerated 

    func saveCheckpoint
        cPath = cCheckpointDir + cModelName + "_ep" + nCurrentEpoch + ".rdata"
        oModel.saveModel(cPath)
        info("[Checkpoint] Saved: " + cPath)

    func info cMsg
        see cMsg + nl
        if !isnull(oGUI) oGUI.log(cMsg) ok
