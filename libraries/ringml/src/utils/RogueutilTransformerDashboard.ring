# File: src/utils/RogueutilTransformerDashboard.ring
# Description: High-End Interactive Dashboard using RogueUtil
# Author: Azzeddine Remmal

load "rogueutil.ring"

class RogueutilTransformerDashboard

    nEpochs
    nBatches
    nEpoch
    nBatch
    nLossSum 
    nStepCount 
    nTotalTime 
    nEpochTime 

    oModel
    oTransformerBlock

    # --- Layout Config ---
    ROW_HEADER    = 3
    ROW_EPOCH_BAR = 4
    ROW_BATCH_BAR = 6
    ROW_STATS     = 8
    ROW_TIMING    = 10
    ROW_MODEL_SUMMARY  = 12
    ROW_HEALTH    = 13

    # --- Graph Config ---
    GRAPH_X       = 71
    GRAPH_Y       = 6     
    GRAPH_H       = 11    
    GRAPH_W       = 45    
    
    # --- Output Config ---
    OUTPUT_TXT_X  = 3
    OUTPUT_TXT_Y  = 21
    N_LAST_OUTPUTTXT = 10 
    MAX_LEN_OUTPUTTXT = 55

    # --- Log Config ---
    LOG_TXT_X  = 62
    LOG_TXT_Y  = 21
    N_LAST_LOGTXT = 10
    MAX_LEN_LOGTXT = 55
    

    # --- State --
    tStart
    tLastBatch
    tLastEpoch
    

    # History Buffer
    aLossHistory = [] 
    aLogHistory = []
    aOutputHistory = []
    
    
    func init 
        setConsoleTitle("AI Training Dashboard")
        hideCursor()
        cls()
        # Draw Header
        setColor(CYAN)
        printXY(4, ROW_HEADER, "    TRANSFORMER TRAINING DASHBOARD v1.0")
        setColor(WHITE)
        printXY(2, ROW_EPOCH_BAR, "Epoch: ")
        printXY(2, ROW_BATCH_BAR, "Batch: ")
        printXY(2, ROW_STATS,  "Loss:            Avg:          LR:")
        printXY(2, ROW_TIMING, "Time:            ETA:          Speed:")
        printXY(2, ROW_MODEL_SUMMARY, "Model Summary")
        setColor(YELLOW)
        printXY(2, ROW_HEALTH, "Vitals: ")
        resetColor()
        drawGraphFrame()

    func start oModelRef, nTotalEpochs, nTotalBatches
        oModel = oModelRef
        nEpochs  = nTotalEpochs
        nBatches = nTotalBatches
        oTransformerBlock = oModel.oTransformer
        modelSummary()

    # Start Timer
    func startTimer
       tStart = clock()
       tLastBatch = tStart
       tLastEpoch = tStart
       nLossSum = 0
       nStepCount = 0

    # Update Dashboard
    func update nEpoch, nBatch, nLoss, nLastLoss, nLR

        this.nEpoch = nEpoch
        this.nBatch = nBatch
        
        # --- Calc Stats ---
        nNow = clock()
        nBatchTime = (nNow - tLastBatch) / clockspersecond()
        tLastBatch = nNow
        
        nLossSum += nLoss
        nStepCount++
        nAvgLoss = nLossSum / nStepCount
        
        nBatchesLeft = (nBatches - nBatch) + ((nEpochs - nEpoch) * nBatches)
        nTimeLeft = nBatchesLeft * nBatchTime
        nSpeed = 0 if nBatchTime > 0 nSpeed = 1 / nBatchTime ok

        # --- Draw Bars & Stats --- 
        drawBar(18, ROW_EPOCH_BAR+1, 35, nEpoch, nEpochs, BLUE)
        drawBar(18, ROW_BATCH_BAR+1, 35, nBatch, nBatches, GREEN)
        
        setColor(CYAN)
        printXY(9, ROW_EPOCH_BAR, "" + nEpoch + " / " + nEpochs)
        printXY(9, ROW_BATCH_BAR, "" + nBatch + " / " + nBatches)

        nColor = WHITE
        cArrow = " "
        if nLoss < nLastLoss nColor = GREEN cArrow = " ↓" ok
        if nLoss > nLastLoss nColor = RED   cArrow = " ↑" ok
        
        setColor(nColor)
        printXY(8, ROW_STATS, pad(nLoss, 8) + cArrow)

        setColor(WHITE)
        printXY(24, ROW_STATS, pad(nAvgLoss, 8))
        setColor(CYAN)
        clear(37, ROW_STATS, 7)
        printXY(37, ROW_STATS, pad(nLR, 7))
        nTimeElapsed = (clock()-tStart)/clockspersecond()
        setColor(MAGENTA)
        printXY(8, ROW_TIMING, formatTime(nTimeElapsed))
        printXY(24, ROW_TIMING, formatTime(nTimeLeft))
        printXY(40, ROW_TIMING, pad(nSpeed, 4) + " it/s")
        resetColor()

        # --- UPDATE GRAPH ---
        updateGraph(nLoss)
        # --- LOG HEALTH ---
        logHealth()

    # Draw Graph Frame
    func drawGraphFrame
        # Draw axes
        setColor(GREY)
        # Y-Axis line
        for r = 0 to GRAPH_H 
            printXY(GRAPH_X - 1, GRAPH_Y + r, "|") 
        next
        # X-Axis line
        printXY(GRAPH_X, GRAPH_Y + GRAPH_H + 1, copy("-", GRAPH_W))
        # Title
        printXY(GRAPH_X, GRAPH_Y - 2, "Loss History (Last " + GRAPH_W + ")")

        drawTable()
        
    # Update Graph
    func updateGraph nNewLoss
        # 1. Add to history
        aLossHistory + nNewLoss
        
        # Keep only last GRAPH_W items (Scrolling)
        while len(aLossHistory) > GRAPH_W
            del(aLossHistory, 1)
        end
        
        # 2. Clear Graph Area
        for r = 1 to GRAPH_H
            printXY(GRAPH_X, GRAPH_Y + r, copy(" ", GRAPH_W))
        next
        
        # 3. Find Range for Auto-Scaling
        nMaxVal = -1000
        nMinVal = 1000
        
        for val in aLossHistory
            if val > nMaxVal nMaxVal = val ok
            if val < nMinVal nMinVal = val ok
        next
        
        # Avoid division by zero
        if nMaxVal = nMinVal nMaxVal = nMinVal + 0.0001 ok
        nRange = nMaxVal - nMinVal

        nColor = WHITE
        nItems = len(aLossHistory)
        # 4. Plot Points
        for i = 1 to nItems
            val = aLossHistory[i]
            
            # Normalize value to Height (0 to GRAPH_H)
            # Relative position: (val - min) / range
            nNormalized = (val - nMinVal) / nRange
            nRowOffset  = floor(nNormalized * GRAPH_H)
            
            # Screen Row (Invert because screen Y grows down)
            nScreenRow = (GRAPH_Y + GRAPH_H) - nRowOffset
            nScreenCol = GRAPH_X + i - 1
            if i > 1
                priviousVal = aLossHistory[i-1]
                if val < priviousVal nColor = GREEN ok
                if val > priviousVal nColor = RED   ok
                if val = priviousVal nColor = WHITE ok
            end

            setColor(nColor)
            # Use different chars for variation
            cChar = "*"
            if i = nItems cChar = "━" ok # Head
            # if nRowOffset = 0 cChar = "━" ok
            if nRowOffset = GRAPH_H cChar = "-" ok
            
            printXY(nScreenCol, nScreenRow, cChar)
        next
        
        # 5. Print Min/Max Labels
        setColor(RED)
        printXY(GRAPH_X + GRAPH_W - 53, GRAPH_Y, "" + round(nMaxVal, 6))
        setColor(GREEN)
        printXY(GRAPH_X + GRAPH_W - 53, GRAPH_Y + GRAPH_H, "" + round(nMinVal, 6))

    # Simple rounding display helper
    func round nNum, nDec
        return left("" + nNum, nDec)

    # Log Health of Transformer
    func logHealth 
        oAttnGrad = oTransformerBlock.oAttention.oW_Q.oGradWeights
        drawHealthTag(10, ROW_HEALTH, "ATTN", checkGrad(oAttnGrad))
        oNormGrad = oTransformerBlock.oNorm1.g_beta
        drawHealthTag(25, ROW_HEALTH, "NORM", checkGrad(oNormGrad))
        oFFNGrad = oTransformerBlock.oFFN_1.oGradWeights
        drawHealthTag(40, ROW_HEALTH, "FFN", checkGrad(oFFNGrad))
        resetColor()

    func modelSummary
        # draw header
        setColor(WHITE)
        printXY(2,  ROW_MODEL_SUMMARY  + 3, "Atten Heads:")
        printXY(20, ROW_MODEL_SUMMARY  + 3, "Atten Para:")
        printXY(2,  ROW_MODEL_SUMMARY  + 5, "Norm1 Para:")
        printXY(20, ROW_MODEL_SUMMARY  + 5, "FFN Para:")
        printXY(40, ROW_MODEL_SUMMARY  + 5, "Norm2 Para:")
        printXY(2,  ROW_MODEL_SUMMARY  + 7, "Total Para:")
        printXY(20, ROW_MODEL_SUMMARY  + 7, "Trainable:")
        printXY(40, ROW_MODEL_SUMMARY  + 7, "Frozen:")
        resetColor()
        # draw values
        setColor(CYAN)
        printXY(15, ROW_MODEL_SUMMARY + 3, "" + oTransformerBlock.getParams()[:Heads])
        printXY(31, ROW_MODEL_SUMMARY + 3, "" + oTransformerBlock.getParams()[:AttnParams])
        printXY(14, ROW_MODEL_SUMMARY + 5, "" + oTransformerBlock.getParams()[:Norm1Params])
        printXY(30, ROW_MODEL_SUMMARY + 5, "" + oTransformerBlock.getParams()[:FFNParams])     
        printXY(52, ROW_MODEL_SUMMARY + 5, "" + oTransformerBlock.getParams()[:Norm2Params])
        printXY(14, ROW_MODEL_SUMMARY + 7, "" + oTransformerBlock.getParams()[:TotalParams])
        printXY(31, ROW_MODEL_SUMMARY + 7, "" + oTransformerBlock.getParams()[:TrainableParams])
        printXY(55, ROW_MODEL_SUMMARY + 7, "" + oTransformerBlock.getParams()[:Frozen])
        resetColor()

    # Check Gradient
    func checkGrad oTensor
        nLimit = 10
        adata = oTensor.toList()
        if len(adata) < 10 nLimit = len(adata) ok
        for i=1 to nLimit if adata[i] != 0 return :ALIVE ok next
        nMid = floor(len(adata) / 2)
        if adata[nMid] != 0 return :ALIVE ok
        return :DEAD

    # Draw Health Tag
    func drawHealthTag x, y, cText, nStatus
        if nStatus = :ALIVE setColor(GREEN) cInd="[OK]" else setColor(RED) cInd="[!!]" ok
        printXY(x, y, cText + ":" + cInd)

    # Finish Epoch
    func finishEpoch
        nNow = clock()
        nEpochTime = (nNow - tLastEpoch) / clockspersecond()
        tLastEpoch = nNow
        setColor(WHITE)
        printXY(2, ROW_TIMING + 1, "Epochs Time:")
        setColor(MAGENTA)
        printXY(13, ROW_TIMING + 1, formatTime(nEpochTime))
        resetColor()
        setColor(GREEN)
        printXY(57, ROW_EPOCH_BAR, "DONE")
        # stay 3 seconds and clear
        sleep(2) 
        clear(57, ROW_EPOCH_BAR, 4)

        

    # Finish Training
    func finishTraining
        # calculate total time
        nTotalTime = nEpochTime

        clear(2, ROW_TIMING, 15)
        setColor(WHITE)
        printXY(2, ROW_TIMING + 1, "Total Time:")

        setColor(MAGENTA)
        printXY(13, ROW_TIMING + 1, formatTime(nTotalTime))
        resetColor()

        locate(LOG_TXT_X + 3, LOG_TXT_Y + 1)
        showCursor()
        setColor(GREEN)
        ?  "Training Finished Successfully."
        resetColor()

    # Draw Rect
    func drawRect x, y, w, h
        for r=y to y+h
            printXY(x, r, "┃")
            printXY(x+w, r, "┃")
        next
        printXY(x, y, copy("━", w))
        printXY(x, y+h, copy("━", w))

    # Draw Bar
    func drawBar x, y, nLen, nVal, nMax, nColor
        nPercent = nVal / nMax
        nFilled  = floor(nPercent * nLen)
        # Draw Bar
        setColor(WHITE)
        printXY(x-1, y, "[")
        printXY(x+nLen, y, "]")
        # Fill Bar
        setColor(nColor)
        cFill = copy("=", nFilled)
        if nFilled < nLen cFill += ">" ok
        printXY(x, y, cFill + copy(" ", nLen - nFilled - 1))
        setColor(WHITE)
        printXY(x + nLen + 2, y, "" + floor(nPercent * 100) + "% ")
        resetColor()
        
    # Draw Table
    func drawTable
        setColor(CYAN)
        drawRect(1, 1, 120, 40)
        # Header Divider
        printXY(1, 3, copy("━", 120))
        printXY(3, 2, "TRANSFORMER AI TRAINER v1.0")
        
        # Vertical Divider (Split Stats/Logs from Graph/Controls)
        for r = 4 to 40 printXY(60, r, "┃") next
        
        # Horizontal Divider (Split Top/Bottom)
        printXY(1, 20, copy("━", 60)) # Left split
        printXY(60, 20, copy("━", 60)) # Right split
        resetColor()
        
        # Titles
        setColor(YELLOW)
        printXY(20, 3, " TRAINING STATS ")
        printXY(75, 3, " LOSS HISTORY ")
        printXY(20, 20, " OUTPUT ")
        printXY(75, 20, " SYSTEM LOG ")
        resetColor()

    # Log System Messages
    func log cLog
        # Add new message
        aLogHistory + cLog
        
        # Keep only last N_LAST_LOGTXT
        while len(aLogHistory) > N_LAST_LOGTXT del(aLogHistory, 1) end
        
        # Clear Area (N_LAST_LOGTXT lines)
        for r = 1 to N_LAST_LOGTXT
            clear(LOG_TXT_X, LOG_TXT_Y + r, MAX_LEN_LOGTXT)
        next
        
        # Redraw
        for i = 1 to len(aLogHistory)
            cLog = aLogHistory[i]
            if len(cLog) > MAX_LEN_LOGTXT
                cLog = left(cLog, MAX_LEN_LOGTXT) + "..."
            ok
            setColor(LIGHTCYAN)
            printXY(LOG_TXT_X, LOG_TXT_Y + i - 1, cLog)
        next
        resetColor()
    
    # Output Model's Generated Text
    func output cOut
        # Add new Output
        cOutput = "" + this.nEpoch + ": " + cOut
        aOutputHistory + cOutput
        
        # Keep only last N_LAST_OUTPUTTXT
        while len(aOutputHistory) > N_LAST_OUTPUTTXT del(aOutputHistory, 1) end
        
        # Clear Area (N_LAST_OUTPUTTXT lines)
        for r = 1 to N_LAST_OUTPUTTXT
            clear(OUTPUT_TXT_X, OUTPUT_TXT_Y + r, MAX_LEN_OUTPUTTXT)
        next
        
        # Redraw
        for i = 1 to len(aOutputHistory)
            cOutput = aOutputHistory[i]
            if len(cOutput) > MAX_LEN_OUTPUTTXT
                cOutput = left(cOutput, MAX_LEN_OUTPUTTXT) + "..."
            ok
            setColor(CYAN)
            printXY(OUTPUT_TXT_X, OUTPUT_TXT_Y + i - 1, cOutput)
            
        next
        resetColor()

    # Clear Area
    func clear nX, nY, nLen
        printXY(nX,nY,copy(" ",nLen))
    
    