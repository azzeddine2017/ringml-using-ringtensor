# File: src/utils/visualizer.ring
# Description: Console Visualization for Training Process using RingConsoleColors
# Author: Azzeddine Remmal


class TrainingVisualizer
    nTotalEpochs
    nTotalBatches
    
    # Spinner animation frames
    aSpinner = ["|", "/", "-", "\"]
    nSpinIdx = 1

    func init nEpochs, nBatches
        nTotalEpochs  = nEpochs
        nTotalBatches = nBatches
        
        # Initial Clear/Header
        ? oStyl.cyan(:BOLD,"==========================================")
        ? oStyl.white(:BOLD," " + RingMLVersion() + " Training Dashboard         ")
        ? oStyl.cyan(:BOLD,"==========================================")

    func update nEpoch, nBatch, nLoss, nAcc
        # 1. Calculate Progress
        nPercent = (nBatch / nTotalBatches) * 100
        nBarLen  = 20
        nFilled  = floor((nPercent / 100) * nBarLen)
        
        # 2. Build Progress Bar String
        cBar = "["
        cBar += copy("=", nFilled)
        if nFilled < nBarLen cBar += ">" ok
        cBar += copy(" ", nBarLen - nFilled)
        cBar += "]"

        # 3. Determine Colors based on performance
        # Loss Color: Red (Bad) -> Yellow -> Green (Good < 1.0)
        nLossColor = :RED
        if nLoss < 2.0 nLossColor = :YELLOW ok
        if nLoss < 1.0 nLossColor = :GREEN ok

        # Accuracy Color: Red -> Yellow -> Green (> 50%)
        nAccColor = :RED
        if nAcc > 20 nAccColor = :YELLOW ok
        if nAcc > 50 nAccColor = :GREEN ok

        # 4. Draw the UI (Using \r to overwrite line)
        # Note: We print multiple components. 
        # Standard Ring `see` adds newlines, so we simulate a dashboard by clearing previous block 
        # or just printing a single dynamic line for the batch progress.
        
        # Move cursor to start of line (Carriage Return)
        see char(13) 
        
        # Print Spinner
        see aSpinner[nSpinIdx] + " "
        nSpinIdx++ 
        if nSpinIdx > 4 nSpinIdx = 1 ok

        # Print Epoch info
        oStyl.cyan(:NONE," Epoch " + nEpoch + "/" + nTotalEpochs + " ")
        
        # Print Progress Bar
        oStyl.white(:NONE,cBar + " " + floor(nPercent) )
        
        # Print Metrics
        oStyl.white(:NONE,"| Loss: ")
        oStyl.seeString(nLossColor,:NONE, "" + nLoss + " ")
        
        oStyl.white(:NONE,"| Accuracy: ")
        oStyl.seeString(nAccColor,:NONE, "" + nAcc + " %") 
        
        # No new line at the end to allow overwriting!

    func finishEpoch nEpoch, nAvgLoss, nValAcc
        # Loss Color: Red (Bad) -> Yellow -> Green (Good < 1.0)
        nLossColor = :RED
        if nAvgLoss < 2.0 nLossColor = :YELLOW ok
        if nAvgLoss < 1.0 nLossColor = :GREEN ok

        # Accuracy Color: Red -> Yellow -> Green (> 50%)
        nAccColor = :RED
        if nValAcc > 20 nAccColor = :YELLOW ok
        if nValAcc > 50 nAccColor = :GREEN ok

        # When epoch is done, we print a permanent summary line
        see char(13) + copy(" ", 80) + char(13) # Clear line
        
        oStyl.green(:NONE,"[DONE]")
        oStyl.yellow(:NONE," Epoch " + nEpoch + ": ")
        oStyl.white(:NONE,"Avg Loss = ")

        oStyl.seeString(nLossColor,:NONE, "" + nAvgLoss)

        oStyl.white(:NONE," | ")
        oStyl.white(:NONE,"Val Accuracy = ")

        ? oStyl.seeString(nAccColor,:NONE, "" + nValAcc + " %") 
        # Print a separator or Brain animation frame if desired
    