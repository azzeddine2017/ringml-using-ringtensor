/*
    Project: Jabr
    File: src/utils/SmartScheduler.ring
    Description: Real-Time Adaptive Controller (EMA Based).
                 Reacts to trends per batch, not just per epoch.
*/

class SmartScheduler

    oOptimizer
    
    # Config
    nFactor     = 0.5		# 50%
    nPatience   = 50       #We'll wait 50 patches (instead of Epochs)
    nMinLR      = 0.00001
    bControlWD  = true
    
    # Smoothing Factor (0.95 means trust history 95%, new val 5%)
    # This filters out the noise of individual batches
    nAlpha      = 0.90 
    
    # State
    nSmoothedLoss = 0.0     # Smoothed loss
    nBestLoss     = 10000.0
    nBadSteps     = 0
    nCooldown     = 0

    func init oOptim, nBatchPatience, nReduction
        oOptimizer = oOptim
        nPatience  = nBatchPatience
        nFactor    = nReduction
        nSmoothedLoss = 0.0

    # This is called inside the batch loop (after calculate loss)
    func stepbatch nRealLoss
        
        # 1. Initialize EMA on first step
        if nSmoothedLoss = 0.0 
            nSmoothedLoss = nRealLoss 
        ok
        
        # 2. Calculate Smoothed Loss (EMA)
        # L_smooth = (alpha * L_old) + ((1-alpha) * L_new)
        nSmoothedLoss = (nAlpha * nSmoothedLoss) + ((1.0 - nAlpha) * nRealLoss)
        
        # 3. Cooldown
        if nCooldown > 0
            nCooldown--
            return
        ok
        
        # 4. Check Trend (Using Smoothed Value)
        # We use a very small threshold for improvement
        if nSmoothedLoss < (nBestLoss - 0.0001)
            nBestLoss = nSmoothedLoss
            nBadSteps = 0
        else
            nBadSteps++
            
            # If things have been worsening for nPatience patches in a row
            if nBadSteps >= nPatience
                
                # --- INTERVENTION ---
                reduceParameters()
                
                # Reset
                nBadSteps = 0
                nCooldown = nPatience # Give it time to adjust to the new learning rate
                
                # Reset Best Loss to current to avoid immediate trigger again
                nBestLoss = nSmoothedLoss 
            ok
        ok
        
        # 5. Emergency Brake (Spike Detection)
        # If the real loss suddenly spikes by 3 times the average, this is a spike
        if nRealLoss > (nSmoothedLoss * 3.0)
             see ">>> ⚠️ EMERGENCY: Gradient Spike detected! (" + nRealLoss + ")" + nl
             # Reduce the learning rate immediately without waiting for patience
             reduceParameters()
             nSmoothedLoss = nRealLoss # Reset EMA to accept new reality
             nCooldown = 50
        ok

    func reduceParameters
        oldLR = oOptimizer.lr
        newLR = oldLR * nFactor
        
        if newLR < nMinLR newLR = nMinLR ok
        
        # If we've reached the minimum, no need to print every time
        if oldLR = nMinLR return ok
        
        oOptimizer.lr = newLR
        
        cMsg = ">>> 📉 Auto-Pilot: Reducing LR to " + newLR
        
        if bControlWD
            oldWD = oOptimizer.weightDecay
            newWD = oldWD * nFactor
            oOptimizer.weightDecay = newWD
            cMsg += " | WD to " + newWD
        ok
        
        see nl + cMsg + " (at Loss: " + nSmoothedLoss + ")" + nl

    func getMetrics
        return [:avg_loss = nSmoothedLoss, :lr = oOptimizer.lr]