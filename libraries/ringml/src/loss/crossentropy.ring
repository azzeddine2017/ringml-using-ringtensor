# File: src/loss/crossentropy.ring
# Description: Cross Entropy Loss
# Author: Azzeddine Remmal



class CrossEntropyLoss

 
    # Cache
    oProbs        
    oTargetCache  

    func calculate oLogits, oTarget
               
        # 1. Softmax (C-Kernel Fast)
        # We clone logits to avoid modifying model output in-place
        oProbs = oLogits.copy() 
        oProbs.softmax()
        
        oTargetCache = oTarget
        
        if oLogits.bGraphMode
            return oLogits.returnGraphNode(OP_CROSSENTROPY, oLogits, oTarget, 1, 1)
        ok
        
        # 2. Loss Calculation (NEW C-KERNEL)
        # Returns number directly, handles masking internally
        nLoss = tensor_crossentropy_loss(oProbs.pData, oTarget.pData)
        
        return nLoss
        
    # Legacy wrappers
    func forward p1, p2 
        return calculate(p1, p2)


    func backwardTensor
        # Prepare Gradient Tensor
        oGrad = oProbs.copy()
        
        # 3. Gradient Calculation (NEW C-KERNEL)
        # Calculates (P - T) / N and handles Masking (Zeroing padding rows)
        tensor_crossentropy_backward(oProbs.pData, oTargetCache.pData, oGrad.pData)
        
        return oGrad

    func backward oPred, oTarget
        # Standard implementation for old examples (e.g. XOR)
        oGrad = oPred.copy()
        oGrad.sub(oTarget)
        
        nTotal = oPred.nRows 
        if nTotal > 0
            oGrad.scalarMul(1.0 / nTotal)
        ok
        
        return oGrad

    
