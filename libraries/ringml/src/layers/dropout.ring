# File: src/layers/dropout.ring
# Description: Dropout Layer to prevent overfitting
# Author: Azzeddine Remmal



class Dropout from Layer
    nDropRate = 0.5
    oMask     = NULL # Cache the mask for backward pass

    func init nRate
        nDropRate = nRate

    func forward oInputTensor
        # If evaluating (Testing), pass input as is
        if !bTraining
            return oInputTensor
        ok

        # 1. Create a Mask Tensor of same shape
        oMask = new Tensor(oInputTensor.nRows, oInputTensor.nCols)
        
        # 2. Fill Mask with Dropout Logic (0 or Scale)
        # We reuse the logic we added to Tensor, but apply it to the mask
        # Ideally, we init mask with dummy data then apply dropout logic
        # Optimization: We can do it in one pass inside Tensor.apply_dropout
        # But wait, apply_dropout modifies IN-PLACE.
        # So we treat oMask as the filter.
        
        # Initialize Mask with 1s (dummy) then overwrite in apply_dropout logic
        # Actually, let's just make apply_dropout generate the mask values directly 
        # on whatever data is there, or better:
        
        # Correct approach:
        # We want: Output = Input * Mask
        # Mask needs to be generated.
        
        # Let's use a dedicated generation method on the mask tensor
        oMask.applyDropout(nDropRate) # This fills oMask with 0s and Scales
        
        # 3. Apply Mask to Input (Element-wise Mul)
        # We copy input first to avoid modifying original data reference
        oOutput = oInputTensor.copy()
        oOutput.mul(oMask)
        
        return oOutput

    func backward oGradOutput
        # Backward: Gradient * Mask
        # If unit was dropped (0), gradient is 0.
        # If unit was kept (Scale), gradient is scaled.
        
        if !bTraining
            return oGradOutput
        ok
        
        oGradOutput.mul(oMask)
        return oGradOutput