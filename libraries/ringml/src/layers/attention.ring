# File: src/layers/attention.ring
# Description: Self-Attention Mechanism using Fused C-Kernel
# Author: Azzeddine Remmal

class SelfAttention

    # Architecture Parameters
    nEmbedDim       
    nScale          # Scaling factor (sqrt(d_k))
    
    # Sub-Layers (Linear Projections)
    oQueryLayer
    oKeyLayer
    oValueLayer
    oOutputLayer
    
    # Cache for Backward Pass
    oQ
    oKey
    oV

    bTrainable = true

    cName = "SelfAttention"

    # Causal flag for masked self-attention
    bCausal = false 
    
    func init nDim, lIsCausal  # Updated Init
        nEmbedDim = nDim
        nScale    = sqrt(nDim)
        bCausal   = lIsCausal  # Store flag
        
        # Initialize Projections: Input (Dim) -> Output (Dim)
        oQueryLayer = new Dense(nDim, nDim)
        oKeyLayer   = new Dense(nDim, nDim)
        oValueLayer = new Dense(nDim, nDim)
        
        # Final projection
        oOutputLayer = new Dense(nDim, nDim)
    
    # Input: Tensor (SeqLen, EmbedDim)
    # Uses Fused Kernel for Zero-Copy execution.    
    func forward oInput

        # 1. Linear Projections (Ring manages this, fast enough)
        oQ = oQueryLayer.forward(oInput)
        oKey = oKeyLayer.forward(oInput)
        oV = oValueLayer.forward(oInput)
        
        # Prepare Output Tensor
        oOutput = new Tensor(oInput.nRows, nEmbedDim)
        
        # 2. THE FUSED KERNEL (Magical Speedup)
        # tensor_attention_fast(Q, K, V, Output, Scale)
        # 1.0 / nScale because we multiply by the inverse of sqrt(d)
        # SELECT KERNEL BASED ON MODE
        if bCausal
            tensor_attention_causal(oQ.pData, oKey.pData, oV.pData, oOutput.pData, 1.0 / nScale)
        else
            tensor_attention_fast(oQ.pData, oKey.pData, oV.pData, oOutput.pData, 1.0 / nScale)
        ok
        
        # 3. Final Projection
        oFinal = oOutputLayer.forward(oOutput)
        
        return oFinal
    
    # Gradient Calculation.
    # Since the Forward Kernel dropped the intermediate scores to save memory/time,
    # we quickly re-compute the necessary parts here using Ring Ops.
    # (This is a technique called Gradient Checkpointing).
    func backward oGradOutput
        
        # 1. Gradient through Output Layer
        oGradContext = oOutputLayer.backward(oGradOutput)
        
        # --- Reconstruct Attention Weights (Needed for derivatives) ---
        # Scores = Softmax( (Q . K^T) / scale )
        oK_T = oKey.transpose()
        oScores = oQ.matmul(oK_T)
        oScores.scalarMul(1.0 / nScale)
        oScores.softmax() 
        oAttnWeights = oScores
        # ---------------------------------------------------------------
        
        # 2. Gradient w.r.t Value (V)
        # dV = Weights^T . GradContext
        oAttnWeights_T = oAttnWeights.transpose()
        oGradV = oAttnWeights_T.matmul(oGradContext)
        
        # 3. Gradient w.r.t Weights (Scores)
        # dScores = GradContext . V^T
        oV_T = oV.transpose()
        oGradScores = oGradContext.matmul(oV_T)
        
        # 4. Gradient w.r.t Q and K (Simplified Softmax derivative flow)
        # This is an approximation for v2.0 training stability
        oGradScores.scalarMul(1.0 / nScale)
        
        # dQ = GradScores . K
        oGradQ = oGradScores.matmul(oKey)
        
        # dK = (GradScores^T . Q)^T
        oQ_T = oQ.transpose()
        oGradK_T = oQ_T.matmul(oGradScores)
        oGradK = oGradK_T.transpose()
        
        # 5. Backprop through Linear Layers
        oGradInputQ = oQueryLayer.backward(oGradQ)
        oGradInputK = oKeyLayer.backward(oGradK)
        oGradInputV = oValueLayer.backward(oGradV)
        
        # Sum gradients
        oGradInput = new Tensor(oGradOutput.nRows, oGradOutput.nCols)
        oGradInput.zeros()
        oGradInput.add(oGradInputQ)
        oGradInput.add(oGradInputK)
        oGradInput.add(oGradInputV)
        
        return oGradInput

    func freeze
        bTrainable = false
        oQueryLayer.freeze()
        oKeyLayer.freeze()
        oValueLayer.freeze()
        oOutputLayer.freeze()

    func unfreeze
        bTrainable = true
        oQueryLayer.unfreeze()
        oKeyLayer.unfreeze()
        oValueLayer.unfreeze()
        oOutputLayer.unfreeze()

    # Update Weights
    func updateWeights oOptimizer
        oQueryLayer.updateWeights(oOptimizer)
        oKeyLayer.updateWeights(oOptimizer)
        oValueLayer.updateWeights(oOptimizer)
        oOutputLayer.updateWeights(oOptimizer)