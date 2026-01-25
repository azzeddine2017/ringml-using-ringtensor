# Project: RingML
# File: tests/test_attention_layer.ring
# Description: Self-Attention Layer Test
# Author: Azzeddine Remmal

load "ringml.ring"	

func main
    see copy("=", 60) + nl
    see "  RingML: Self-Attention Mechanism Test (Fused Kernel)" + nl
    see copy("=", 60) + nl

    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    nSeqLen   = 5   # e.g., Sentence with 5 tokens
    nEmbedDim = 4   # Embedding Dimension
    
    see "[1] Initializing SelfAttention Layer..." + nl
    see "    Sequence Length: " + nSeqLen + " | Embed Dim: " + nEmbedDim + nl
    
    oAttn = new SelfAttention(nEmbedDim)
    
    # Create Dummy Input (Batch=1, Seq=5, Dim=4)
    oInput = new Tensor(nSeqLen, nEmbedDim)
    oInput.random() # Fill with random values [0, 1]
    
    # ---------------------------------------------------------
    # 2. Test Forward Pass (The C Kernel)
    # ---------------------------------------------------------
    see nl + "[2] Running FORWARD Pass (C Fused Kernel)..." + nl
    
    try
        oOutput = oAttn.forward(oInput)
        
        see "    Output Shape: (" + oOutput.nRows + ", " + oOutput.nCols + ")" + nl
        
        # Verify Shape
        if oOutput.nRows = nSeqLen and oOutput.nCols = nEmbedDim
            see "    >> Status: PASS (Dimensions Correct)" + nl
            
            # Check for NaNs or Zeros (Basic sanity check)
            val = oOutput.getVal(1,1)
            see "    Sample Output Value[1,1]: " + val + nl
            if val != 0 
                see "    >> Status: PASS (Data looks active)" + nl
            else
                see "    >> Warning: Output might be all zeros." + nl
            ok
        else
            see "    >> Status: FAIL (Dimension Mismatch)" + nl
            raise("Dim Error")
        ok
        
    catch
        see "    >> CRITICAL ERROR in Forward Pass!" + nl
        see "    Error: " + cCatchError + nl
        return
    done

    # ---------------------------------------------------------
    # 3. Test Backward Pass (Ring Logic)
    # ---------------------------------------------------------
    see nl + "[3] Running BACKWARD Pass (Ring Reconstruction)..." + nl
    
    try
        # Create fake gradients coming from next layer
        oGradOut = new Tensor(nSeqLen, nEmbedDim)
        oGradOut.fill(0.1) # Simple gradient
        
        oGradIn = oAttn.backward(oGradOut)
        
        see "    Input Gradient Shape: (" + oGradIn.nRows + ", " + oGradIn.nCols + ")" + nl
        
        if oGradIn.nRows = nSeqLen and oGradIn.nCols = nEmbedDim
            see "    >> Status: PASS (Backward Flow Successful)" + nl
        else
            see "    >> Status: FAIL (Gradient Shape Mismatch)" + nl
        ok
        
    catch
        see "    >> CRITICAL ERROR in Backward Pass!" + nl
        see "    Error: " + cCatchError + nl
    done

    see nl + copy("=", 60) + nl
    see "  Attention Test Completed." + nl
    see copy("=", 60) + nl