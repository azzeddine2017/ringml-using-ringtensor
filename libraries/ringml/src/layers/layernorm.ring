# File: src/layers/layernorm.ring
# Description: Layer Normalization utilizing RingTensor C-Kernel
# Author: Azzeddine Remmal

class LayerNorm

    # Parameters
    gamma       # Scale (1 x Dim)
    beta        # Shift (1 x Dim)
    
    # Gradients
    g_gamma     
    g_beta
    
    # Cache
    oInputCache
    nEps = 0.0000001
    nDim

    bTrainable = true
    
    cName = "LayerNorm"

    func init nDimensions
        nDim = nDimensions
        
        # Init Gamma to 1.0
        gamma = new Tensor(1, nDim)
        gamma.fill(1.0)
        
        # Init Beta to 0.0
        beta = new Tensor(1, nDim)
        beta.fill(0.0)
        
        # Init Gradients
        g_gamma = new Tensor(1, nDim)
        g_gamma.zeros()
        g_beta  = new Tensor(1, nDim)
        g_beta.zeros()
        
    func forward oInput
        oInputCache = oInput
        return oInput.layerNorm(gamma, beta, nEps)

    func backward oGradOutput
        # 1. Gradient for Beta (Bias)
        oDb = oGradOutput.sum(0)
        g_beta.add(oDb)
        
        # 2. Gradient for Gamma (Scale)
        oDg = oGradOutput.sum(0)
        g_gamma.add(oDg)
        
        # 3. Gradient for Input
        oGradInput = oGradOutput.copy()
        return oGradInput

    func freeze
        bTrainable = false

    func unfreeze
        bTrainable = true

    func updateWeights oOptimizer
        if !bTrainable return ok 
        oOptimizer.updateTensor(gamma, g_gamma)
        oOptimizer.updateTensor(beta, g_beta)
        
        g_gamma.fill(0)
        g_beta.fill(0)

    func getParams
        return [ [gamma, g_gamma], [beta, g_beta] ]