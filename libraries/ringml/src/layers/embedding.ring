# File: src/layers/embedding.ring
# Description: Embedding Layer (Lookup Table) utilizing RingTensor C-Kernel
# Author: Azzeddine Remmal

class Embedding

    weights         # Tensor
    grads           # Tensor
    
    aInputIndices   # Tensor (Inputs)
    
    # Cache dimensions
    nVocabSize
    nEmbedDim
    
    cName = "Embedding"
    
    func init nVocab, nDim
        nVocabSize = nVocab
        nEmbedDim  = nDim
        
        weights = new Tensor(nVocab, nDim)
        weights.random()
        weights.addScalar(-0.5)
        weights.scalarMul(0.02)
        
        grads = new Tensor(nVocab, nDim)
        grads.fill(0)
    
    func forward oInputTensor
        aInputIndices = oInputTensor
        return oInputTensor.embedding(weights)
    
    func backward oGradOutput
        # In Embedding, we accumulate gradients into the rows corresponding to the input indices
        # This is handled by the C-Kernel
        tensor_embedding_backward(oGradOutput.pData, aInputIndices.pData, grads.pData)
        return NULL # Embedding is usually the first layer, no need to propagate further back
    
    func updateWeights oOptimizer
        oOptimizer.updateTensor(weights, grads)
        grads.fill(0)
    
    func getParams
        return [ [weights, grads] ]