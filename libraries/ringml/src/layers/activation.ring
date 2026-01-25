# File: src/layers/activation.ring
# Description: Activation Layers (ReLU, Sigmoid) with Backprop
# Author: Azzeddine Remmal



class Activation from Layer
    oOutput # Cache output

class Sigmoid from Activation
    func forward oInputTensor
        # Cache output for backward
        oOutput = oInputTensor.copy()
        return oOutput.sigmoid()

    func backward oGradOutput
        # dInput = dOutput * Sigmoid'(Output)
        # S'(x) = x * (1-x) calculated by :sigmoidprime
        
        oDerivative = oOutput.copy()
        oDerivative.sigmoidPrime() 
        
        # Element-wise multiplication
        oGradOutput.mul(oDerivative)
        
        return oGradOutput

class ReLU from Activation
    oInputCache 
    
    func forward oInputTensor
        oInputCache = oInputTensor.copy()
        
        oOutput = oInputTensor.copy()
        return oOutput.relu()

    func backward oGradOutput
        # dInput = dOutput * ReLU'(Input)
        
        oDerivative = oInputCache.copy()
        oDerivative.reluPrime() 
        
        oGradOutput.mul(oDerivative)
        return oGradOutput

class Tanh from Activation
    oOutput

    func forward oInputTensor
        # Cache output for backward pass
        oOutput = oInputTensor.copy()
        return oOutput.tanh()

    func backward oGradOutput
        # Gradient = GradOutput * (1 - Output^2)
        oDerivative = oOutput.copy()
        oDerivative.tanhPrime()
        
        oGradOutput.mul(oDerivative)
        return oGradOutput

class GELU from Activation

    oInputCache

    func forward oInput
        # Save input for backward pass
        oInputCache = oInput.copy()
        
        # Apply GELU in-place on the input (passed as result)
        # Note: In Sequential/Graph, usually we create new tensor to avoid modifying source
        # But for efficiency, if we own the tensor, we modify it.
        # Let's be safe and copy first if not done outside.
        
        oOutput = oInput.copy()
        return oOutput.gelu()

    func backward oGradOutput
        /*
            Derivative: dX = dOut * GELU_Prime(X)
        */
        
        # 1. Calculate GELU Prime of X
        # We use the cached input
        oDerivative = oInputCache
        oDerivative.geluPrime() # Transforms X to GELU'(X) in-place
        
        # 2. Multiply by Gradient
        # dX = Grad * Derivative
        oGradInput = oGradOutput.mul(oDerivative)
        
        return oGradInput
