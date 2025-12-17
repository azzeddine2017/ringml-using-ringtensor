# File: src/core/tensor.ring
# Description: Core Tensor class - Updated for Adam Optimizer
# Author: Azzeddine Remmal


class Tensor
    pData   = NULL  # C Pointer
    nRows   = 0     
    nCols   = 0     

    func init nR, nC
        self.nRows = nR
        self.nCols = nC
        pData = tensor_init(nR, nC)
        return self
        
    func getVal r, c
        return tensor_get(pData, r, c)
        
    func setVal r, c, val
        tensor_set(pData, r, c, val)

    func copy
        oNew = new Tensor(nRows, nCols)
        # We can implement tensor_copy in C later for speed
        # For now, manual copy is fine as it's rare in training loop
        for r=1 to nRows
            for c=1 to nCols
                oNew.setVal(r,c, getVal(r,c))
            next
        next
        return oNew

    # --- Math ---
    func add oTensor
        tensor_add(pData, oTensor.pData)
        return self

    func sub oTensor
        tensor_sub(pData, oTensor.pData)
        return self

    func mul oTensor
        tensor_mul_elem(pData, oTensor.pData)
        return self
        
    func div oTensor
        tensor_div(pData, oTensor.pData)
        return self

    func scalar_mul nVal
        tensor_scalar_mul(pData, nVal)
        return self
        
    func add_scalar nVal
        tensor_add_scalar(pData, nVal)
        return self

    # --- Matrix Ops ---
    func matmul oTensor
        if self.nCols != oTensor.nRows raise("Dim Mismatch") ok
        oRes = new Tensor(nRows, oTensor.nCols)
        tensor_matmul(pData, oTensor.pData, oRes.pData)
        return oRes

    func transpose
        oRes = new Tensor(nCols, nRows)
        tensor_transpose(pData, oRes.pData)
        return oRes

    func sum nAxis
        if nAxis = 1
            oRes = new Tensor(nRows, 1)
        else
            oRes = new Tensor(1, nCols)
        ok
        tensor_sum(pData, nAxis, oRes.pData)
        return oRes
        
    func mean
        return tensor_mean(pData)

    # --- Transforms ---
    func square
        tensor_square(pData)
        return self
        
    func sqrt
        tensor_sqrt(pData)
        return self
        
    func exp_func
        tensor_exp(pData)
        return self

    # --- Init ---
    func random
        tensor_random(pData)
        return self
        
    func zeros
        tensor_fill(pData, 0.0)
        return self
        
    func fill nVal
        tensor_fill(pData, nVal)
        return self

    # --- Activations ---
    func sigmoid
        tensor_sigmoid(pData)
        return self
    
    func sigmoid_prime
        tensor_sigmoid_prime(pData)
        return self

    func tanh
        tensor_tanh(pData)
        return self
        
    func tanh_prime
        tensor_tanh_prime(pData)
        return self
        
    func relu
        tensor_relu(pData)
        return self
        
    func relu_prime
        tensor_relu_prime(pData)
        return self
        
    func softmax
        tensor_softmax(pData)
        return self
        
    func apply_dropout nRate
        tensor_dropout(pData, nRate)
        return self

    # --- Utilities ---
    func print_shape
        see "(" + nRows + ", " + nCols + ")" + nl

    func decimals nPlaces
        # Display logic, doesn't affect data
        return self

    func checkDimensions oTensor
        if nRows != oTensor.nRows or nCols != oTensor.nCols
            raise("Dimension Mismatch")
        ok

    func print
        see "Tensor Shape: (" + nRows + ", " + nCols + ")" + nl
        if nRows <= 10 and nCols <= 10
            for r=1 to nRows
                see "| "
                for c=1 to nCols
                    val = getVal(r,c)
                    if val >= 0 see " " ok
                    see "" + floor(val*10000)/10000 
                    if c != nCols see ", " ok
                next
                see " |" + nl
            next
            see nl
        else
            see "Data is too large to display." + nl
        ok
        see nl
    
    func toList
        aList = list(nRows)
        for r=1 to nRows
            aList[r] = list(nCols)
            for c=1 to nCols
                aList[r][c] = getVal(r,c)
            next
        next
        return aList

    func fromList aList
        for r=1 to nRows
            for c=1 to nCols
                setVal(r,c, aList[r][c])
            next
        next