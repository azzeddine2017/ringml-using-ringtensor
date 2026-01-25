

func testGELU
    ? "═══════════════════════════════════════════════"
    ? "GELU TEST"
    ? "═══════════════════════════════════════════════"
    
    oGelu = new GELU
    
    # Test input
    oX = new Tensor(1, 5)
    oX.setVal(1, 1, -2.0)
    oX.setVal(1, 2, -1.0)
    oX.setVal(1, 3,  0.0)
    oX.setVal(1, 4,  1.0)
    oX.setVal(1, 5,  2.0)
    
    ? "Input: [-2, -1, 0, 1, 2]"
    
    # Forward
    oY = oGelu.forward(oX)
    
    ? "GELU output:"
    for i = 1 to 5
        ? "  GELU(" + oX.getVal(1, i) + ") = " + oY.getVal(1, i)
    next
    
    # Expected (approximate):
    # GELU(-2) ≈ -0.046
    # GELU(-1) ≈ -0.159
    # GELU(0)  = 0.0
    # GELU(1)  ≈ 0.841
    # GELU(2)  ≈ 1.954
    
    ? ""
    
    # Test backward
    oGrad = new Tensor(1, 5)
    oGrad.fill(1.0)  # Gradient = 1
    
    oGradInput = oGelu.backward(oGrad)
    
    ? "Gradient (should be different for each):"
    for i = 1 to 5
        ? "  d/dx at x=" + oX.getVal(1, i) + " : " + 
          oGradInput.getVal(1, i)
    next
    
    ? "═══════════════════════════════════════════════" + nl
