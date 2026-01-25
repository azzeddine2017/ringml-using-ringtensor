

    

func testClippingDirectly
    ? "═══════════════════════════════════════════════"
    ? "DIRECT CLIPPING TEST"
    ? "═══════════════════════════════════════════════"
    
    # Create 3 tensors with known gradients
    oG1 = new Tensor(2, 2)
    oG1.fill(10.0)  # All values = 10
    
    oG2 = new Tensor(2, 2)
    oG2.fill(10.0)
    
    oG3 = new Tensor(2, 2)
    oG3.fill(10.0)
    
    # Calculate expected norm
    # 3 tensors × 4 values × 10² = 1200
    # √1200 = 34.64
    
    ? "Before clipping:"
    ? "  Tensor 1 [0,0]: " + oG1.getVal(1, 1)
    ? "  Expected norm: ~34.64"
    ? ""
    
    # Collect pointers
    aGrads = [oG1.pData, oG2.pData, oG3.pData]
    
    # Clip to max_norm = 1.0
    nNorm = tensor_clip_global_norm(aGrads, 1.0)
    
    ? "After clipping:"
    ? "  Returned norm: " + nNorm
    ? "  Expected norm: ~34.64"
    ? ""
    
    ? "  Tensor 1 [0,0]: " + oG1.getVal(1, 1)
    ? "  Expected value: " + (10.0 * (1.0 / 34.64))
    ? "  (should be ~1.44)"
    ? ""
    
    nExpectedValue = 10.0 * (1.0 / 34.64)
    nActualValue = oG1.getVal(1, 1)
    nDiff = fabs(nActualValue - nExpectedValue)
    
    if nDiff < 0.1
        ? "✅ PASS: Clipping working correctly!"
    else
        ? "❌ FAIL: Clipping NOT working!"
        ? "   Difference: " + nDiff
    ok
    
    ? "═══════════════════════════════════════════════" + nl