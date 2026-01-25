# Project: RingML
# File: tests/test_autograph_integration.ring
# Description: Unit Test for Auto-Graph Integration (Standard & Linear Attention)
# Author: Azzeddine Remmal

load "ringml.ring"

decimals(5)
func main
    see copy("=", 70) + nl
    see "  RingML: Auto-Graph Integration Unit Test" + nl
    see copy("=", 70) + nl

    # ---------------------------------------------------------
    # 1. Setup Data
    # ---------------------------------------------------------
    nBatch    = 1
    nSeqLen   = 4
    nEmbedDim = 8
    nHeads    = 2
    
    see "[1] Preparing Data..." + nl
    oInput  = new Tensor(nSeqLen, nEmbedDim)
    oInput.random()
    
    oTarget = new Tensor(nSeqLen, nEmbedDim)
    oTarget.fill(0.5)
    
    # ---------------------------------------------------------
    # 2. Test Standard Attention in Graph Mode
    # ---------------------------------------------------------
    see nl + "[2] Testing Standard Attention (Graph Mode)..." + nl
    
    oModel = new Sequential()
    oModel.add(new MultiHeadAttention(nEmbedDim, nHeads, true, :STANDARD))
    
    oLoss = new MSELoss()
    oOptim = new Adam(0.01, 0)
    
    try
        # Enable Graph Mode
        oInput.asGraphInput()
        oTarget.asGraphInput()
        
        # Bind Parameters
        aParams = oModel.getParams()
        for aPair in aParams
            oW = aPair[1]
            oG = aPair[2]
            oW.asGraphWeight(oG)
        next
        
        # Compile
        oLogitsNode = oModel.compile(oInput)
        oLossNode = oLoss.calculate(oLogitsNode, oTarget)
        
        see "    >> Model Compiled Successfully. Graph Node ID: " + oLossNode.nGraphNodeID + nl
        
        # Run Forward/Backward/Update in Graph
        graph_set_optimizer(1) # Adam
        graph_run(100, 0.01, 5.0)
        
        # Get Loss
        pLossTensor = graph_get_output(oLossNode.nGraphNodeID)
        nLoss = tensor_get(pLossTensor, 1, 1)
        
        see "    >> Initial Loss: " + nLoss + nl
        if nLoss >= 0
            see "    >> Status: PASS (Graph Execution Successful)" + nl
        else
            see "    >> Status: FAIL (Invalid Loss)" + nl
        ok
        
    catch
        see "    >> CRITICAL ERROR in Standard Graph Test!" + nl
        see "    Error: " + cCatchError + nl
    done

    # ---------------------------------------------------------
    # 3. Test Linear Attention (Causal) in Graph Mode
    # ---------------------------------------------------------
    see nl + "[3] Testing Linear Causal Attention (Graph Mode)..." + nl
    
    graph_init() # Reset Graph
    
    oModelL = new Sequential()
    oModelL.add(new MultiHeadAttention(nEmbedDim, nHeads, true, :LINEAR))
    
    oInputL = new Tensor(nSeqLen, nEmbedDim)
    oInputL.random()
    oTargetL = new Tensor(nSeqLen, nEmbedDim)
    oTargetL.fill(0.5)
    
    try
        oInputL.asGraphInput()
        oTargetL.asGraphInput()
        
        aParams = oModelL.getParams()
        for aPair in aParams
            aPair[1].asGraphWeight(aPair[2])
        next
        
        oLogitsL = oModelL.compile(oInputL)
        oLossL = oLoss.calculate(oLogitsL, oTargetL)
        
        graph_set_optimizer(1)
        graph_run(50, 0.01, 5.0) # Run 5 steps
        
        pLossL = graph_get_output(oLossL.nGraphNodeID)
        nLossL = tensor_get(pLossL, 1, 1)
        
        see "    >> Loss after 5 steps: " + nLossL + nl
        if nLossL >= 0
            see "    >> Status: PASS (Linear Causal Graph Successful)" + nl
        else
            see "    >> Status: FAIL" + nl
        ok
        
    catch
        see "    >> CRITICAL ERROR in Linear Graph Test!" + nl
        see "    Error: " + cCatchError + nl
    done

    # ---------------------------------------------------------
    # 4. Test Linear Attention (Global) in Graph Mode
    # ---------------------------------------------------------
    see nl + "[4] Testing Linear Global Attention (Graph Mode)..." + nl
    
    graph_init()
    
    oModelG = new Sequential()
    oModelG.add(new MultiHeadAttention(nEmbedDim, nHeads, false, :LINEAR))
    
    oInputG = new Tensor(nSeqLen, nEmbedDim)
    oInputG.random()
    
    try
        oInputG.asGraphInput()
        oTargetL.asGraphInput() # Reuse target
        
        for aPair in oModelG.getParams() aPair[1].asGraphWeight(aPair[2]) next
        
        oLogitsG = oModelG.compile(oInputG)
        oLossG = oLoss.calculate(oLogitsG, oTargetL)
        
        graph_set_optimizer(1)
        graph_run(50, 0.01, 5.0)
        
        see "    >> Status: PASS (Linear Global Graph Successful)" + nl
        
    catch
        see "    >> CRITICAL ERROR in Linear Global Test!" + nl
        see "    Error: " + cCatchError + nl
    done

    see nl + copy("=", 70) + nl
    see "  Auto-Graph Integration Tests Completed." + nl
    see copy("=", 70) + nl
