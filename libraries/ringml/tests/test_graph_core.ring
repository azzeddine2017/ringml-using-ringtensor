load "ringml.ring"

func main
    
    ? "=================================================="
    ? "   CORE ENGINE SANITY CHECK (The Truth Test)"
    ? "=================================================="
    
    # 1. Data (5 samples, 4 features)
    # Pattern: If feature 1 is high -> Class 1, else Class 0
    nBatch = 5
    nIn    = 4
    nOut   = 2
    
    oInput = new Tensor(nBatch, nIn)
    # Sample 1: [1, 0, 0, 0] -> Class 0
    oInput.setVal(1,1, 1.0) oInput.setVal(1,2, 0.1) oInput.setVal(1,3, 0.1) oInput.setVal(1,4, 0.1)
    # Sample 2: [0, 1, 0, 0] -> Class 1
    oInput.setVal(2,1, 0.1) oInput.setVal(2,2, 1.0) oInput.setVal(2,3, 0.1) oInput.setVal(2,4, 0.1)
    # ... Repeats to ensure stability
    oInput.setVal(3,1, 0.9) oInput.setVal(3,2, 0.1) oInput.setVal(3,3, 0.1) oInput.setVal(3,4, 0.1)
    oInput.setVal(4,1, 0.1) oInput.setVal(4,2, 0.9) oInput.setVal(4,3, 0.1) oInput.setVal(4,4, 0.1)
    oInput.setVal(5,1, 0.8) oInput.setVal(5,2, 0.2) oInput.setVal(5,3, 0.1) oInput.setVal(5,4, 0.1)

    oTarget = new Tensor(nBatch, nOut)
    oTarget.zeros()
    oTarget.setVal(1, 1, 1.0) # Class 0
    oTarget.setVal(2, 2, 1.0) # Class 1
    oTarget.setVal(3, 1, 1.0) # Class 0
    oTarget.setVal(4, 2, 1.0) # Class 1
    oTarget.setVal(5, 1, 1.0) # Class 0
    
    # 2. Model (Single Linear Layer)
    # Weights: (4, 2)
    oW = new Tensor(nIn, nOut)
    oW.random() 
    oW.subScalar(0.5) 
    oW.scalarMul(0.1) # Small init
    
    # Bias: (1, 2)
    oB = new Tensor(1, nOut)
    oB.zeros()
    
    # 3. Graph Construction
    ? "[1] Building Graph..."
    
    # Inputs
    id_In = graph_node(1, -1, -1, -1, 0.0) # OP_INPUT
    graph_bind_memory(id_In, oInput.pData)
    
    id_Tg = graph_node(1, -1, -1, -1, 0.0) # OP_INPUT
    graph_bind_memory(id_Tg, oTarget.pData)
    
    # Weights (Trainable)
    id_W = graph_node(2, -1, -1, -1, 0.0) # OP_WEIGHT
    graph_bind_memory(id_W, oW.pData)
    # Create Grad buffers
    pGradW = tensor_init(nIn, nOut) 
    graph_bind_grad(id_W, pGradW)
    
    id_B = graph_node(2, -1, -1, -1, 0.0) # OP_WEIGHT
    graph_bind_memory(id_B, oB.pData)
    pGradB = tensor_init(1, nOut)
    graph_bind_grad(id_B, pGradB)
    
    # Operation: Linear -> Softmax -> CrossEntropy
    # Z = X * W
    id_MatMul = graph_node(10, id_In, id_W, -1, 0.0) # OP_MATMUL
    
    # Z = Z + B
    id_Add = graph_node(32, id_MatMul, id_B, -1, 0.0) # OP_ADD_ROW_VEC
    
    # A = Softmax(Z) - (Included in CrossEntropy usually, but doing explicit for test)
    # Let's use direct CrossEntropy Loss (which fuses Softmax) if we have it
    # Op 28 = CROSSENTROPY
    
    # Note: If your CrossEntropy expects Logits, feed id_Add directly.
    id_Loss = graph_node(28, id_Add, id_Tg, -1, 0.0)
    
    # 4. Training Loop (Pure C)
    ? "[2] Running Training (500 Epochs)..."
    
    # Optimizer: Adam (Type 1), LR=0.01
    graph_set_optimizer(1) 
    
    # Run
    t1 = clock()
    graph_run(500, 0.01)
    t2 = clock()
    
    # 5. Check Results
    
    # Get Loss Value
    pLossVal = graph_get_output(id_Loss)
    nFinalLoss = tensor_get(pLossVal, 1, 1)
    
    ? "Final Loss: " + nFinalLoss
    ? "Time: " + ((t2-t1)/clockspersecond()) + "s"
    
    if nFinalLoss < 0.1
        ? oStyl.green(:BOLD, ">>> TEST PASSED: Engine is Healthy! <<<")
        ? "Now we know the problem is in the Transformer complexity."
    else
        ? oStyl.red(:BOLD, ">>> TEST FAILED: Engine is Broken! <<<")
        ? "If a simple Linear layer can't learn, Transformer never will."
        ? "Check: ring_tensor.c (MatMul or Adam Update logic)"
    ok