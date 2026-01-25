load "ringml.ring"


func main
    
    # --- Experiment 1: Normal Mode ---
    see "1. Normal Mode (Immediate Execution):" + nl
    A = new Tensor(2, 2)
    A.fill(1.0)
    
    B = new Tensor(2, 2)
    B.fill(2.0)
    
    C = A.add(B)
    see "   Result (1+2): " + C.getVal(1,1) + nl 
    
    # --- Experiment 2: Auto-Graph Mode ---
    see nl + "2. Auto-Graph Mode (Deferred Execution):" + nl
    
    # 1. Initialize the Graph
    g = new Graph
    g.init()
    
    # 2. Prepare Inputs
    X = new Tensor(2, 2)
    X.fill(10.0) 
    X.asGraphInput() 
    
    W = new Tensor(2, 2)
    W.fill(0.5)
    W.asGraphConstant()
    
    # 3. Record Operations (Seamless API)
    # This builds the graph without executing math yet
    see "   Recording: Y = (X * W) + X" + nl
    Y = X.matmul(W).add(X)
    
    # 4. Bind Memory for Output
    # This links the virtual result tensor to the graph's output buffer
    Y.bindMemory()
    
    # 5. Execute Graph
    see "   Executing Graph..." + nl
    g.forward()
    
    # Expected: (10 * 0.5 + 10 * 0.5) + 10 = 10 + 10 = 20
    see "   Result after Forward: " + Y.getVal(1,1) + nl
    
    # 6. Backward Pass
    see "   Running Backward..." + nl
    Y.backward()
    
    see "   Graph completed successfully!" + nl