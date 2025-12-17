# File: main.ring
# Description: Main entry point
# Author: Azzeddine Remmal
# Date: 2025-12-15
# Version: 1.1.0
# License: MIT License

load "stdlib.ring"
load "ringml.ring"



func main

    RingMLInfo()

    see nl
    
    # --- Metadata ---
    printMeta("Library",   "RingML - Machine Learning Library for Ring")
    printMeta("Version",   "1.1.0")
    printMeta("Developer", "Azzeddine Remmal")
    printMeta("License",   "MIT License")
    see nl

    # --- Quick Start ---
    printHeader("QUICK START GUIDE")

    printStep("1. Data Preparation:")
    see "   Use DataSplitter to handle raw CSV data and DataLoader for batching." + nl
    printPath("   see samples/UsingRingML/loader_demo.ring")
    see nl

    printStep("2. Building the Model:")
    see "   Construct a model using Tanh for hidden layers and Dropout for regularization." + nl
    printCode("   model = new Sequential")
    printCode("   model.add(new Dense(10, 64))")
    printCode("   model.add(new Tanh)")
    printCode("   model.add(new Dropout(0.2))")
    printCode("   model.add(new Dense(64, 3))")
    printCode("   model.add(new Softmax)")
    see nl

    printStep("3. Training:")
    see "   Use Adam or SGD optimizers." + nl
    printPath("   see samples/UsingRingML/mnist/mnist_train.ring")
    see nl

    # --- Examples ---
    printHeader("EXAMPLES")
    see "Find ready-to-run examples in the "; printPath("'samples/UsingRingML'"); see " folder:" + nl + nl

    printExample("samples/UsingRingML/xor_train.ring",         "Binary Classification (XOR)")
    printExample("samples/UsingRingML/classify_demo.ring",     "Multi-Class Classification")
    printExample("samples/UsingRingML/mnist/mnist_train.ring", "MNIST Digit Recognition")
    printExample("samples/UsingRingML/Chess_End_Game/",        "Full Real-world Project")
    see nl

    # --- Run Command ---
    printHeader("RUN AN EXAMPLE")
    printCode("cd samples/UsingRingML/")
    printCode("ring xor_train.ring")
    see nl

# --- Helper Functions for Style ---

func printHeader cTitle
    oStyl.bright_blue(:BOLD, "=========================================" + nl)
    oStyl.bright_blue(:BOLD, "           " + cTitle + "             " + nl)
    oStyl.bright_blue(:BOLD, "=========================================" + nl)

func printMeta cKey, cVal
    oStyl.cyan(:BOLD, cKey) 
    see " : " + cVal + nl

func printStep cText
    oStyl.white(:BOLD, cText + nl)

func printCode cCode
    oStyl.green(:BOLD, cCode + nl)

func printPath cPath
    oStyl.bright_blue(:BOLD, cPath + nl)

func printExample cPath, cDesc
    oStyl.cyan(:BOLD, "- " + cPath)
    # حساب المسافة للمحاذاة
    nSpaces = 50 - len(cPath)
    if nSpaces < 1 nSpaces = 1 ok
    see copy(" ", nSpaces) + ": " + cDesc + nl


