# File: main.ring
# Description: Main entry point
# Author: Azzeddine Remmal


load "stdlib.ring"
load "ringml.ring"


func main

    RingMLInfo()

    see nl
    
    # --- Metadata ---
    printMeta("Library",   "RingML - Machine Learning Library for Ring")
    printMeta("Version",   "1.1.1")
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

func RingMLInfo

 oStyl.seeString(:YELLOW, :NONE, "
  _____  _             __  __ _      
 |  __ \(_)           |  \/  | |     
 | |__) |_ _ __   __ _| \  / | |     
 |  _  /| | '_ \ / _` | |\/| | |     
 | | \ \| | | | | (_| | |  | | |____ 
 |_|  \_\_|_| |_|\__, |_|  |_|______|
                  __/ |              
                 |___/               
")
	banner = []
	banner[:topBorder] = colorText([:text = "╭───────────────────────────────────────────────╮", :color = :BRIGHT_BLUE, :style = :BOLD])
	banner[:tEmptyLine] = colorText([:text = "│                                               │", :color = :BRIGHT_BLUE])
	
	title = "RingML"
	titlePad = floor((47 - len(title)) / 2)
	banner[:titleLine] = colorText([:text = "│" + space(titlePad), :color = :BRIGHT_BLUE]) + colorText([:text = title, :color = :CYAN, :style = :BOLD]) + colorText([:text = space(47-titlePad-len(title)) + "│", :color = :BRIGHT_BLUE])
	
	banner[:tEmptyLine2] = colorText([:text = "│                                               │", :color = :BRIGHT_BLUE])
	
	# Description lines
	desc1 = "High-performance, object-oriented"
	desc1Pad = floor((47 - len(desc1)) / 2)
	banner[:descLine1] = colorText([:text = "│" + space(desc1Pad), :color = :BRIGHT_BLUE]) + colorText([:text = desc1, :color = :WHITE]) + colorText([:text = space(47-desc1Pad-len(desc1)) + "│", :color = :BRIGHT_BLUE])
	
	desc2 = "Deep Learning framework"
	desc2Pad = floor((47 - len(desc2)) / 2)
	banner[:descLine2] = colorText([:text = "│" + space(desc2Pad), :color = :BRIGHT_BLUE]) + colorText([:text = desc2, :color = :WHITE]) + colorText([:text = space(47-desc2Pad-len(desc2)) + "│", :color = :BRIGHT_BLUE])
	
	desc3 = "built for the Ring programming language."
	desc3Pad = floor((47 - len(desc3)) / 2)
	banner[:descLine3] = colorText([:text = "│" + space(desc3Pad), :color = :BRIGHT_BLUE]) + colorText([:text = desc3, :color = :WHITE]) + colorText([:text = space(47-desc3Pad-len(desc3)) + "│", :color = :BRIGHT_BLUE])
	
	banner[:tEmptyLine3] = colorText([:text = "│                                               │", :color = :BRIGHT_BLUE])
	
	versionStr = "Version 1.1.1"
	versionPad = floor((47 - len(versionStr)) / 2)
	banner[:versionLine] = colorText([:text = "│" + space(versionPad), :color = :BRIGHT_BLUE]) + colorText([:text = versionStr, :color = :YELLOW, :style = :BOLD]) + colorText([:text = space(47-versionPad-len(versionStr)) + "│", :color = :BRIGHT_BLUE])
	
	authorStr = "Author: Azzeddine Remmal"
	authorPad = floor((47 - len(authorStr)) / 2)
	banner[:authorLine] = colorText([:text = "│" + space(authorPad), :color = :BRIGHT_BLUE]) + colorText([:text = authorStr, :color = :MAGENTA]) + colorText([:text = space(47-authorPad-len(authorStr)) + "│", :color = :BRIGHT_BLUE])
	
	urlStr = "License: MIT License"
	urlPad = floor((47 - len(urlStr)) / 2)
	banner[:urlLine] = colorText([:text = "│" + space(urlPad), :color = :BRIGHT_BLUE]) + colorText([:text = urlStr, :color = :GREEN, :style = :UNDERLINE]) + colorText([:text = space(47-urlPad-len(urlStr)) + "│", :color = :BRIGHT_BLUE])
	
	banner[:bEmptyLine] = colorText([:text = "│                                               │", :color = :BRIGHT_BLUE])
	banner[:bottomBorder] = colorText([:text = "╰───────────────────────────────────────────────╯", :color = :BRIGHT_BLUE, :style = :BOLD])
	
	for line in banner
		see line[2] + nl
	next

func colorText aParams
	cText = aParams[:text]
	cColor = aParams[:color]
	if not cColor cColor = :WHITE ok
	cStyle = aParams[:style] 
	if not cStyle cStyle = :NONE ok
	return oStyl.costom(cColor, cStyle, cText)
