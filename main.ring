# The Main File

func main

	? "
  _____  _             __  __ _      
 |  __ \(_)           |  \/  | |     
 | |__) |_ _ __   __ _| \  / | |     
 |  _  /| | '_ \ / _` | |\/| | |     
 | | \ \| | | | | (_| | |  | | |____ 
 |_|  \_\_|_| |_|\__, |_|  |_|______|
                  __/ |              
                 |___/               
"
	? "RingML - Machine Learning Library for Ring"
	? "Version   : 1.0.0"
	? "Developer : Azzeddine Remmal"
	? "License   : MIT License"
	? ""
	? "========================================="
	? "           QUICK START GUIDE             "
	? "========================================="
	? "1. Data Preparation:"
	? "   Use DataSplitter to handle raw CSV data and DataLoader for batching."
	? "   see samples/UsingRingML/loader_demo.ring"
	? ""
	? "2. Building the Model:"
	? "   Construct a model using Tanh for hidden layers and Dropout for regularization."
	? "   model = new Sequential"
	? "   model.add(new Dense(10, 64))"
	? "   model.add(new Tanh)"
	? "   model.add(new Dropout(0.2))"
	? "   model.add(new Dense(64, 3))"
	? "   model.add(new Softmax)"
	? ""
	? "3. Training:"
	? "   Use Adam or SGD optimizers."
	? "   see samples/UsingRingML/mnist/mnist_train.ring"
	? ""
	? "========================================="
	? "              EXAMPLES                   "
	? "========================================="
	? "Find ready-to-run examples in the 'samples/UsingRingML' folder:"
	? "- samples/UsingRingML/xor_train.ring          : Binary Classification (XOR)"
	? "- samples/UsingRingML/classify_demo.ring      : Multi-Class Classification"
	? "- samples/UsingRingML/mnist/mnist_train.ring  : MNIST Digit Recognition"
	? "- samples/UsingRingML/Chess_End_Game/         : Full Real-world Project"
	? ""
	? "Run an example:"
	? "cd samples/UsingRingML/"
	? "ring xor_train.ring"