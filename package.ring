aPackageInfo = [
	:name = "The RingML-using-ringtensor Package",
	:description = "Our RingML-using-ringtensor package using the Ring programming language",
	:folder = "ringml-using-ringtensor",
	:developer = "Azzeddine Remmal",
	:email = "azzeddine.remmal@gmail.com",
	:license = "MIT License",
	:version = "1.1.0",
	:ringversion = "1.24",
	:versions = 	[
		[
			:version = "1.1.0",
			:branch = "master"
		]
	],
	:libs = 	[
		[
			:name = "stdlib",
			:version = "1.0.22",
			:providerusername = ""
		],
		[
			:name = "ringtensor",
			:version = "1.1.0",
			:providerusername = ""
		],
		[
			:name = "csvlib",
			:version = "1.0.7",
			:providerusername = ""
		],
		[
			:name = "jsonlib",
			:version = "1.0.4",
			:providerusername = ""
		]
	],
	:files = 	[
		"main.ring",
		"README.md",
		"setup.ring"
	],
	:ringfolderfiles = 	[
		"bin/load/ringml.ring",
		"libraries/ringml/project_documents/Performance Optimization Report.txt",
		"libraries/ringml/project_documents/RingML/FastPro_Fix_Roadmap.md",
		"libraries/ringml/project_documents/RingML/FastPro_Technical_Report.md",
		"libraries/ringml/project_documents/RingML/README.md",
		"libraries/ringml/project_documents/RingML/uml/class_diagram.md",
		"libraries/ringml/README.md",
		"libraries/ringml/src/core/tensor.ring",
		"libraries/ringml/src/data/dataset.ring",
		"libraries/ringml/src/data/datasplitter.ring",
		"libraries/ringml/src/data/universaldataset.ring",
		"libraries/ringml/src/layers/activation.ring",
		"libraries/ringml/src/layers/dense.ring",
		"libraries/ringml/src/layers/dropout.ring",
		"libraries/ringml/src/layers/layer.ring",
		"libraries/ringml/src/layers/softmax.ring",
		"libraries/ringml/src/loss/crossentropy.ring",
		"libraries/ringml/src/loss/mse.ring",
		"libraries/ringml/src/model/sequential.ring",
		"libraries/ringml/src/optim/adam.ring",
		"libraries/ringml/src/optim/sgd.ring",
		"libraries/ringml/src/ringml.ring",
		"libraries/ringml/src/utils/serializer.ring",
		"libraries/ringml/src/utils/Styler.ring",
		"libraries/ringml/src/utils/visualizer.ring",
		"libraries/ringml/tests/test_step1.ring",
		"libraries/ringml/tests/test_step2.ring",
		"libraries/ringml/tests/test_step3.ring",
		"samples/UsingRingML/benchmark_bottleneck.ring",
		"samples/UsingRingML/Chess_End_Game/chess_ai_tool.ring",
		"samples/UsingRingML/Chess_End_Game/chess_app.ring",
		"samples/UsingRingML/Chess_End_Game/chess_dataset.ring",
		"samples/UsingRingML/Chess_End_Game/chess_final_model.ring",
		"samples/UsingRingML/Chess_End_Game/chess_train_lite.ring",
		"samples/UsingRingML/Chess_End_Game/chess_train_split.ring",
		"samples/UsingRingML/Chess_End_Game/chess_train_universal.ring",
		"samples/UsingRingML/Chess_End_Game/chess_utils.ring",
		"samples/UsingRingML/Chess_End_Game/data/chess.csv",
		"samples/UsingRingML/Chess_End_Game/imgs/BKing.png",
		"samples/UsingRingML/Chess_End_Game/imgs/INSTRUCTIONS.md",
		"samples/UsingRingML/Chess_End_Game/imgs/WKing.png",
		"samples/UsingRingML/Chess_End_Game/imgs/WRook.png",
		"samples/UsingRingML/Chess_End_Game/model/chess_final.rdata",
		"samples/UsingRingML/Chess_End_Game/model/chess_model_lite.rdata",
		"samples/UsingRingML/Chess_End_Game/model/chess_universal.rdata",
		"samples/UsingRingML/Chess_End_Game/User Guidt.txt",
		"samples/UsingRingML/classify_demo.ring",
		"samples/UsingRingML/fast_viz_demo.ring",
		"samples/UsingRingML/loader_demo.ring",
		"samples/UsingRingML/mnist/data/mnist_test.csv",
		"samples/UsingRingML/mnist/data/mnist_test_lite.csv",
		"samples/UsingRingML/mnist/mnist_app.ring",
		"samples/UsingRingML/mnist/mnist_dataset.ring",
		"samples/UsingRingML/mnist/mnist_train_split.ring",
		"samples/UsingRingML/mnist/mnist_train_universal.ring",
		"samples/UsingRingML/mnist/model/mnist_model.rdata",
		"samples/UsingRingML/mnist/model/mnist_split_model.rdata",
		"samples/UsingRingML/mnist/model/mnist_universal.rdata",
		"samples/UsingRingML/README.md",
		"samples/UsingRingML/save_load_demo.ring",
		"samples/UsingRingML/Visual_test_model.rdata",
		"samples/UsingRingML/xor_train.ring",
		"samples/UsingRingML/xor_weights.rdata"
	],
	:windowsfiles = 	[

	],
	:linuxfiles = 	[

	],
	:ubuntufiles = 	[

	],
	:fedorafiles = 	[

	],
	:freebsdfiles = 	[

	],
	:macosfiles = 	[

	],
	:windowsringfolderfiles = 	[

	],
	:linuxringfolderfiles = 	[

	],
	:ubunturingfolderfiles = 	[

	],
	:fedoraringfolderfiles = 	[

	],
	:freebsdringfolderfiles = 	[

	],
	:macosringfolderfiles = 	[

	],
	:run = "ring main.ring",
	:windowsrun = "",
	:linuxrun = "",
	:macosrun = "",
	:ubunturun = "",
	:fedorarun = "",
	:setup = "ring setup.ring",
	:windowssetup = "",
	:linuxsetup = "",
	:macossetup = "",
	:ubuntusetup = "",
	:fedorasetup = "",
	:remove = "",
	:windowsremove = "",
	:linuxremove = "",
	:macosremove = "",
	:ubunturemove = "",
	:fedoraremove = ""
]