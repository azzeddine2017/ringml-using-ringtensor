

load "ringml.ring"

load "../AdamModel2.ring"
load "../Inference.ring"
load "../BiDirectionalDataset.ring"

load "testBiDirectionalDataset.ring"
load "testDataLoaderTargets.ring"
load "testDataLoaderDetailed.ring"
load "testClippingDirectly.ring"
load "testAdamModel2.ring"
load "testAdamModel2WithDiagnostics.ring"
load "testFullTrainingPipeline.ring"
load "testGradientCollection.ring"
load "testTransformerBlock.ring"
load "testGELU.ring"
load "debugGradientCollection.ring"


see "num cores :" + tensor_get_cores()
tensor_set_threads(2)

decimals(8)

func main
	testTransformerBlock()
	testAdamModel2()
	testClippingDirectly()
	testDataLoaderDetailed()
	testDataLoaderTargets()
	testBiDirectionalDataset()
	testGradientCollection()
	testAdamModel2WithDiagnostics()
	testFullTrainingPipeline()
	debugGradientCollection()
	testGELU()

