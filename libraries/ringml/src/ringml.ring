# File: src/ringml.ring
# Description: Main entry point
# Author: Azzeddine Remmal


load "stdlib.ring"
load "csvlib.ring"
load "jsonlib.ring"
load "ringtensor.ring"
load "AlQalam.ring"

load "utils/Styler.ring"
oStyl = new Styler()

load "nlp/tokenizer.ring"

load "utils/functions.ring"
load "utils/visualizer.ring"
load "utils/RogueutilTransformerDashboard.ring"
load "utils/GUI_dashboard.ring"
load "utils/trainer.ring"
load "utils/GUI_trainer.ring"
load "utils/GUI_trainer_graph.ring"

load "core/tensor.ring"

load "model/sequential.ring"
load "model/transformer_block.ring"

load "serialize/BlockSerializer.ring"
load "serialize/ModelSerializer.ring"

load "data/dataset.ring"
load "data/datasplitter.ring"
load "data/universaldataset.ring"
load "data/BatchDataLoader.ring"
load "data/DataLoader.ring"
load "data/StandardDataset.ring"

load "layers/embedding.ring"
load "layers/layernorm.ring"
load "layers/layer.ring"
load "layers/dense.ring"
load "layers/activation.ring"
load "layers/softmax.ring"
load "layers/dropout.ring" 
load "layers/multihead_attention.ring"
load "layers/linear_attention.ring"

load "loss/mse.ring"
load "loss/crossentropy.ring"
load "optim/sgd.ring"
load "optim/adam.ring"  

load "Montoring/GradientDiagnostics.ring"
load "utils/SmartScheduler.ring"     





func RingMLVersion
    see " RingML v - (1.2.0)"

func raise(cMessage)
    oStyl.Error(cMessage)
    Bye	

func info(cMessage)
    oStyl.Info(cMessage)

func warning(cMessage)
    oStyl.Warning(cMessage)

func success(cMessage)
    oStyl.Success(cMessage)

func error(cMessage)
    oStyl.Error(cMessage)      