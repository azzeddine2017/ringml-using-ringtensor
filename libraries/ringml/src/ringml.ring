# File: src/ringml.ring
# Description: Main entry point
# Author: Azzeddine Remmal
# Date: 2025-12-15
# Version: 1.1.0
# License: MIT License

load "stdlib.ring"
load "csvlib.ring"
load "jsonlib.ring"
load "ringtensor.ring"
load "utils/functions.ring"
load "core/tensor.ring"
load "utils/serializer.ring"
load "data/dataset.ring"
load "data/datasplitter.ring"
load "data/universaldataset.ring"
load "utils/visualizer.ring"
load "layers/layer.ring"
load "layers/dense.ring"
load "layers/activation.ring"
load "layers/softmax.ring"
load "layers/dropout.ring" 
load "model/sequential.ring"
load "loss/mse.ring"
load "loss/crossentropy.ring"
load "optim/sgd.ring"
load "optim/adam.ring"         
load "utils/Styler.ring"

oStyl = new Styler()

func RingMLVersion
    see " RingML v - (1.1.0)"

func raise(cMessage)
    oStyl.Error(cMessage)
    Bye	