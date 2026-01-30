# 🧠 RingML: Deep Learning & Transformers for Ring

[![Version](https://img.shields.io/badge/version-1.2.4-blue.svg)](https://github.com/Azzeddine2017/ringml-using-ringtensor)
[![Ring](https://img.shields.io/badge/Ring-1.25+-green.svg)](http://ring-lang.net)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![RingTensor](https://img.shields.io/badge/RingTensor-1.3.3+-red.svg)](https://github.com/Azzeddine2017/ringtensor)

**RingML** is a high-performance, object-oriented Deep Learning framework built for the Ring programming language. It is powered by **RingTensor**, a custom C-extension designed to provide fast, double-precision matrix operations, fused optimizer kernels, and **Flash Attention** mechanisms.

The library offers a PyTorch-like API, adhering to **Jacob's Law** by providing a familiar interface. It has evolved from simple MLPs to supporting state-of-the-art **Transformer Architectures (GPT/BERT)**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Tech Stack](#️-tech-stack--architecture)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🚀 **High Performance**: Critical operations executed in optimized C kernels
- 🧠 **Transformer Support**: Multi-head attention, layer normalization, GELU activation
- 💾 **Memory Efficient**: Zero-copy design with managed pointers
- ⚡ **Dual Execution Modes**: Eager (flexible) and Graph (optimized)
- 🎯 **PyTorch-Like API**: Familiar interface for easy adoption
- 🔧 **Production Ready**: Binary serialization, model save/load
- 🖥️ **Hardware Acceleration**: CPU (OpenMP) and GPU (OpenCL) support
- 📊 **Complete Toolkit**: Data loaders, optimizers, loss functions, utilities

---

## 📦 Installation

### Using RingPM (Recommended)

```bash
ringpm install ringml-using-ringtensor from Azzeddine2017
```

### Requirements

- **Ring Programming Language** (v1.25 or higher) - [Download](http://ring-lang.net)
- **RingTensor Extension** (v1.3.2 or higher) - Installed automatically with RingPM

> **Note**: Ensure `ring_tensor.dll` (Windows) or `libring_tensor.so` (Linux/macOS) is in your execution path.

### Verify Installation

```bash
ringpm run ringml-using-ringtensor
```

You should see the RingML welcome banner.

---

## 🚀 Quick Start

### Your First Neural Network (XOR Problem)

```ring
load "ringml.ring"

# 1. Create Data
aInputs = [[0,0], [0,1], [1,0], [1,1]]
aTargets = [[0], [1], [1], [0]]

# Convert to tensors
aDataset = []
for i = 1 to len(aInputs)
    oInput = new Tensor(1, 2)
    oInput.setVal(1, 1, aInputs[i][1])
    oInput.setVal(1, 2, aInputs[i][2])
    
    oTarget = new Tensor(1, 1)
    oTarget.setVal(1, 1, aTargets[i][1])
    
    add(aDataset, [oInput, oTarget])
next

# 2. Build Model
oModel = new Sequential()
oModel.add(new Dense(2, 4))
oModel.add(new Tanh)
oModel.add(new Dense(4, 1))
oModel.add(new Sigmoid)

# 3. Train
oCriterion = new MSELoss
oOptimizer = new SGD(0.1)

oModel.train()
for epoch = 1 to 1000
    for i = 1 to len(aDataset)
        oPred = oModel.forward(aDataset[i][1])
        loss = oCriterion.forward(oPred, aDataset[i][2])
        
        oGrad = oCriterion.backward(oPred, aDataset[i][2])
        oModel.backward(oGrad)
        oModel.update_weights(oOptimizer)
    next
next

# 4. Test
oModel.evaluate()
for i = 1 to len(aDataset)
    oPred = oModel.forward(aDataset[i][1])
    see "Input: [" + aInputs[i][1] + ", " + aInputs[i][2] + "] "
    see "-> Output: " + oPred.getVal(1,1) + nl
next
```

**Output**:
```
Input: [0, 0] -> Output: 0.02
Input: [0, 1] -> Output: 0.98
Input: [1, 0] -> Output: 0.97
Input: [1, 1] -> Output: 0.03
```

---

## 📚 Documentation

### Getting Started
- 📖 [Getting Started Guide](docs/GETTING_STARTED.md) - Complete tutorial for beginners
- 🏗️ [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- 📘 [API Reference](docs/API_REFERENCE.md) - Complete API documentation

### Guides
- ⚡ [Performance Optimization](docs/PERFORMANCE.md) - Speed up training and inference
- 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- 🤝 [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

### Project Info
- 📝 [Changelog](CHANGELOG.md) - Version history and updates

---

## 💡 Examples

### Basic Examples

| Example | Description | Difficulty |
|---------|-------------|------------|
| [xor_train.ring](samples/UsingRingML/xor_train.ring) | Binary classification (XOR) | ⭐ Beginner |
| [classify_demo.ring](samples/UsingRingML/classify_demo.ring) | Multi-class classification | ⭐ Beginner |
| [save_load_demo.ring](samples/UsingRingML/save_load_demo.ring) | Model serialization | ⭐ Beginner |
| [loader_demo.ring](samples/UsingRingML/loader_demo.ring) | Data loading utilities | ⭐ Beginner |

### Advanced Projects

#### 🔢 MNIST Digit Recognition
**Path**: `samples/UsingRingML/mnist/`

Train a neural network to recognize handwritten digits.

```bash
cd samples/UsingRingML/mnist
ring mnist_train_universal.ring
```

#### ♟️ Chess End-Game Prediction
**Path**: `samples/UsingRingML/Chess_End_Game/`

Multi-class classification (18 classes) for chess endgame positions.

```bash
cd samples/UsingRingML/Chess_End_Game
ring chess_train_universal.ring
```

#### 🌐 English-Arabic Translation (Transformer)
**Path**: `samples/UsingRingML/train_translate_bidir/`

Complete Transformer-based translation engine with:
- Bidirectional translation (EN ↔ AR)
- Multi-head attention
- Graph mode execution
- Custom tokenizer

```bash
cd samples/UsingRingML/train_translate_bidir
ring main.ring
```

---

## 🛠️ Tech Stack & Architecture

### Core Technology

- **Language**: Ring (v1.25+)
- **Backend**: RingTensor C Extension
- **Acceleration**: OpenMP (CPU), OpenCL (GPU)
- **Precision**: Double (FP64)

### Execution Modes

#### Eager Mode (Default)
- Immediate execution
- Easy debugging
- Flexible control flow

#### Graph Mode (Optimized)
- JIT-compiled computation graph
- 2-3x faster for large models
- Zero interpreter overhead

```ring
# Enable graph mode
oInput = new Tensor(32, 512)
oInput.setGraphMode(true)
oOutput = oModel.compile(oInput)

# Fast execution
oResult = graph_run(oOutput.pGraph, oRealInput.pData)
```

### Hardware Support

| Platform | CPU | GPU |
|----------|-----|-----|
| Windows | ✅ OpenMP | ✅ OpenCL |
| Linux | ✅ OpenMP | ✅ OpenCL |
| macOS | ✅ OpenMP | ✅ OpenCL |

**Supported GPUs**: Intel HD, NVIDIA, AMD (via OpenCL)

---

## ⚡ Performance

### Benchmarks (Intel i7-10700K, 32GB RAM)

| Model | Batch Size | Mode | Throughput |
|-------|------------|------|------------|
| MLP (784→128→10) | 64 | Eager | 15k samples/sec |
| MLP (784→128→10) | 64 | Graph | 25k samples/sec |
| Transformer (512d, 8h) | 32 | Eager | 120 samples/sec |
| Transformer (512d, 8h) | 32 | Graph | 320 samples/sec |

**Speedup**: Graph mode provides **1.7-2.7x** improvement.

### Why RingML is Fast

- ✅ **Fused Kernels**: Optimizers update weights directly in C memory
- ✅ **Zero-Copy**: No data marshalling between Ring and C
- ✅ **Flash Attention**: Optimized attention mechanism
- ✅ **Multi-Threading**: OpenMP parallelization
- ✅ **GPU Dispatch**: Automatic GPU usage for large operations

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests
5. Submit a pull request

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features (layers, optimizers, etc.)
- 📚 Documentation improvements
- 🧪 Additional tests
- 💡 Example projects

---

## 📝 License

RingML is open source software licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Azzeddine Remmal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Author

**Azzeddine Remmal**
- Email: azzeddine.remmal@gmail.com
- GitHub: [@Azzeddine2017](https://github.com/Azzeddine2017)

---

## 🙏 Acknowledgments

- Ring Programming Language by Mahmoud Fayed
- Inspired by PyTorch and TensorFlow
- Community contributors

---

## 📞 Support

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Azzeddine2017/ringml-using-ringtensor/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Azzeddine2017/ringml-using-ringtensor/discussions)
- 📧 **Email**: azzeddine.remmal@gmail.com

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ using Ring Programming Language

</div>