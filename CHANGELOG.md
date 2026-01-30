# Changelog

All notable changes to the RingML project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.4] - 2026-01-30

### Added
- Comprehensive documentation suite
- API Reference documentation
- Architecture guide
- Getting Started tutorial
- Contributing guidelines
- Performance optimization guide
- Troubleshooting guide

### Changed
- Updated main README.md with enhanced structure
- Standardized version numbers across all documentation
- Improved examples documentation

### Fixed
- Version inconsistencies between documentation files

## [1.2.1] - Previous Release

### Added
- Transformer architecture support
- Flash Attention mechanisms
- Graph mode execution
- Binary serialization improvements

### Changed
- Performance optimizations in RingTensor backend
- Enhanced memory management

## [1.2.0] - Previous Release

### Added
- Multi-head attention layers
- Layer normalization
- Dropout regularization
- Adam optimizer with fused kernels
- UniversalDataset for flexible data loading

### Changed
- Refactored Sequential model API
- Improved gradient clipping utilities

## [1.1.0] - Previous Release

### Added
- Basic transformer block implementation
- Embedding layers
- Custom dataset support
- Model serialization (save/load)

### Changed
- Enhanced DataLoader with batching strategies
- Optimized matrix operations

## [1.0.0] - Initial Release

### Added
- Core tensor operations via RingTensor
- Sequential model API
- Dense layers
- Activation functions (Tanh, Sigmoid, ReLU, GELU)
- Softmax layer
- Loss functions (MSE, CrossEntropy)
- SGD optimizer
- Basic data loading utilities
- XOR and classification examples

---

## Version Support

- **Ring Language**: v1.25+
- **RingTensor Extension**: v1.3.2+
- **Dependencies**: stdlib (1.0.22+), csvlib (1.0.7+), jsonlib (1.0.4+), AlQalam (1.0.0+)
