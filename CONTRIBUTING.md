# Contributing to RingML

Thank you for your interest in contributing to RingML! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## Getting Started

1. **Fork the Repository**: Create your own fork of the RingML repository
2. **Clone Your Fork**: 
   ```bash
   git clone https://github.com/YOUR_USERNAME/ringml-using-ringtensor.git
   cd ringml-using-ringtensor
   ```
3. **Set Up Upstream**:
   ```bash
   git remote add upstream https://github.com/Azzeddine2017/ringml-using-ringtensor.git
   ```

## Development Setup

### Prerequisites

- Ring Programming Language (v1.25 or higher)
- RingTensor Extension (v1.3.2 or higher)
- Git for version control

### Installation

1. Install Ring from [ring-lang.net](http://ring-lang.net)
2. Install RingML:
   ```bash
   ringpm install ringml-using-ringtensor from Azzeddine2017
   ```

### Verify Installation

```bash
ringpm run ringml-using-ringtensor
```

You should see the RingML welcome banner.

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Minimal code example that reproduces the bug
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: Ring version, OS, RingTensor version
- **Error Messages**: Complete error output

### Suggesting Features

Feature requests should include:

- **Use Case**: Why this feature is needed
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you've considered
- **Examples**: Code examples showing the proposed API

### Contributing Code

1. **Create a Branch**: 
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Changes**: Implement your feature or fix
3. **Test**: Ensure all tests pass
4. **Commit**: Write clear commit messages
5. **Push**: Push to your fork
6. **Pull Request**: Open a PR against the main repository

## Coding Standards

### Ring Code Style

- **Indentation**: Use tabs (as per Ring conventions)
- **Naming Conventions**:
  - Classes: `PascalCase` (e.g., `Sequential`, `Dense`)
  - Functions/Methods: `camelCase` (e.g., `forward`, `backward`)
  - Variables: `camelCase` with prefix notation:
    - `o` for objects (e.g., `oModel`)
    - `a` for arrays (e.g., `aGrads`)
    - `c` for strings (e.g., `cFilename`)
    - `n` for numbers (e.g., `nEpochs`)
- **Comments**: Use `#` for single-line comments, document complex logic
- **File Headers**: Include file description and author

Example:
```ring
# File: layers/custom_layer.ring
# Description: Custom layer implementation
# Author: Your Name

class CustomLayer from Layer
    # Layer attributes
    nInputSize
    nOutputSize
    oWeights
    
    func init nIn, nOut
        nInputSize = nIn
        nOutputSize = nOut
        # Initialize weights
        oWeights = new Tensor(nIn, nOut)
        
    func forward oInput
        # Forward pass implementation
        return oOutput
```

### Code Organization

- **One Class Per File**: Each class should be in its own file
- **Logical Grouping**: Group related files in directories (layers, optim, data, etc.)
- **Load Order**: Be mindful of dependencies when using `load`

## Testing Guidelines

### Writing Tests

- Place tests in the `libraries/ringml/tests/` directory
- Name test files with `test_` prefix (e.g., `test_dense_layer.ring`)
- Test both success and failure cases
- Include edge cases

### Test Structure

```ring
load "ringml.ring"

func main
    testDenseLayerForward()
    testDenseLayerBackward()
    see "All tests passed!" + nl

func testDenseLayerForward
    # Test setup
    oLayer = new Dense(10, 5)
    oInput = new Tensor(1, 10)
    
    # Execute
    oOutput = oLayer.forward(oInput)
    
    # Verify
    if oOutput.shape()[2] != 5
        raise("Dense layer output shape incorrect")
    ok
    
    see "✓ Dense layer forward test passed" + nl
```

### Running Tests

```bash
cd libraries/ringml/tests
ring test_your_feature.ring
```

## Pull Request Process

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the changelog

## Documentation

### Documentation Standards

- Write in clear, concise English
- Use code examples to illustrate concepts
- Include parameter descriptions and return values
- Document edge cases and limitations

### API Documentation Format

```ring
# Function: forward
# Description: Performs forward pass through the layer
# Parameters:
#   - oInput: Tensor - Input tensor of shape (batch_size, input_size)
# Returns:
#   - oOutput: Tensor - Output tensor of shape (batch_size, output_size)
# Example:
#   oLayer = new Dense(10, 5)
#   oOutput = oLayer.forward(oInput)
```

### Updating Documentation

When adding new features:

1. Update relevant README files
2. Add entries to API reference
3. Create examples if applicable
4. Update CHANGELOG.md

## Questions?

If you have questions:

- Check existing documentation
- Search closed issues
- Open a new issue with the "question" label
- Contact: azzeddine.remmal@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to RingML!** 🎉
