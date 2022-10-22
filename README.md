# CompiledNN: A JIT Compiler for Neural Network Inference

[![Build Status](https://travis-ci.org/bhuman/CompiledNN.svg?branch=master)](https://travis-ci.org/bhuman/CompiledNN)

## Features

- compiles Keras HDF5 models into machine code
- generates single-threaded code for x86/64 processors with SSSE3/SSE4

## Dependencies

- HDF5 (C bindings only)

## Compiling

CompiledNN can be compiled into a library via CMake:

```bash
mkdir build
cd build
cmake ..
make
make install
```

Another way to integrate CompiledNN is to add it (and its dependency [AsmJit](https://github.com/asmjit/asmjit)) as source files to your project.

## Supported layers

- Core
  - Dense
  - Activation
    - relu
    - tanh (approximated)
    - sigmoid (approximated)
    - hard_sigmoid
    - linear
    - softmax (approximated)
    - elu (approximated)
    - selu (approximated)
    - softsign
  - Dropout
  - Flatten
  - Reshape (does not support dimension inference, i.e. specifying -1 as dimension is not allowed)
- Convolutional
  - Conv2D (only with `dilation_rate=1`)
  - SeparableConv2D (only with `dilation_rate=1` and `depth_multiplier=1`)
  - DepthwiseConv2D (only with `dilation_rate=1`, `depth_multiplier=1`, `use_bias=False` and `activation=None`)
  - Cropping2D
  - UpSampling2D (only with `interpolation=nearest`, number of channels must be at most 32/64 and divisible by 4)
  - ZeroPadding2D (number of channels per row must be divisible by 4)
- Pooling
  - MaxPooling2D
  - AveragePooling2D
  - GlobalMaxPooling2D (at most 28/60 channels)
  - GlobalAveragePooling2D (at most 28/60 channels)
- Merge
  - Add
  - Subtract
  - Multiply
  - Average
  - Maximum
  - Minimum
  - Concatenate (number of channels after the concatenation axis must be divisible by 4 for each input)
- Advanced Activations
  - LeakyReLU
  - ELU
  - ThresholdedReLU
  - Softmax (only for flat tensors)
  - ReLU
- Normalization
  - BatchNormalization (only for flat tensors or channel dimension)

## Example

```cpp
#include <CompiledNN/Model.h>
#include <CompiledNN/CompiledNN.h>

using namespace NeuralNetwork;

int main()
{
  Model model;
  model.load("model.h5");
  // Optionally, indicate which input tensors should be converted from unsigned chars to floats in the beginning.
  // model.setInputUInt8(0);
  CompiledNN nn;
  nn.compile(model);
  // ... fill nn.input(i) with data
  nn.apply();
  // ... obtain the results from nn.output(i)
  return 0;
}
```
