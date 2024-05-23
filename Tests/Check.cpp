/**
 * @file Check.cpp
 *
 * This file contains a program to check whether SimpleNN and CompiledNN yield the same result on a model.
 *
 * @author Felix Thielke
 * @author Arne Hasselbring
 */

#include "CompiledNN/CompiledNN.h"
#include "CompiledNN/Model.h"
#include "CompiledNN/SimpleNN.h"
#include "CompiledNN/Tensor.h"
#include "Platform/BHAssert.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
  if(argc < 2 || argc > 4)
  {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "Check") << " <path to model> [<min input> [<max input>]]\n";
    return EXIT_FAILURE;
  }

  NeuralNetwork::Model model(argv[1]);

  float minInput = -1.f, maxInput = 1.f;
  if(argc > 2)
    minInput = std::strtof(argv[2], nullptr);
  if(argc > 3)
    maxInput = std::strtof(argv[3], nullptr);

  std::vector<NeuralNetwork::TensorXf> testInputs(model.getInputs().size());
  std::vector<NeuralNetwork::TensorXf> testOutputs(model.getOutputs().size());

  // A deterministic seed is okay here.
  std::mt19937 generator;
  std::uniform_real_distribution<float> inputDistribution(minInput, maxInput);

  const std::vector<NeuralNetwork::TensorLocation>& inputs = model.getInputs();
  for(std::size_t i = 0; i < testInputs.size(); ++i)
  {
    testInputs[i].reshape(inputs[i].layer->nodes[inputs[i].nodeIndex].outputDimensions[inputs[i].tensorIndex]);
    float* p = testInputs[i].data();
    for(std::size_t n = testInputs[i].size(); n; --n)
      *(p++) = inputDistribution(generator);
  }

  // Set compilation settings.
  NeuralNetwork::CompilationSettings settings;
  settings.useExpApproxInSigmoid = false;
  settings.useExpApproxInTanh = false;
  settings.debug = true;

  // Apply the simple NN and compare the output of each node to what the compiled NN calculates.
  std::vector<NeuralNetwork::TensorXf> testInputsCopied(testInputs);
  NeuralNetwork::SimpleNN::apply(testInputsCopied, testOutputs, model, [&settings](const NeuralNetwork::Node& node, const std::vector<const NeuralNetwork::TensorXf*>& inputs, const std::vector<NeuralNetwork::TensorXf*>& outputs)
  {
    // Compile a net consisting only of this single node.
    NeuralNetwork::CompiledNN compiledNN;
    compiledNN.compile(node, settings);
    ASSERT(inputs.size() == compiledNN.numOfInputs());
    ASSERT(outputs.size() == compiledNN.numOfOutputs());

    // Set inputs of the compiled NN to the same input that the simple NN got.
    for(std::size_t i = 0; i < inputs.size(); ++i)
      compiledNN.input(i).copyFrom(*inputs[i]);

    // Do the test.
    compiledNN.apply();

    // Compute and output error.
    switch(node.layer->type)
    {
      case NeuralNetwork::LayerType::input:
        std::cout << "Input";
        break;
      case NeuralNetwork::LayerType::dense:
        std::cout << "Dense";
        break;
      case NeuralNetwork::LayerType::activation:
        std::cout << "Activation";
        break;
      case NeuralNetwork::LayerType::dropout:
        std::cout << "Dropout";
        break;
      case NeuralNetwork::LayerType::flatten:
        std::cout << "Flatten";
        break;
      case NeuralNetwork::LayerType::reshape:
        std::cout << "Reshape";
        break;
      case NeuralNetwork::LayerType::conv1D:
        std::cout << "Conv1D";
        break;
      case NeuralNetwork::LayerType::conv2D:
        std::cout << "Conv2D";
        break;
      case NeuralNetwork::LayerType::separableConv2D:
        std::cout << "SeparableConv2D";
        break;
      case NeuralNetwork::LayerType::depthwiseConv2D:
        std::cout << "DepthwiseConv2D";
        break;
      case NeuralNetwork::LayerType::cropping2D:
        std::cout << "Cropping2D";
        break;
      case NeuralNetwork::LayerType::upSampling2D:
        std::cout << "UpSampling2D";
        break;
      case NeuralNetwork::LayerType::zeroPadding1D:
        std::cout << "ZeroPadding1D";
        break;
      case NeuralNetwork::LayerType::zeroPadding2D:
        std::cout << "ZeroPadding2D";
        break;
      case NeuralNetwork::LayerType::maxPooling1D:
        std::cout << "MaxPooling1D";
        break;
      case NeuralNetwork::LayerType::averagePooling1D:
        std::cout << "AveragePooling1D";
        break;
      case NeuralNetwork::LayerType::maxPooling2D:
        std::cout << "MaxPooling2D";
        break;
      case NeuralNetwork::LayerType::averagePooling2D:
        std::cout << "AveragePooling2D";
        break;
      case NeuralNetwork::LayerType::globalMaxPooling2D:
        std::cout << "GlobalMaxPooling2D";
        break;
      case NeuralNetwork::LayerType::globalAveragePooling2D:
        std::cout << "GlobalAveragePooling2D";
        break;
      case NeuralNetwork::LayerType::add:
        std::cout << "Add";
        break;
      case NeuralNetwork::LayerType::subtract:
        std::cout << "Subtract";
        break;
      case NeuralNetwork::LayerType::multiply:
        std::cout << "Multiply";
        break;
      case NeuralNetwork::LayerType::average:
        std::cout << "Average";
        break;
      case NeuralNetwork::LayerType::maximum:
        std::cout << "Maximum";
        break;
      case NeuralNetwork::LayerType::minimum:
        std::cout << "Minimum";
        break;
      case NeuralNetwork::LayerType::concatenate:
        std::cout << "Concatenate";
        break;
      case NeuralNetwork::LayerType::leakyRelu:
        std::cout << "LeakyReLU";
        break;
      case NeuralNetwork::LayerType::elu:
        std::cout << "ELU";
        break;
      case NeuralNetwork::LayerType::thresholdedRelu:
        std::cout << "ThresholdedReLU";
        break;
      case NeuralNetwork::LayerType::softmax:
        std::cout << "Softmax";
        break;
      case NeuralNetwork::LayerType::relu:
        std::cout << "ReLU";
        break;
      case NeuralNetwork::LayerType::batchNormalization:
        std::cout << "BatchNormalization";
        break;
      default:
        FAIL("Implement this.");
    }

    std::cout << "Layer error (SimpleNN vs CompiledNN):";
    if(outputs.size() == 1)
      std::cout << " rel " << outputs[0]->maxRelError(compiledNN.output(0)) << ", abs " << outputs[0]->maxAbsError(compiledNN.output(0)) << '\n';
    else
    {
      std::cout << '\n';
      for(std::size_t i = 0; i < outputs.size(); ++i)
        std::cout << "    rel " << outputs[i]->maxRelError(compiledNN.output(i)) << ", abs " << outputs[i]->maxAbsError(compiledNN.output(i)) << '\n';
    }
  });

  // Test whole network.
  NeuralNetwork::CompiledNN compiledNN;
  compiledNN.compile(model, settings);
  ASSERT(compiledNN.numOfInputs() == testInputs.size());
  for(std::size_t i = 0; i < testInputs.size(); ++i)
    compiledNN.input(i).copyFrom(testInputs[i]);
  compiledNN.apply();
  ASSERT(compiledNN.numOfOutputs() == testOutputs.size());
  std::cout << "Total error (SimpleNN vs CompiledNN):";
  if(testOutputs.size() == 1)
    std::cout << " rel " << testOutputs[0].maxRelError(compiledNN.output(0)) << ", abs " << testOutputs[0].maxAbsError(compiledNN.output(0)) << '\n';
  else
  {
    std::cout << std::endl;
    for(std::size_t i = 0; i < testOutputs.size(); ++i)
      std::cout << "    rel " << testOutputs[i].maxRelError(compiledNN.output(i)) << ", abs " << testOutputs[i].maxAbsError(compiledNN.output(i)) << '\n';
  }

  return EXIT_SUCCESS;
}
