/**
 * Implements a methods to load neural network models from a file.
 *
 * @author Felix Thielke
 * @author Arne Hasselbring
 */

#include "Model.h"
#ifdef WITH_KERAS_HDF5
#include "Formats/KerasHDF5.h"
#endif
#ifdef WITH_ONNX
#include "Formats/ONNX.h"
#endif
#include "Platform/BHAssert.h"
#include <functional>
#include <utility>

namespace NeuralNetwork
{
  void Node::setDimensions()
  {
    inputDimensions.reserve(inputs.size());
    for(const TensorLocation& l : inputs)
      inputDimensions.push_back(l.layer->nodes[l.nodeIndex].outputDimensions[l.tensorIndex]);
    layer->calcOutputDimensions(*this);
  }

  void InputLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.empty());
    node.outputDimensions.push_back(dimensions);
  }

  void DenseLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    if(node.inputDimensions[0].size() != 1)
      FAIL("Dense layers can currently only be applied to flat tensors. Use a 1x1 convolution if it is really needed.");
    ASSERT(node.inputDimensions[0][0] == weights.dims(0));
    node.outputDimensions.push_back({weights.dims(1)});
  }

  void ActivationLayer::calcOutputDimensions(Node& node) const
  {
    if(node.inputDimensions.size() != 1)
      FAIL("Activation layers must currently have exactly one input tensor.");
    node.outputDimensions = node.inputDimensions;
  }

  void DropoutLayer::calcOutputDimensions(Node& node) const
  {
    node.outputDimensions = node.inputDimensions;
  }

  void FlattenLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    node.outputDimensions.push_back({std::accumulate(node.inputDimensions[0].begin(), node.inputDimensions[0].end(), 1u, std::multiplies<>())});
  }

  void ReshapeLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(std::accumulate(node.inputDimensions[0].begin(), node.inputDimensions[0].end(), 1u, std::multiplies<>()) ==
           std::accumulate(dimensions.begin(), dimensions.end(), 1u, std::multiplies<>()));
    node.outputDimensions.push_back(dimensions);
  }

  void Conv2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][0] >= weights.dims(0));
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][1] >= weights.dims(1));
    ASSERT(node.inputDimensions[0][2] == weights.dims(2));
    node.outputDimensions.push_back({{(node.inputDimensions[0][0] - (padding == PaddingType::valid ? weights.dims(0) - 1 : 0) + strides[0] - 1) / strides[0],
                                      (node.inputDimensions[0][1] - (padding == PaddingType::valid ? weights.dims(1) - 1 : 0) + strides[1] - 1) / strides[1],
                                      weights.dims(3)}});
  }

  void SeparableConv2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][0] >= depthwiseWeights.dims(0));
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][1] >= depthwiseWeights.dims(1));
    ASSERT(node.inputDimensions[0][2] == depthwiseWeights.dims(2));
    ASSERT(node.inputDimensions[0][2] * depthwiseWeights.dims(3) == pointwiseWeights.dims(2));
    node.outputDimensions.push_back({{(node.inputDimensions[0][0] - (padding == PaddingType::valid ? depthwiseWeights.dims(0) - 1 : 0) + strides[0] - 1) / strides[0],
                                      (node.inputDimensions[0][1] - (padding == PaddingType::valid ? depthwiseWeights.dims(1) - 1 : 0) + strides[1] - 1) / strides[1],
                                      pointwiseWeights.dims(3)}});
  }

  void DepthwiseConv2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][0] >= weights.dims(0));
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][1] >= weights.dims(1));
    ASSERT(node.inputDimensions[0][2] == weights.dims(2));
    node.outputDimensions.push_back({{(node.inputDimensions[0][0] - (padding == PaddingType::valid ? weights.dims(0) - 1 : 0) + strides[0] - 1) / strides[0],
                                      (node.inputDimensions[0][1] - (padding == PaddingType::valid ? weights.dims(1) - 1 : 0) + strides[1] - 1) / strides[1],
                                      node.inputDimensions[0][2] * weights.dims(3)}});
  }

  void Cropping2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    node.outputDimensions = node.inputDimensions;
    node.outputDimensions[0][0] -= cropping[TOP] + cropping[BOTTOM];
    node.outputDimensions[0][1] -= cropping[LEFT] + cropping[RIGHT];
  }

  void UpSampling2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    node.outputDimensions = node.inputDimensions;
    node.outputDimensions[0][0] *= size[0];
    node.outputDimensions[0][1] *= size[1];
  }

  void ZeroPadding2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    node.outputDimensions = node.inputDimensions;
    node.outputDimensions[0][0] += padding[TOP] + padding[BOTTOM];
    node.outputDimensions[0][1] += padding[LEFT] + padding[RIGHT];
  }

  void Pooling2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][0] >= kernelSize[0]);
    ASSERT(padding == PaddingType::same || node.inputDimensions[0][1] >= kernelSize[1]);
    node.outputDimensions.push_back({{(node.inputDimensions[0][0] - (padding == PaddingType::valid ? kernelSize[0] - 1 : 0) + strides[0] - 1) / strides[0],
                                      (node.inputDimensions[0][1] - (padding == PaddingType::valid ? kernelSize[1] - 1 : 0) + strides[1] - 1) / strides[1],
                                      node.inputDimensions[0][2]}});
  }

  void GlobalPooling2DLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    ASSERT(node.inputDimensions[0].size() == 3);
    node.outputDimensions.push_back({node.inputDimensions[0][2]});
  }

  void AddLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
#ifndef NDEBUG
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
      ASSERT(node.inputDimensions[i] == node.inputDimensions[0]);
#endif
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void SubtractLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 2);
    ASSERT(node.inputDimensions[1] == node.inputDimensions[0]);
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void MultiplyLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
#ifndef NDEBUG
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
      ASSERT(node.inputDimensions[i] == node.inputDimensions[0]);
#endif
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void AverageLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
#ifndef NDEBUG
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
      ASSERT(node.inputDimensions[i] == node.inputDimensions[0]);
#endif
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void MaximumLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
#ifndef NDEBUG
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
      ASSERT(node.inputDimensions[i] == node.inputDimensions[0]);
#endif
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void MinimumLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
#ifndef NDEBUG
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
      ASSERT(node.inputDimensions[i] == node.inputDimensions[0]);
#endif
    node.outputDimensions.push_back(node.inputDimensions[0]);
  }

  void ConcatenateLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() > 1);
    std::vector<unsigned int> dimensions = node.inputDimensions[0];
    ASSERT(static_cast<std::size_t>(axis >= 0 ? axis : -(axis + 1)) < dimensions.size());
    std::size_t realAxis = axis >= 0 ? axis : dimensions.size() + axis;
    for(std::size_t i = 1; i < node.inputDimensions.size(); ++i)
    {
      ASSERT(node.inputDimensions[i].size() == dimensions.size());
      for(std::size_t j = 0; j < dimensions.size(); ++j)
      {
        if(j == realAxis)
          dimensions[j] += node.inputDimensions[i][j];
        else
          ASSERT(dimensions[j] == node.inputDimensions[i][j]);
      }
    }
    node.outputDimensions.push_back(dimensions);
  }

  void LeakyReluLayer::calcOutputDimensions(Node& node) const
  {
    if(node.inputDimensions.size() != 1)
      FAIL("LeakyReLU layers must currently have exactly one input tensor.");
    node.outputDimensions = node.inputDimensions;
  }

  void EluLayer::calcOutputDimensions(Node& node) const
  {
    if(node.inputDimensions.size() != 1)
      FAIL("ELU layers must currently have exactly one input tensor.");
    node.outputDimensions = node.inputDimensions;
  }

  void ThresholdedReluLayer::calcOutputDimensions(Node& node) const
  {
    if(node.inputDimensions.size() != 1)
      FAIL("ThresholdedReLU layers must currently have exactly one input tensor.");
    node.outputDimensions = node.inputDimensions;
  }

  void SoftmaxLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
    node.outputDimensions = node.inputDimensions;
  }

  void ReluLayer::calcOutputDimensions(Node& node) const
  {
    if(node.inputDimensions.size() != 1)
      FAIL("ReLU layers must currently have exactly one input tensor.");
    node.outputDimensions = node.inputDimensions;
  }

  void BatchNormalizationLayer::calcOutputDimensions(Node& node) const
  {
    ASSERT(node.inputDimensions.size() == 1);
#ifndef NDEBUG
    if(axis >= 0)
    {
      ASSERT(static_cast<std::size_t>(axis) < node.inputDimensions[0].size());
      ASSERT(node.inputDimensions[0][axis] == factor.size());
    }
    else
    {
      ASSERT(static_cast<std::size_t>(-(axis + 1)) < node.inputDimensions[0].size());
      ASSERT(node.inputDimensions[0][node.inputDimensions[0].size() + axis] == factor.size());
    }
#endif
    node.outputDimensions = node.inputDimensions;
  }

  void Model::setInputUInt8(std::size_t index)
  {
    ASSERT(index < inputs.size());
    uint8Inputs.resize(index + 1, false);
    uint8Inputs[index] = true;
  }

  bool Model::isInputUInt8(std::size_t index) const
  {
    return index < uint8Inputs.size() && uint8Inputs[index];
  }

  void Model::load(const std::string& file)
  {
    clear();

#ifdef WITH_KERAS_HDF5
    if(!file.empty() && file.back() == '5')
    {
      KerasHDF5 reader(layers, inputs, outputs);
      reader.read(file);
      return;
    }
#endif
#ifdef WITH_ONNX
    if(file.length() >= 5 && file.substr(file.length() - 5) == ".onnx")
    {
      ONNX reader(layers, inputs, outputs);
      reader.read(file);
      return;
    }
#endif
    FAIL("Unsupported format.");
  }
}
