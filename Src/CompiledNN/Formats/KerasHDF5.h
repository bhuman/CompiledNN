/**
 * @file KerasHDF5.h
 *
 * This file declares a class that reads Keras HDF5 models.
 *
 * @author Arne Hasselbring
 */

#pragma once

#include "../Model.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace NeuralNetwork
{
  class KerasHDF5
  {
  public:
    /// getWeights with layer name as first parameter
    using GetWeightsFuncType = std::function<void(const std::string&, const std::string&, std::vector<float>&, std::vector<unsigned int>&)>;
    /// getWeights with bound layer name
    using GetWeights2FuncType = std::function<void(const std::string&, std::vector<float>&, std::vector<unsigned int>&)>;

  private:
    /**
     * Parses a model from a JSON description.
     */
    void parseJSONModel(In& stream, const std::string& fileName, const GetWeightsFuncType& getWeights, unsigned long kerasVersion);

    std::vector<std::unique_ptr<Layer>>& layers;
    std::vector<TensorLocation>& inputs;
    std::vector<TensorLocation>& outputs;

  public:
    /**
     * Constructor.
     */
    KerasHDF5(std::vector<std::unique_ptr<Layer>>& layers, std::vector<TensorLocation>& inputs, std::vector<TensorLocation>& outputs) :
      layers(layers),
      inputs(inputs),
      outputs(outputs)
    {}

    /**
     * Reads a neural network model from the given file in the native Keras HDF5 format.
     */
    void read(const std::string& file);
  };
}
