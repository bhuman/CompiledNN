/**
 * @file ONNX.h
 *
 * This file declares a class that reads ONNX models.
 *
 * @author Arne Hasselbring
 */

#pragma once

#include "../Model.h"
#include <memory>
#include <string>
#include <vector>

namespace NeuralNetwork
{
  class ONNX
  {
  private:
    std::vector<std::unique_ptr<Layer>>& layers;
    std::vector<TensorLocation>& inputs;
    std::vector<TensorLocation>& outputs;

  public:
    /**
     * Constructor.
     */
    ONNX(std::vector<std::unique_ptr<Layer>>& layers, std::vector<TensorLocation>& inputs, std::vector<TensorLocation>& outputs) :
      layers(layers),
      inputs(inputs),
      outputs(outputs)
    {}

    /**
     * Reads a neural network model from the given file in the ONNX format.
     */
    void read(const std::string& file);
  };
}
