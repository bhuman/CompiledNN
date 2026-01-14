/**
 * @file ONNX.h
 *
 * This file declares a class that reads ONNX models.
 *
 * @author Arne Hasselbring
 */

#pragma once

#include "../Model.h"
#include "../Tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx
{
  class NodeProto;
}

namespace NeuralNetwork
{
  class ONNX
  {
  private:
    std::vector<std::unique_ptr<Layer>>& layers;
    std::vector<TensorLocation>& inputs;
    std::vector<TensorLocation>& outputs;

    using WeightMap = std::unordered_map<std::string, Tensor<float, 1>>;
    using ShapeMap = std::unordered_map<std::string, Tensor<std::int64_t, 1>>;
    using VariableMap = std::unordered_map<std::string, TensorLocation>; // TODO: maybe we also need the hwc/chw info

    WeightMap weights;
    ShapeMap shapes;
    VariableMap variables; // TODO: maybe we also need the hwc/chw info

    // If there are models that use a newer version, we must check if the newer opset version changes any of the relevant operators.
    // If there are only irrelevant changes, this number can safely be set to the newest version.
    // Otherwise, the new versions of the operators have to be implemented.
    static constexpr std::int64_t maxDefaultOpsetVersion = 15 /* 13 */;

    void doAdd(const onnx::NodeProto&);
    void doBatchNormalization(const onnx::NodeProto&);
    void doCast(const onnx::NodeProto&);
    void doClip(const onnx::NodeProto&);
    void doConcat(const onnx::NodeProto&);
    void doConv(const onnx::NodeProto&);
    void doGather(const onnx::NodeProto&);
    void doMatMul(const onnx::NodeProto&);
    void doMaxPool(const onnx::NodeProto&);
    void doMul(const onnx::NodeProto&);
    void doPad(const onnx::NodeProto&);
    void doRelu(const onnx::NodeProto&);
    void doReshape(const onnx::NodeProto&);
    void doShape(const onnx::NodeProto&);
    void doSlice(const onnx::NodeProto&);
    void doSub(const onnx::NodeProto&);
    void doTranspose(const onnx::NodeProto&);

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
