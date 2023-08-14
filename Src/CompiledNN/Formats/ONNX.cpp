/**
 * @file ONNX.cpp
 *
 * This file implements a class that reads ONNX models.
 *
 * @author Arne Hasselbring
 */

#include "ONNX.h"
#include "Platform/BHAssert.h"
#include <onnx.pb.h>
#include <fstream>
#include <iostream>

namespace NeuralNetwork
{
  void ONNX::read(const std::string &file)
  {
    std::vector<char> binary;
    {
      std::ifstream f(file, std::ifstream::binary);
      if(!f.is_open())
        FAIL("Model \"" << file << "\" could not be opened.");
      f.seekg(0, std::ifstream::end);
      binary.resize(f.tellg());
      f.seekg(0);
      f.read(binary.data(), binary.size());
      f.close();
    }

    onnx::ModelProto model;
    model.ParseFromArray(binary.data(), binary.size());

    if(model.ir_version() < onnx::IR_VERSION_2019_1_22)
      FAIL("Unsupported ONNX IR version: " << model.ir_version() << ".");
    // ignore producer_name
    // ignore producer_version
    // ignore domain
    // ignore model_version
    // ignore doc_string
    ASSERT(model.has_graph());
    const auto& graph = model.graph();
    std::unordered_map<std::string, std::int64_t> opsetVersions;
    for(const onnx::OperatorSetIdProto& opset : model.opset_import())
      opsetVersions[opset.domain()] = opset.version();
    if(opsetVersions.find("") == opsetVersions.end())
      FAIL("No version defined for the default operator set.");
    // ignore metadata_props
    if(!model.training_info().empty())
      FAIL("Non-empty training info is not supported.");
    if(!model.functions().empty())
      FAIL("Non-empty functions is not supported.");

    std::unordered_map<std::string, Tensor<float, 1>> weights;
    std::unordered_map<std::string, Tensor<std::int64_t, 1>> shapes;
    std::unordered_map<std::string, TensorLocation> variables;

    for(const onnx::TensorProto& initializer : graph.initializer())
    {
      ASSERT(!initializer.name().empty());
      if(initializer.has_segment())
        FAIL("Segmented initializers are not supported.");
      ASSERT(initializer.data_location() == onnx::TensorProto::DEFAULT);

      if(initializer.data_type() == onnx::TensorProto::FLOAT)
      {
        Tensor<float, 1>& tensor = weights[initializer.name()];
        {
          std::vector<unsigned int> dims;
          dims.reserve(initializer.dims().size());
          for(auto dim : initializer.dims())
            dims.push_back(dim);
          tensor.reshape(dims);
        }

        if(tensor.size() * sizeof(float) != initializer.raw_data().size())
          FAIL("Only raw_data initializers are supported.");
        std::memcpy(tensor.data(), initializer.raw_data().data(), initializer.raw_data().size()); // TODO: do we need to reorder due to channels first/last?
        // TODO: raw_data is always little endian, so endianess must be changed on BE architectures. (irrelevant as none are supported)
      }
      else if(initializer.data_type() == onnx::TensorProto::INT64)
      {
        Tensor<std::int64_t, 1>& tensor = shapes[initializer.name()];
        {
          std::vector<unsigned int> dims;
          dims.reserve(initializer.dims().size());
          for(auto dim : initializer.dims())
            dims.push_back(dim);
          tensor.reshape(dims);
        }

        if(tensor.size() * sizeof(std::int64_t) != initializer.raw_data().size())
          FAIL("Only raw_data initializers are supported.");
        std::memcpy(tensor.data(), initializer.raw_data().data(), initializer.raw_data().size()); // TODO: LE to host?
      }
      else
        FAIL("Only FLOAT and INT64 initializers are supported.");
    }

    // Sparse initializers are not suppoted at the moment (would be of type SparseTensorProto).
    if(!graph.sparse_initializer().empty())
      FAIL("Sparse initializers are not supported at the moment.");

    for(const onnx::ValueInfoProto& input : graph.input())
    {
      ASSERT(!input.name().empty());
      if(weights.find(input.name()) != weights.end())
        continue; // TODO: we could check shapes/types to match
      ASSERT(input.has_type());
      if(input.type().value_case() != onnx::TypeProto::kTensorType || !input.type().has_tensor_type())
        FAIL("All inputs must be tensors.");
      if(input.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT)
        FAIL("Only float inputs are supported (from ONNX perspective).");
      if(!input.type().tensor_type().has_shape())
        FAIL("All input tensors must have a shape.");
      if(input.type().tensor_type().shape().dim().size() <= 1)
        FAIL("All input tensors must have at least one batch axis and one real axis."); // TODO: In theory, we could support "0" data dimensions.
      if(input.type().tensor_type().shape().dim()[0].value_case() != onnx::TensorShapeProto::Dimension::kDimParam && !(input.type().tensor_type().shape().dim()[0].value_case() == onnx::TensorShapeProto::Dimension::kDimValue && input.type().tensor_type().shape().dim()[0].dim_value() == 1))
        FAIL("All input tensors must have a first (batch) dimension that is either 1 or variable.");
      // ignore input.type.tensor_type.shape.dim[i].denotation
      // ignore input.type.denotation
      // ignore input.doc_string

      std::unique_ptr<InputLayer> layer = std::make_unique<InputLayer>();
      layer->dimensions.resize(input.type().tensor_type().shape().dim().size() - 1);
      for(std::size_t i = 0; i < layer->dimensions.size(); ++i)
      {
        ASSERT(input.type().tensor_type().shape().dim()[i + 1].value_case() == onnx::TensorShapeProto::Dimension::kDimValue);
        layer->dimensions[i] = input.type().tensor_type().shape().dim()[i + 1].dim_value();
        ASSERT(layer->dimensions[i] > 0);
      }

      // Special case for input layers: There is a "virtual" node without inputs.
      layer->nodes.emplace_back(layer.get());
      layer->nodes.back().outputDimensions.push_back(layer->dimensions);
      layer->nodes.back().outputs.emplace_back(layer.get(), 0, 0);

      layers.push_back(std::move(layer));
      inputs.emplace_back(layers.back().get(), 0, 0);

      variables.emplace(input.name(), inputs.back());
    }

    // ONNX guarantees that the nodes are topologically sorted.
    for(const onnx::NodeProto& node : graph.node())
    {
      // node.output is handled below.
      // ignore node.name
      // ignore node.doc_string

      if(!node.domain().empty())
        FAIL("Only the default opset (and not even that) is supported yet, but some node tries to use " << node.domain() << ".");

      if(node.op_type() == "Add")
      {
        // 2 inputs
        // 0 attributes
        // 1 output
      }
      else if(node.op_type() == "AveragePool")
      {
        // 1 input
        // attributes: auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides
        // 1 output
      }
      else if(node.op_type() == "BatchNormalization")
      {
      }
      else if(node.op_type() == "Concat")
      {
      }
      else if(node.op_type() == "Conv")
      {
        ASSERT(node.input().size() == 2 || node.input().size() == 3);
        variables.find(node.input()[0]);
        weights.find(node.input()[1]);
        if(node.input().size() > 2)
          weights.find(node.input()[2]);
        // attributes: "dilation"
      }
      unsigned int i = 0;
      for(const std::string& output : node.output())
      {
        // TODO: assert not output in variables? There might be models for which this is the case, but we probably don't want them
        variables.emplace(output, TensorLocation(layers.back().get(), 0, i++));
      }
    }

    for(const onnx::ValueInfoProto& output : graph.output())
    {
      ASSERT(!output.name().empty());
      const auto outputVariable = variables.find(output.name());
      ASSERT(outputVariable != variables.end());
      ASSERT(output.has_type());
      if(output.type().value_case() != onnx::TypeProto::kTensorType || !output.type().has_tensor_type())
        FAIL("All outputs must be tensors.");
      if(output.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT)
        FAIL("Only float outputs are supported (from ONNX perspective).");
      if(!output.type().tensor_type().has_shape())
        FAIL("All output tensors must have a shape.");
      if(output.type().tensor_type().shape().dim().size() <= 1)
        FAIL("All output tensors must have at least one batch axis and one real axis."); // TODO: In theory, we could support "0" data dimensions.
      if(output.type().tensor_type().shape().dim()[0].value_case() != onnx::TensorShapeProto::Dimension::kDimParam && !(output.type().tensor_type().shape().dim()[0].value_case() == onnx::TensorShapeProto::Dimension::kDimValue && output.type().tensor_type().shape().dim()[0].dim_value() == 1))
        FAIL("All output tensors must have a first (batch) dimension that is either 1 or variable.");
#ifndef NDEBUG
      const auto& dims1 = outputVariable->second.layer->nodes[outputVariable->second.nodeIndex].outputDimensions[outputVariable->second.tensorIndex];
      const auto& dims2 = output.type().tensor_type().shape().dim();
      ASSERT(dims1.size() == static_cast<std::size_t>(dims2.size() - 1));
      for(std::size_t i = 0; i < dims1.size(); ++i)
      {
        ASSERT(dims2[i + 1].value_case() == onnx::TensorShapeProto::Dimension::kDimValue);
        ASSERT(dims1[i] == dims2[i + 1].dim_value());
      }
#endif
      // ignore output.type.denotation
      // ignore output.doc_string

      outputs.push_back(outputVariable->second);
    }
  }
}
