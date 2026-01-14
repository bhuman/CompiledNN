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

  static const onnx::AttributeProto* getAttribute(const onnx::NodeProto& node, const std::string& name)
  {
    auto it = std::find_if(node.attribute().begin(), node.attribute().end(), [&](const onnx::AttributeProto& a){return a.name() == name;});
    return it == node.attribute().end() ? nullptr : &*it;
  }

  void ONNX::doAdd(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);
    // if(defaultOpsetVersion < 6) // allow axis, broadcast and consumed_inputs
    // else if(defaultOpsetVersion < 7) // allow axis and broadcast
    // else ASSERT(node.attribute().size() == 0);
    // TODO: Add-1 had up to three attributes (axis, broadcast and consumed_inputs)
    // TODO: Add-6 had up to two attributes (axis and broadcast)
    // TODO: Add-7, Add-13 and Add-14 do automatic broadcasting.
    ASSERT(node.attribute().size() == 0);
    const auto A = variables.find(node.input()[0]);
    const auto B = variables.find(node.input()[1]); // TODO: one of them may be a weight
    ASSERT(A != variables.end());
    if(B == variables.end())
    {
      const auto B = weights.find(node.input()[1]);
      ASSERT(B != weights.end());
      std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>();
      layer->weights.reshape({B->second.dims(0), B->second.dims(0)}); // TODO: identity
      for(unsigned int i = 0; i < B->second.dims(0); ++i)
        layer->weights(i, i) = 1.f;
      layer->biases.resize(B->second.dims(0));
      std::copy(B->second.begin(), B->second.end(), layer->biases.begin());
      layer->hasBiases = true;
      layer->activationId = ActivationFunctionId::linear;
      layer->nodes.emplace_back(layer.get());
      Node& n = layer->nodes.back();
      n.inputs.push_back(A->second);
      n.setDimensions();
      ASSERT(n.outputDimensions.size() == 1);
      n.outputs.emplace_back(layer.get(), 0, 0);
      variables.emplace(node.output()[0], n.outputs.front());
      layers.emplace_back(std::move(layer));
      return;
    }
    ASSERT(B != variables.end());
    ASSERT(variables.find(node.output()[0]) == variables.end());

    std::unique_ptr<AddLayer> layer = std::make_unique<AddLayer>();
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(A->second);
    n.inputs.push_back(B->second);
    // TODO: support broadcasting
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));

    // This operator can be used both to add variables and to add weights to variables. (adding weights to weights would be possible but is not supported)
  }

  void ONNX::doBatchNormalization(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 5);
    ASSERT(node.output().size() >= 1 && node.output().size() <= 5);
    const auto X = variables.find(node.input()[0]);
    const auto scale = weights.find(node.input()[1]);
    const auto B = weights.find(node.input()[2]);
    const auto mean = weights.find(node.input()[3]);
    const auto var = weights.find(node.input()[4]);
    ASSERT(X != variables.end());
    ASSERT(scale != weights.end());
    ASSERT(B != weights.end());
    ASSERT(mean != weights.end());
    ASSERT(var != weights.end());
    ASSERT(scale->second.dims() == B->second.dims());
    ASSERT(scale->second.dims() == mean->second.dims());
    ASSERT(scale->second.dims() == var->second.dims());
    ASSERT(scale->second.rank() == 1);
    // ASSERT(scale->second.dims(0) == ); // TODO: num of input channels
    // ignore momentum attribute (is 0.9f by default)
    const auto* attr = getAttribute(node, "epsilon");
    const float epsilon = attr ? attr->f() : 1e-5f;
    const std::size_t N = scale->second.dims(0);

    std::unique_ptr<BatchNormalizationLayer> layer = std::make_unique<BatchNormalizationLayer>();
    layer->axis = -1;
    layer->factor.resize(N);
    for(unsigned int i = 0; i < N; ++i)
      layer->factor[i] = 1.f / std::sqrt(var->second[i] + epsilon);
    for(unsigned int i = 0; i < N; ++i)
      layer->factor[i] *= scale->second[i];
    layer->offset.resize(N);
    for(unsigned int i = 0; i < N; ++i)
      layer->offset[i] = -mean->second[i] * layer->factor[i];
    for(unsigned int i = 0; i < N; ++i)
      layer->offset[i] += B->second[i];
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(X->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doCast(const onnx::NodeProto& node)
  {
    // This is according to opset version 13.
    ASSERT(node.input().size() == 1);
    ASSERT(node.output().size() == 1);
    const auto* toAttr = getAttribute(node, "to");
    ASSERT(toAttr);

    fprintf(stderr, "i: %s, o: %s, to: %ld\n", node.input()[0].c_str(), node.output()[0].c_str(), toAttr->i());

    const auto input = shapes.find(node.input()[0]);
    if(input == shapes.end())
      FAIL("Only shapes can be cast.");

    shapes.emplace(node.output()[0], input->second);
  }

  void ONNX::doClip(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() >= 1 && node.input().size() <= 3);
    ASSERT(node.output().size() == 1);
    const auto input = variables.find(node.input()[0]);
    ASSERT(input != variables.end());

    variables.emplace(node.output()[0], input->second); // TODO: actually do something (ReLU?)
  }

  void ONNX::doConcat(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() >= 1);
    ASSERT(node.output().size() == 1);
    const auto* axisAttr = getAttribute(node, "axis");
    ASSERT(axisAttr);

    if(const auto it = shapes.find(node.input()[0]); it != shapes.end())
    {
      shapes.emplace(node.output()[0], it->second); // TODO: actually concatenate with something
      return;
    }

    std::unique_ptr<ConcatenateLayer> layer = std::make_unique<ConcatenateLayer>();
    // TODO: axis assumes a different channel layout than we're actually using
    layer->axis = axisAttr->i();
    if(layer->axis == 0)
      FAIL("");
    if(layer->axis < 0)
      FAIL("");
    if(layer->axis == 1)
      layer->axis = 2;
    else if(layer->axis == 2)
      layer->axis = 0;
    else if(layer->axis == 3)
      layer->axis = 1;
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    for(const std::string& input : node.input())
    {
      const auto it = variables.find(input);
      ASSERT(it != variables.end());
      n.inputs.push_back(it->second);
    }
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doConv(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2 || node.input().size() == 3);
    ASSERT(node.output().size() == 1);
    auto X = variables.find(node.input()[0]);
    auto W = weights.find(node.input()[1]);
    auto B = node.input().size() > 2 ? weights.find(node.input()[2]) : weights.end();
    ASSERT(X != variables.end());
    ASSERT(W != weights.end());
    ASSERT(node.input().size() <= 2 || B != weights.end());
    const auto* autoPadAttr = getAttribute(node, "auto_pad");
    ASSERT(!autoPadAttr || autoPadAttr->s() == "NOTSET");
    const auto* dilationsAttr = getAttribute(node, "dilations");
    ASSERT(!dilationsAttr || dilationsAttr->ints().size() == 2);
    if(dilationsAttr && (dilationsAttr->ints()[0] != 1 || dilationsAttr->ints()[1] != 1))
      FAIL("Dilated Conv is not supported.");
    auto* groupAttr = getAttribute(node, "group");
    // ASSERT(!depthwise || groupAttr->i() == numOfInputChannels);
    const bool depthwise = groupAttr && groupAttr->i() > 1;
    const auto* kernelShapeAttr = getAttribute(node, "kernel_shape");
    ASSERT(kernelShapeAttr);
    ASSERT(kernelShapeAttr->ints().size() == 2);
    ASSERT(W->second.dims(2) == kernelShapeAttr->ints()[0]);
    ASSERT(W->second.dims(3) == kernelShapeAttr->ints()[1]);
    const auto* padsAttr = getAttribute(node, "pads");
    ASSERT(!padsAttr || padsAttr->ints().size() == 4);
    PaddingType padding = PaddingType::valid;
    if(padsAttr)
    {
      ASSERT(padsAttr->ints().size() == 4);
      if(std::any_of(padsAttr->ints().begin(), padsAttr->ints().end(), [](auto x){return x > 0;}))
        padding = PaddingType::same; // TODO: check if this is actually compatible to same padding
    }
    // ...
    auto* stridesAttr = getAttribute(node, "strides");
    ASSERT(!stridesAttr || stridesAttr->ints().size() == 2);

    // TODO: for now only support 2D convs (i.e. 3 input channels, 4 weight dimensions)

    if(depthwise)
    {
      std::unique_ptr<DepthwiseConv2DLayer> layer = std::make_unique<DepthwiseConv2DLayer>();
      layer->strides[0] = stridesAttr ? stridesAttr->ints()[0] : 1;
      layer->strides[1] = stridesAttr ? stridesAttr->ints()[1] : 1;
      layer->activationId = ActivationFunctionId::linear;
      layer->padding = padding;
      ASSERT(W->second.rank() == 4);
      ASSERT(W->second.dims(1) == 1);
      layer->weights.reshape(W->second.dims(2), W->second.dims(3), W->second.dims(0), W->second.dims(1));
      fprintf(stderr, "DConv %u %u %u %u\n", W->second.dims(0), W->second.dims(1), W->second.dims(2), W->second.dims(3));
      std::vector<unsigned int> index(4), i2(4);
      for(index[0] = 0; index[0] < W->second.dims(2); ++index[0])
        for(index[1] = 0; index[1] < W->second.dims(3); ++index[1])
          for(index[2] = 0; index[2] < W->second.dims(0); ++index[2])
            for(index[3] = 0; index[3] < W->second.dims(1); ++index[3])
            {
              i2[0] = index[2];
              i2[1] = index[3];
              i2[2] = index[0];
              i2[3] = index[1];
              layer->weights(index) = W->second(i2);
            }
      // std::copy(W->second.begin(), W->second.end(), layer->weights.begin()); // TODO: transpose weights
      if((layer->hasBiases = B != weights.end()))
      {
        ASSERT(B->second.rank() == 1);
        ASSERT(B->second.dims(0) == W->second.dims(0));
        layer->biases.resize(B->second.size());
        std::copy(B->second.begin(), B->second.end(), layer->biases.begin());
      }
      layer->nodes.emplace_back(layer.get());
      Node& n = layer->nodes.back();
      n.inputs.push_back(X->second);
      n.setDimensions();
      ASSERT(n.outputDimensions.size() == 1);
      n.outputs.emplace_back(layer.get(), 0, 0);
      variables.emplace(node.output()[0], n.outputs.front());
      layers.emplace_back(std::move(layer));
    }
    else
    {
      std::unique_ptr<Conv2DLayer> layer = std::make_unique<Conv2DLayer>();
      layer->strides[0] = stridesAttr ? stridesAttr->ints()[0] : 1;
      layer->strides[1] = stridesAttr ? stridesAttr->ints()[1] : 1;
      layer->activationId = ActivationFunctionId::linear;
      layer->padding = padding;
      ASSERT(W->second.rank() == 4);
      fprintf(stderr, "Conv %u %u %u %u\n", W->second.dims(0), W->second.dims(1), W->second.dims(2), W->second.dims(3));
      layer->weights.reshape(W->second.dims(2), W->second.dims(3), W->second.dims(1), W->second.dims(0));
      std::vector<unsigned int> index(4), i2(4);
      for(index[0] = 0; index[0] < W->second.dims(2); ++index[0])
        for(index[1] = 0; index[1] < W->second.dims(3); ++index[1])
          for(index[2] = 0; index[2] < W->second.dims(1); ++index[2])
            for(index[3] = 0; index[3] < W->second.dims(0); ++index[3])
            {
              i2[0] = index[3];
              i2[1] = index[2];
              i2[2] = index[0];
              i2[3] = index[1];
              layer->weights(index) = W->second(i2);
            }
      // layer->weights[I, J, K, L] = W->second[L, K, I, J]
      // std::copy(W->second.begin(), W->second.end(), layer->weights.begin()); // TODO: transpose weights
      if((layer->hasBiases = B != weights.end()))
      {
        ASSERT(B->second.rank() == 1);
        ASSERT(B->second.dims(0) == W->second.dims(0));
        layer->biases.resize(B->second.size());
        std::copy(B->second.begin(), B->second.end(), layer->biases.begin());
      }
      layer->nodes.emplace_back(layer.get());
      Node& n = layer->nodes.back();
      n.inputs.push_back(X->second);
      n.setDimensions();
      ASSERT(n.outputDimensions.size() == 1);
      n.outputs.emplace_back(layer.get(), 0, 0);
      variables.emplace(node.output()[0], n.outputs.front());
      layers.emplace_back(std::move(layer));
    }
  }

  void ONNX::doGather(const onnx::NodeProto& node)
  {
    // this is according to version 13
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);
    const auto* axisAttr = getAttribute(node, "axis");
    const int axis = axisAttr ? axisAttr->i() : 0; // TODO: The axis is what ONNX thinks.
    const auto data = shapes.find(node.input()[0]);
    if(data == shapes.end())
      FAIL("Can only Gather shapes.");

    shapes[node.output()[0]] = data->second;
  }

  void ONNX::doMatMul(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);
    auto X = variables.find(node.input()[0]);
    ASSERT(X != variables.end());
    auto Y = weights.find(node.input()[1]);
    ASSERT(Y != weights.end());

    std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>();
    layer->weights = Y->second; // .reshape(Y->second.dims());
    layer->hasBiases = false;
    layer->activationId = ActivationFunctionId::linear;
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(X->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doMaxPool(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 1);
    ASSERT(node.output().size() >= 1 && node.output().size() <= 2);
    auto X = variables.find(node.input()[0]);
    ASSERT(X != variables.end());
    const auto& inDims = X->second.layer->nodes[X->second.nodeIndex].outputDimensions[X->second.tensorIndex];
    if(inDims.size() != 3)
      FAIL("Can only pool 2D+channel tensors.");
    const auto* dilationsAttr = getAttribute(node, "dilations");
    ASSERT(!dilationsAttr || dilationsAttr->ints().size() == 2);
    if(dilationsAttr && (dilationsAttr->ints()[0] != 1 || dilationsAttr->ints()[1] != 1))
      FAIL("Dilated MaxPool is not supported.");
    const auto* kernelShapeAttr = getAttribute(node, "kernel_shape");
    ASSERT(kernelShapeAttr);
    ASSERT(kernelShapeAttr->ints().size() == 2);
    const auto* stridesAttr = getAttribute(node, "strides");
    ASSERT(!stridesAttr || stridesAttr->ints().size() == 2);
    // TODO: handle padding attributes
    const auto* padsAttr = getAttribute(node, "pads");
    ASSERT(!padsAttr || padsAttr->ints().size() == 4);
    PaddingType padding = PaddingType::valid;
    if(padsAttr)
    {
      ASSERT(padsAttr->ints().size() == 4);
      if(std::any_of(padsAttr->ints().begin(), padsAttr->ints().end(), [](auto x){return x > 0;}))
        padding = PaddingType::same; // TODO: check if this is actually compatible to same padding
    }

    std::unique_ptr<Pooling2DLayer> layer = std::make_unique<Pooling2DLayer>(LayerType::maxPooling2D, PoolingMethod::max);
    layer->method = PoolingMethod::max;
    layer->padding = padding;
    layer->kernelSize[0] = kernelShapeAttr->ints()[0];
    layer->kernelSize[1] = kernelShapeAttr->ints()[1];
    layer->strides[0] = stridesAttr ? stridesAttr->ints()[0] : 1;
    layer->strides[1] = stridesAttr ? stridesAttr->ints()[1] : 1;
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(X->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doMul(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);

    // TODO: do a batch normalization
  }

  void ONNX::doPad(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() >= 2 && node.input().size() <= 3);
    ASSERT(node.output().size() == 1);
    const auto* modeAttr = getAttribute(node, "mode");
    if(modeAttr && modeAttr->s() != "constant")
      FAIL("Non-constant Pad is not supported.");
    const auto data = variables.find(node.input()[0]);
    const auto pads = shapes.find(node.input()[1]);
    ASSERT(data != variables.end());
    ASSERT(pads != shapes.end());
    ASSERT(pads->second.size() == 8);
    ASSERT(pads->second[0] == 0);
    ASSERT(pads->second[1] == 0);
    ASSERT(pads->second[4] == 0);
    ASSERT(pads->second[5] == 0);
    // TODO: assert 2D+channel input
    // const auto constantValue = weights.find(node.input()[2]);

    std::unique_ptr<ZeroPadding2DLayer> layer = std::make_unique<ZeroPadding2DLayer>();
    layer->padding[ZeroPadding2DLayer::TOP] = pads->second[2];
    layer->padding[ZeroPadding2DLayer::BOTTOM] = pads->second[6];
    layer->padding[ZeroPadding2DLayer::LEFT] = pads->second[3];
    layer->padding[ZeroPadding2DLayer::RIGHT] = pads->second[7];
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(data->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doRelu(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 1);
    ASSERT(node.output().size() == 1);
    ASSERT(node.attribute().size() == 0);
    auto X = variables.find(node.input()[0]);
    ASSERT(X != variables.end());

    std::unique_ptr<ReluLayer> layer = std::make_unique<ReluLayer>();
    layer->maxValue = std::numeric_limits<float>::max();
    layer->negativeSlope = 0.f;
    layer->threshold = 0;
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(X->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doReshape(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);
    ASSERT(node.attribute().size() == 0);
    const auto data = variables.find(node.input()[0]);
    ASSERT(data != variables.end());
    const auto shape = shapes.find(node.input()[1]);
    ASSERT(shape != shapes.end());

    ASSERT(shape->second.size() > 0); // if we're dealing with a scalar or a [1, 1, 1, 1, 1] tensor this could actually be the case (?)
    ASSERT(shape->second[0] == -1 || shape->second[0] == 1); // TODO: 0 also possible as "copy from input"?

    if(shape->second.size() == 4)
    {
      // TODO: sometimes, reshape is used in place of a transpose if the number of channels is 1
      variables.emplace(node.output()[0], data->second); // TODO change channel order tag
      return;
    }

    std::unique_ptr<ReshapeLayer> layer = std::make_unique<ReshapeLayer>();
    layer->dimensions.resize(shape->second.size() - 1);
    for(std::size_t i = 0; i < layer->dimensions.size(); ++i)
    {
      // TODO: check -1 and stuff
      layer->dimensions[i] = static_cast<unsigned>(shape->second[i + 1]);
    }
    // TODO: since we may be flattening something that ONNX thinks is a CHW tensor, the weights of subsequent dense layers need to be adjusted
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(data->second);
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));
  }

  void ONNX::doShape(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 1);
    ASSERT(node.output().size() == 1);
    ASSERT(node.attribute().size() == 0);
    const auto data = variables.find(node.input()[0]);
    ASSERT(data != variables.end());
    const auto& shape = data->second.layer->nodes[data->second.nodeIndex].outputDimensions[data->second.tensorIndex];
    // TODO: This shape is a HWC tensor at most, however, ONNX may think it's a NCHW tensor
    // In general we must put into "shapes" what ONNX thinks is the shape (i.e. prepend a batch axis and move channels front)
    //
    Tensor<int64_t, 1> otherShape({1 + static_cast<unsigned int>(shape.size())});
    otherShape[0] = 1;
    for(std::size_t i = 0; i < shape.size(); ++i)
      otherShape[i + 1] = shape[i];
    ASSERT(shapes.find(node.output()[0]) == shapes.end());
    shapes.emplace(node.output()[0], otherShape);
  }

  void ONNX::doSlice(const onnx::NodeProto& node)
  {
    // if(defaultOpsetVersion ...)
    // sorry, we can only slice shapes at the moment
    ASSERT(node.input().size() >= 3 && node.input().size() <= 5);
    ASSERT(node.output().size() == 1);

    const auto data = shapes.find(node.input()[0]);
    ASSERT(data != shapes.end());

    shapes.emplace(node.output()[0], data->second); // TODO: actually do something
  }

  void ONNX::doSub(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 2);
    ASSERT(node.output().size() == 1);
    ASSERT(node.attribute().size() == 0);
    const auto A = variables.find(node.input()[0]);
    const auto B = weights.find(node.input()[1]); // TODO: this could also be a variable
    ASSERT(A != variables.end());
    ASSERT(B != weights.end());
    ASSERT(variables.find(node.output()[0]) == variables.end());

    std::unique_ptr<SubtractLayer> layer = std::make_unique<SubtractLayer>();
    layer->nodes.emplace_back(layer.get());
    Node& n = layer->nodes.back();
    n.inputs.push_back(A->second);
    n.inputs.push_back(A->second); // TODO: actually, we want to subtract B, but B is a weight
    // TODO: support broadcasting
    n.setDimensions();
    ASSERT(n.outputDimensions.size() == 1);
    n.outputs.emplace_back(layer.get(), 0, 0);
    variables.emplace(node.output()[0], n.outputs.front());
    layers.emplace_back(std::move(layer));

    // This operator can be used both to add variables and to add weights to variables. (adding weights to weights would be possible but is not supported)
  }

  void ONNX::doTranspose(const onnx::NodeProto& node)
  {
    ASSERT(node.input().size() == 1);
    ASSERT(node.output().size() == 1);
    ASSERT(node.attribute().size() == 1);
    ASSERT(node.attribute()[0].name() == "perm");
    ASSERT(node.attribute()[0].type() == onnx::AttributeProto::INTS);
    auto data = variables.find(node.input()[0]);
    ASSERT(data != variables.end());
    const auto& perm = node.attribute()[0].ints();
    if(perm.size() != 4 || perm[0] != 0)
      FAIL("The transpose operation is only implemented for NHWC<->NCHW.");
    // 0 3 1 2: NHWC -> NCHW (usually the first thing a model does)
    // 0 2 3 1: NCHW -> NHWC (usually the last thing a model does)
    // TODO: do nothing really
    ASSERT(variables.find(node.output()[0]) == variables.end());
    variables.emplace(node.output()[0], data->second); // TODO change channel order tag
  }

  /*
  static const Operator allOperators[] =
  {
    // {"Add", 1, doAdd1},
    // {"Add", 6, doAdd6},
    {"Add", 7, doAdd13}, // TODO
    {"Add", 13, doAdd13},
    // {"Add", 14, doAdd14},
    // {"BatchNormalization", 1, doBatchNormalization1},
    // {"BatchNormalization", 6, doBatchNormalization6},
    // {"BatchNormalization", 7, doBatchNormalization7},
    {"BatchNormalization", 9, doBatchNormalization9},
    // {"BatchNormalization", 14, doBatchNormalization14},
    // {"BatchNormalization", 15, doBatchNormalization15},
    // {"Clip", 1, doClip1}, // min&max are float-attributes (with default lowest:max)
    // {"Clip", 6, doClip6}, // same as version 1 except for consumed_inputs legacy attribute
    {"Clip", 11, doClip11}, // min&max are inputs now (empty shape however)
    // {"Clip", 12, doClip12}, // same as version 11 but with more allowed datatypes
    // {"Clip", 13, doClip13}, // same as version 12 but even more datatypes + min&max are specified as non-differentiable
    {"Concat", 1, doConcat1},
    {"Concat", 4, doConcat1},
    {"Concat", 11, doConcat1},
    {"Concat", 13, doConcat1},
    {"Conv", 1, doConv11}, // TODO
    {"Conv", 11, doConv11},
    // {"MaxPool", 1, doMaxPool1},
    {"MaxPool", 8, doMaxPool12}, // TODO
    // {"MaxPool", 10, doMaxPool10},
    // {"MaxPool", 11, doMaxPool11},
    {"MaxPool", 12, doMaxPool12},
    {"Mul", 7, doSub7}, // TODO
    // {"Pad", 1, doPad1},
    // {"Pad", 2, doPad2},
    // {"Pad", 11, doPad11},
    {"Pad", 13, doPad13},
    // {"Pad", 18, doPad18},
    // {"Pad", 19, doPad19},
    // {"Relu", 1, doRelu1},
    {"Relu", 6, doRelu13}, // basically the same
    {"Relu", 13, doRelu13},
    // {"Relu", 14, doRelu14},
    // {"Reshape", 1, doReshape1}, // the shape is an attribute in this version
    {"Reshape", 5, doReshape5}, // the shape is an int64 input in this version
    // {"Reshape", 13, doReshape13}, // the shape is an int64 input in this version
    // Reshape-14 and -19 have an attribute "allowzero"
    {"Slice", 1, doSlice1}, // TODO
    {"Shape", 1, doShape13}, // TODO
    {"Shape", 13, doShape13},
    // {"Shape", 15, doShape15},
    // {"Sub", 1, doSub1},
    // {"Sub", 6, doSub6},
    {"Sub", 7, doSub7},
    // {"Sub", 13, doSub13},
    // {"Sub", 14, doSub14},
    {"Transpose", 1, doTranspose13}, // basically the same as Transpose-13, just not differentiable
    {"Transpose", 13, doTranspose13},
  };
  */


  void ONNX::read(const std::string &file)
  {
    weights.clear();
    shapes.clear();
    variables.clear();

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
    ASSERT(model.has_graph());
    const auto& graph = model.graph();
    std::int64_t defaultOpsetVersion = -1;
    for(const onnx::OperatorSetIdProto& opset : model.opset_import())
      if(opset.domain().empty())
      {
        defaultOpsetVersion = opset.version();
        break;
      }
    if(defaultOpsetVersion < 0)
      FAIL("No version defined for the default operator set.");
    if(defaultOpsetVersion > maxDefaultOpsetVersion)
      FAIL("Version " << defaultOpsetVersion << " of the default operator set is not supported.");

    // ignore metadata_props
    if(!model.training_info().empty())
      FAIL("Non-empty training info is not supported.");
    if(!model.functions().empty())
      FAIL("Non-empty functions is not supported.");

    std::unordered_map<std::string, void (ONNX::*)(const onnx::NodeProto&)> operators;
    operators.emplace("Add", &ONNX::doAdd);
    operators.emplace("BatchNormalization", &ONNX::doBatchNormalization);
    operators.emplace("Cast", &ONNX::doCast);
    operators.emplace("Clip", &ONNX::doClip);
    operators.emplace("Concat", &ONNX::doConcat);
    operators.emplace("Conv", &ONNX::doConv);
    operators.emplace("Gather", &ONNX::doGather);
    operators.emplace("MatMul", &ONNX::doMatMul);
    operators.emplace("MaxPool", &ONNX::doMaxPool);
    operators.emplace("Mul", &ONNX::doMul);
    operators.emplace("Pad", &ONNX::doPad);
    operators.emplace("Relu", &ONNX::doRelu);
    operators.emplace("Reshape", &ONNX::doReshape);
    operators.emplace("Shape", &ONNX::doShape);
    operators.emplace("Slice", &ONNX::doSlice);
    operators.emplace("Sub", &ONNX::doSub);
    operators.emplace("Transpose", &ONNX::doTranspose);

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
      // TODO: INT32 for indices
      /*
      else
        FAIL("Only FLOAT and INT64 initializers are supported (but not " << initializer.data_type() << " for " << initializer.name() << ").");
      */
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
      // ignore node.name
      // ignore node.doc_string

      if(!node.domain().empty())
        FAIL("Only the default opset (and not even that) is supported yet, but some node tries to use " << node.domain() << ".");

      fprintf(stderr, "%s\n", node.op_type().c_str());

      auto op = operators.find(node.op_type());
      if(op == operators.end())
        FAIL("Unknown operator " << node.op_type() << " (in default opset version " << defaultOpsetVersion << ").");
      (this->*op->second)(node);

      // TODO: can we check the shape of generated variables here?
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
