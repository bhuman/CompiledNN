/**
 * Contains structs that define a neural network model as well as a method to
 * load such models from a file.
 *
 * @author Felix Thielke
 */

#pragma once

#include "Tensor.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

class In;

namespace NeuralNetwork
{
  enum class LayerType
  {
    input,
    dense,
    activation,
    conv1D,
    conv2D,
    separableConv2D,
    depthwiseConv2D,
    maxPooling1D,
    maxPooling2D,
    averagePooling1D,
    averagePooling2D,
    globalMaxPooling2D,
    globalAveragePooling2D,
    batchNormalization,
    dropout,
    reshape,
    flatten,
    cropping2D,
    upSampling2D,
    zeroPadding1D,
    zeroPadding2D,
    concatenate,
    average,
    maximum,
    minimum,
    add,
    subtract,
    multiply,
    relu,
    softmax,
    leakyRelu,
    elu,
    thresholdedRelu
  };

  struct Layer;

  /**
   * A struct that describes the location of a tensor in a network.
   */
  struct TensorLocation final
  {
    const Layer* layer;
    unsigned int nodeIndex;
    unsigned int tensorIndex;

    TensorLocation(const Layer* layer, unsigned int nodeIndex, unsigned int tensorIndex) : layer(layer), nodeIndex(nodeIndex), tensorIndex(tensorIndex) {}

    bool operator==(const TensorLocation& other) const
    {
      return layer == other.layer && nodeIndex == other.nodeIndex && tensorIndex == other.tensorIndex;
    }
  };

  /**
   * A struct that describes a node in a network, i.e. an instance of a layer with known inputs and outputs.
   */
  struct Node final
  {
    const Layer* const layer;
    std::vector<TensorLocation> inputs;
    std::vector<TensorLocation> outputs;
    std::vector<std::vector<unsigned int>> inputDimensions;
    std::vector<std::vector<unsigned int>> outputDimensions;

    void setDimensions();

    Node(const Layer* layer) : layer(layer) {}
  };

  struct Layer
  {
    const LayerType type;
    std::vector<Node> nodes;

    Layer() = delete;
    virtual ~Layer() = default;

    virtual void calcOutputDimensions(Node& node) const = 0;
  protected:
    Layer(const LayerType type) : type(type) {}
  };

  /**
   * A struct that describes a neural network model.
   */
  struct Model final
  {
  private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<bool> uint8Inputs;
    std::vector<TensorLocation> inputs;
    std::vector<TensorLocation> outputs;

  public:
    Model() = default;
    Model(const std::string& file) : Model() { load(file); }
    ~Model()
    {
      layers.clear();
    }

    /**
     * Returns a const reference to a vector of the layers that make up this NN.
     */
    inline const std::vector<std::unique_ptr<Layer>>& getLayers() const { return layers; }

    /**
     * Returns a const reference to a vector of the inputs that this NN has.
     */
    inline const std::vector<TensorLocation>& getInputs() const { return inputs; }

    /**
     * Returns a const reference to a vector of the outputs that this NN has.
     */
    inline const std::vector<TensorLocation>& getOutputs() const { return outputs; }

    /*
     * Indicates that an input with a specified index should be interpreted as a tensor of unsigned chars.
     */
    void setInputUInt8(std::size_t index);

    /*
     * Checks whether an input with a specified index should be interpreted as a tensor of unsigned chars.
     */
    bool isInputUInt8(std::size_t index) const;

    /**
     * Removes all layers from this model.
     */
    void clear() { layers.clear(); inputs.clear(); outputs.clear(); uint8Inputs.clear(); }

    /**
     * Loads a neural network model from the given file.
     */
    void load(const std::string& file);
  };

  enum class ActivationFunctionId
  {
    linear,
    relu,
    sigmoid,
    tanH,
    hardSigmoid,
    softmax,
    elu,
    selu,
    exponential,
    softsign
  };

  enum class PaddingType
  {
    valid,
    same
  };

  enum class InterpolationMethod
  {
    nearest,
    bilinear
  };

  enum class PoolingMethod
  {
    average,
    max
  };

  struct InputLayer : Layer
  {
    std::vector<unsigned int> dimensions;

    InputLayer() : Layer(LayerType::input) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct DenseLayer : Layer
  {
    Tensor<float, 1> weights;
    std::vector<float> biases;
    bool hasBiases;
    ActivationFunctionId activationId;

    DenseLayer() : Layer(LayerType::dense) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ActivationLayer : Layer
  {
    ActivationFunctionId activationId;

    ActivationLayer() : Layer(LayerType::activation) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct Conv1DLayer : Layer
  {
    unsigned int stride;
    Tensor<float, 1> weights;
    std::vector<float> biases;
    bool hasBiases;
    ActivationFunctionId activationId;
    PaddingType padding;

    Conv1DLayer() : Layer(LayerType::conv1D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct Conv2DLayer : Layer
  {
    std::array<unsigned int, 2> strides;
    Tensor<float, 1> weights;
    std::vector<float> biases;
    bool hasBiases;
    ActivationFunctionId activationId;
    PaddingType padding;

    Conv2DLayer() : Layer(LayerType::conv2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct SeparableConv2DLayer : Layer
  {
    std::array<unsigned int, 2> strides;
    Tensor<float, 1> depthwiseWeights;
    Tensor<float, 1> pointwiseWeights;
    std::vector<float> biases;
    bool hasBiases;
    ActivationFunctionId activationId;
    PaddingType padding;

    SeparableConv2DLayer() : Layer(LayerType::separableConv2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct DepthwiseConv2DLayer : Layer
  {
    std::array<unsigned int, 2> strides;
    Tensor<float, 1> weights;
    std::vector<float> biases;
    bool hasBiases;
    ActivationFunctionId activationId;
    PaddingType padding;

    DepthwiseConv2DLayer() : Layer(LayerType::depthwiseConv2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct Pooling1DLayer : Layer
  {
    PoolingMethod method;
    PaddingType padding;
    unsigned int kernelSize;
    unsigned int stride;

    Pooling1DLayer(LayerType type, PoolingMethod method) : Layer(type), method(method) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct Pooling2DLayer : Layer
  {
    PoolingMethod method;
    PaddingType padding;
    std::array<unsigned int, 2> kernelSize;
    std::array<unsigned int, 2> strides;

    Pooling2DLayer(LayerType type, PoolingMethod method) : Layer(type), method(method) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct MaxPooling1DLayer : Pooling1DLayer
  {
    MaxPooling1DLayer() : Pooling1DLayer(LayerType::maxPooling1D, PoolingMethod::max) {}
  };

  struct MaxPooling2DLayer : Pooling2DLayer
  {
    MaxPooling2DLayer() : Pooling2DLayer(LayerType::maxPooling2D, PoolingMethod::max) {}
  };

  struct AveragePooling1DLayer : Pooling1DLayer
  {
    AveragePooling1DLayer() : Pooling1DLayer(LayerType::averagePooling1D, PoolingMethod::average) {}
  };

  struct AveragePooling2DLayer : Pooling2DLayer
  {
    AveragePooling2DLayer() : Pooling2DLayer(LayerType::averagePooling2D, PoolingMethod::average) {}
  };

  struct GlobalPooling2DLayer : Layer
  {
    PoolingMethod method;

    GlobalPooling2DLayer(LayerType type, PoolingMethod method) : Layer(type), method(method) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct GlobalMaxPooling2DLayer : GlobalPooling2DLayer
  {
    GlobalMaxPooling2DLayer() : GlobalPooling2DLayer(LayerType::globalMaxPooling2D, PoolingMethod::max) {}
  };

  struct GlobalAveragePooling2DLayer : GlobalPooling2DLayer
  {
    GlobalAveragePooling2DLayer() : GlobalPooling2DLayer(LayerType::globalAveragePooling2D, PoolingMethod::average) {}
  };

  struct BatchNormalizationLayer : Layer
  {
    int axis;
    std::vector<float> factor;
    std::vector<float> offset;

    BatchNormalizationLayer() : Layer(LayerType::batchNormalization) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct DropoutLayer : Layer
  {
    DropoutLayer() : Layer(LayerType::dropout) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ReshapeLayer : Layer
  {
    std::vector<unsigned int> dimensions;

    ReshapeLayer() : Layer(LayerType::reshape) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct FlattenLayer : Layer
  {
    FlattenLayer() : Layer(LayerType::flatten) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct Cropping2DLayer : Layer
  {
    enum Side
    {
      TOP,
      BOTTOM,
      LEFT,
      RIGHT,
    };
    std::array<unsigned int, 4> cropping;

    Cropping2DLayer() : Layer(LayerType::cropping2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct UpSampling2DLayer : Layer
  {
    std::array<unsigned int, 2> size;
    InterpolationMethod interpolation;

    UpSampling2DLayer() : Layer(LayerType::upSampling2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ZeroPadding1DLayer : Layer
  {
    enum Side
    {
      LEFT,
      RIGHT,
    };
    std::array<unsigned int, 2> padding;

    ZeroPadding1DLayer() : Layer(LayerType::zeroPadding1D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ZeroPadding2DLayer : Layer
  {
    enum Side
    {
      TOP,
      BOTTOM,
      LEFT,
      RIGHT,
    };
    std::array<unsigned int, 4> padding;

    ZeroPadding2DLayer() : Layer(LayerType::zeroPadding2D) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ConcatenateLayer : Layer
  {
    int axis;

    ConcatenateLayer() : Layer(LayerType::concatenate) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct AverageLayer : Layer
  {
    AverageLayer() : Layer(LayerType::average) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct MaximumLayer : Layer
  {
    MaximumLayer() : Layer(LayerType::maximum) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct MinimumLayer : Layer
  {
    MinimumLayer() : Layer(LayerType::minimum) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct AddLayer : Layer
  {
    AddLayer() : Layer(LayerType::add) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct SubtractLayer : Layer
  {
    SubtractLayer() : Layer(LayerType::subtract) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct MultiplyLayer : Layer
  {
    MultiplyLayer() : Layer(LayerType::multiply) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ReluLayer : Layer
  {
    float maxValue;
    float negativeSlope;
    float threshold;

    ReluLayer() : Layer(LayerType::relu) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct SoftmaxLayer : Layer
  {
    int axis;

    SoftmaxLayer() : Layer(LayerType::softmax) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct LeakyReluLayer : Layer
  {
    float alpha;

    LeakyReluLayer() : Layer(LayerType::leakyRelu) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct EluLayer : Layer
  {
    float alpha;

    EluLayer() : Layer(LayerType::elu) {}

    void calcOutputDimensions(Node& node) const override;
  };

  struct ThresholdedReluLayer : Layer
  {
    float theta;

    ThresholdedReluLayer() : Layer(LayerType::thresholdedRelu) {}

    void calcOutputDimensions(Node& node) const override;
  };
}
