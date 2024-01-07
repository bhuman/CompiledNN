/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"
#include "BatchNormalization.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct DConv2DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        const BatchNormalizationCompiler::Parameters* batchNormalization = nullptr;
        const Tensor<float, 1>* weights;
        const std::vector<float>* biases = nullptr;
        ActivationFunctionDescriptor postActivation;
        std::array<unsigned int, 2> strides;

        bool operator==(const Parameters& other) const
        {
          return batchNormalization == other.batchNormalization &&
            weights == other.weights &&
            biases == other.biases &&
            strides == other.strides &&
            postActivation == other.postActivation;
        }
      };
      const Parameters p;

      DConv2DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override
      {
        return p.strides[0] >= p.weights->dims(0) && p.strides[1] >= p.weights->dims(1) && p.weights->dims(3) <= 1;
      }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 3);
        return {{(inputDimensions[0] - p.weights->dims(0) + p.strides[0]) / p.strides[0], (inputDimensions[1] - p.weights->dims(1) + p.strides[1]) / p.strides[1], inputDimensions[2] * p.weights->dims(3)}};
      }

    private:
      mutable unsigned int biasOffset = 0;
      unsigned int outputBatchSize = 0;

      void compileFilter(x86::Assembler& a, const bool inputAligned, const unsigned int remainingOutputs, const unsigned int remainingInput, const bool lastFilter = false) const;
      void compileOutputBatch(x86::Assembler& a, ActivationFunctionHandler& afHandler, const unsigned int inputWidth, const unsigned int remainingOutputs) const;
      void compileSimpleConvolution(x86::Assembler& a, ActivationFunctionHandler& afHandler, const unsigned int inputWidth, const unsigned int outputHeight, const unsigned int outputWidth) const;
    };
  }
}
